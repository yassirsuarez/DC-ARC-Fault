#!/usr/bin/env python3
"""
generate_calibration_dataset_hydra.py
======================================
Calibration dataset per Hydra + Ridge (STM32 / ST Edge AI)

FIX: aggiunto 'input_diff' — Hydra ONNX richiede DUE input:
  - 'input':      segnale originale  (N, 1, 1000)
  - 'input_diff': differenza prima   (N, 1, 999)

Senza 'input_diff' ST Edge AI dà:
  "cannot select an axis to squeeze out which has size not equal to one"

Input:  arc_dataset_new.npz
Output: calibration_hydra.npz
"""

import argparse
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

RAND_STATE  = 42
N_PER_CLASS = 200


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--out", default="./calibration_hydra")
    parser.add_argument("--n-per-class", type=int, default=N_PER_CLASS)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ── LOAD DATA ──
    data = np.load(args.dataset)
    X    = data["X"].astype(np.float32)   # (N, T)
    y    = data["y"]

    log.info("Dataset: %s", X.shape)

    # ── BALANCED SAMPLING ──
    rng  = np.random.default_rng(RAND_STATE)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    n0 = min(args.n_per_class, len(idx0))
    n1 = min(args.n_per_class, len(idx1))

    if n0 == 0 or n1 == 0:
        raise ValueError("Una delle classi è vuota nel dataset!")

    idx0 = rng.choice(idx0, n0, replace=False)
    idx1 = rng.choice(idx1, n1, replace=False)
    idx  = np.concatenate([idx0, idx1])
    rng.shuffle(idx)

    X_cal = X[idx]   # (N, T) — es. (400, 1000)

    # ── PREPARA I DUE INPUT RICHIESTI DA HYDRA ──
    # Input 1: segnale originale → (N, 1, T)
    X_input = X_cal[:, np.newaxis, :].astype(np.float32)   # (400, 1, 1000)

    # Input 2: differenza prima (X[t+1] - X[t]) → (N, 1, T-1)
    # FIX: questo era mancante e causava l'errore "cannot select an axis to squeeze"
    X_diff = np.diff(X_cal, axis=-1).astype(np.float32)    # (400, 999)
    X_diff = X_diff[:, np.newaxis, :]                       # (400, 1, 999)

    log.info("input shape:      %s", X_input.shape)
    log.info("input_diff shape: %s", X_diff.shape)

    # ── SAVE ──
    out_path = os.path.join(args.out, "calibration_hydra.npz")
    np.savez(
        out_path,
        input=X_input,     # chiave 'input'      → segnale grezzo
        input_diff=X_diff, # chiave 'input_diff' → differenza prima (FIX)
    )

    log.info("✔ Saved: %s", out_path)

    # ── DEBUG ──
    loaded = np.load(out_path)
    print("\nKeys:", loaded.files)
    print("input shape:     ", loaded["input"].shape)
    print("input_diff shape:", loaded["input_diff"].shape)
    print("\nCome usare in ST Edge AI:")
    print("  1. Importa hydra.onnx  (NON mrh_ridge.onnx)")
    print("  2. Seleziona quantizzazione INT8")
    print("  3. Carica:", out_path)
    print("  4. Avvia la quantizzazione")


if __name__ == "__main__":
    main()