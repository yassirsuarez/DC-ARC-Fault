#!/usr/bin/env python3
"""
generate_calibration_hydra.py
=============================
Genera il dataset di calibrazione per la quantizzazione INT8 del modello
Hydra (hydra.onnx) tramite ST Edge AI / X-CUBE-AI.

IMPORTANTE — cosa quantizzare e cosa NO:
  ✅ hydra.onnx          → quantizza con ST Edge AI (input: segnale grezzo)
  ❌ mrh_ridge.onnx      → NON quantizzare — usa mrh_ridge_weights.h in C
  ❌ ridge.onnx          → NON quantizzare — usa ridge_weights.h in C

Hydra richiede DUE input nel file .npz:
  1. 'input':      segnale originale  (batch, 1, n_timepoints)
  2. 'input_diff': differenza prima   (batch, 1, n_timepoints-1)

I nomi delle chiavi DEVONO corrispondere esattamente ai nomi
definiti nell'ONNX durante l'export (input_names=["input","input_diff"]).

Uso:
    python generate_calibration_hydra.py <arc_dataset_new.npz>
    python generate_calibration_hydra.py <arc_dataset_new.npz> --n-per-class 200
    python generate_calibration_hydra.py <arc_dataset_new.npz> --out <cartella>

Requisiti:
    pip install numpy
"""

import argparse
import logging
import os
import sys
import numpy as np

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ── parametri default ─────────────────────────────────────────────────────────
N_PER_CLASS = 100    # 100 arco + 100 no-arco = 200 totali
                     # ST Edge AI suggerisce almeno 100-200 campioni totali
RAND_STATE  = 42


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Genera dataset di calibrazione multi-input per Hydra/MRH.\n"
            "Quantizza SOLO hydra.onnx — NON mrh_ridge.onnx."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        help="Percorso al file arc_dataset_new.npz",
    )
    parser.add_argument(
        "--out", "-o",
        default="./calibration_hydra",
        help="Cartella di output (default: ./calibration_hydra)",
    )
    parser.add_argument(
        "--n-per-class",
        type=int,
        default=N_PER_CLASS,
        help=f"Numero di campioni per classe (default: {N_PER_CLASS})",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.dataset):
        log.error("File non trovato: %s", args.dataset)
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)

    # ── 1. Carica dataset ─────────────────────────────────────────────────────
    log.info("Caricamento dataset: %s", args.dataset)
    data = np.load(args.dataset)
    X    = data["X"]   # shape (N, n_timepoints)  float32
    y    = data["y"]   # shape (N,)               int32

    n_timepoints = X.shape[1]
    log.info("  X shape:       %s  dtype=%s", X.shape, X.dtype)
    log.info("  y shape:       %s  dtype=%s", y.shape, y.dtype)
    log.info("  n_timepoints:  %d", n_timepoints)
    log.info("  label=0:       %d  (no arco)", int((y == 0).sum()))
    log.info("  label=1:       %d  (arco)",    int((y == 1).sum()))

    # ── 2. Campionamento bilanciato ───────────────────────────────────────────
    rng  = np.random.default_rng(RAND_STATE)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    n0 = min(args.n_per_class, len(idx0))
    n1 = min(args.n_per_class, len(idx1))

    if n0 < args.n_per_class:
        log.warning("  label=0: richiesti %d ma disponibili solo %d",
                    args.n_per_class, n0)
    if n1 < args.n_per_class:
        log.warning("  label=1: richiesti %d ma disponibili solo %d",
                    args.n_per_class, n1)

    idx0_cal = rng.choice(idx0, size=n0, replace=False)
    idx1_cal = rng.choice(idx1, size=n1, replace=False)
    idx_cal  = np.concatenate([idx0_cal, idx1_cal])
    rng.shuffle(idx_cal)

    X_cal = X[idx_cal].astype(np.float32)   # shape (N_tot, n_timepoints)
    y_cal = y[idx_cal]

    log.info("")
    log.info("  Campioni selezionati:")
    log.info("    label=0 (no arco): %d", n0)
    log.info("    label=1 (arco):    %d", n1)
    log.info("    Totale:            %d", len(idx_cal))

    # ── 3. Statistiche ────────────────────────────────────────────────────────
    log.info("")
    log.info("  Statistiche X_cal:")
    log.info("    min:  %.4f", float(X_cal.min()))
    log.info("    max:  %.4f", float(X_cal.max()))
    log.info("    mean: %.4f", float(X_cal.mean()))
    log.info("    std:  %.4f", float(X_cal.std()))

    # ── 4. Prepara i due input richiesti da Hydra ─────────────────────────────
    #
    # Input 1: segnale originale → shape (N, 1, n_timepoints)
    # Corrisponde al nome 'input' definito nell'export ONNX di Hydra
    # input principale
    X_input = X_cal[:, np.newaxis, :]              # (N,1,1000)
    X_input = X_input[:, np.newaxis, :]            # (N,1,1,1000)

    # Input 2: differenza prima (derivata discreta) → shape (N, 1, n_timepoints-1)
    # Corrisponde al nome 'input_diff' definito nell'export ONNX di Hydra
    # np.diff(X, axis=-1) calcola X[t+1] - X[t] per ogni t
    X_diff  = np.diff(X_cal, axis=-1).astype(np.float32)  # (N,999)
    X_diff  = X_diff[:, np.newaxis, :]                    # (N,1,999)
    X_diff  = X_diff[:, np.newaxis, :]                    # (N,1,1,999)
    
    log.info("")
    log.info("  Shape input (segnale):    %s", X_input.shape)
    log.info("  Shape input_diff (diff):  %s", X_diff.shape)

    # ── 5. Salvataggio NPZ per ST Edge AI ─────────────────────────────────────
    # Le chiavi DEVONO corrispondere esattamente ai nomi dell'ONNX:
    #   input_names=["input", "input_diff"] in export_model.py
    path_npz = os.path.join(args.out, "calibration_data_hydra.npz")
    np.savez(
        path_npz,
        input=X_input,       # chiave 'input'      → segnale grezzo
        input_diff=X_diff,   # chiave 'input_diff' → differenza prima
    )
    size_mb = os.path.getsize(path_npz) / 1024 / 1024
    log.info("")
    log.info("  Salvato: %s  (%.2f MB)", path_npz, size_mb)
    log.info("  Chiave 'input':      shape %s  dtype %s",
             X_input.shape, X_input.dtype)
    log.info("  Chiave 'input_diff': shape %s  dtype %s",
             X_diff.shape, X_diff.dtype)

    # ── 6. Salva label per verifica (non serve a ST Edge AI) ──────────────────
    path_labels = os.path.join(args.out, "calibration_labels.npy")
    np.save(path_labels, y_cal)
    log.info("  Salvato: %s  (solo per verifica)", path_labels)

    # ── 7. Report testuale ────────────────────────────────────────────────────
    report_path = os.path.join(args.out, "calibration_hydra_info.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("DATASET DI CALIBRAZIONE — Hydra ONNX — ST Edge AI\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Sorgente dataset:    {args.dataset}\n")
        f.write(f"n_timepoints:        {n_timepoints}\n")
        f.write(f"Campioni per classe: {args.n_per_class}\n")
        f.write(f"Totale campioni:     {len(idx_cal)}\n")
        f.write(f"  label=0 (no arco): {n0}\n")
        f.write(f"  label=1 (arco):    {n1}\n\n")
        f.write("File generati:\n")
        f.write(f"  calibration_data_hydra.npz\n")
        f.write(f"    chiave 'input':      shape={X_input.shape}  "
                f"dtype={X_input.dtype}  ← segnale grezzo\n")
        f.write(f"    chiave 'input_diff': shape={X_diff.shape}  "
                f"dtype={X_diff.dtype}  ← differenza prima\n\n")
        f.write("Statistiche X:\n")
        f.write(f"  min:  {float(X_cal.min()):.4f}\n")
        f.write(f"  max:  {float(X_cal.max()):.4f}\n")
        f.write(f"  mean: {float(X_cal.mean()):.4f}\n")
        f.write(f"  std:  {float(X_cal.std()):.4f}\n\n")
        f.write("IMPORTANTE — cosa quantizzare:\n")
        f.write("  ✅ hydra.onnx          → quantizza con ST Edge AI\n")
        f.write("  ❌ mrh_ridge.onnx      → NON quantizzare (usa .h in C)\n")
        f.write("  ❌ ridge.onnx          → NON quantizzare (usa .h in C)\n\n")
        f.write("Come usare in ST Edge AI:\n")
        f.write("  1. Importa hydra.onnx  (NON mrh_ridge.onnx)\n")
        f.write("  2. Seleziona quantizzazione INT8\n")
        f.write("  3. Carica calibration_data_hydra.npz\n")
        f.write("  4. Avvia la quantizzazione\n")

    log.info("  Salvato: %s", report_path)
    log.info("")
    log.info("=" * 60)
    log.info("RIEPILOGO — Come usare in ST Edge AI:")
    log.info("  ✅ Carica:   hydra.onnx")
    log.info("  ❌ NON usare: mrh_ridge.onnx  (va in errore opset)")
    log.info("  Dataset:    %s", path_npz)
    log.info("  Chiavi:     'input'  +  'input_diff'")
    log.info("=" * 60)


if __name__ == "__main__":
    main()