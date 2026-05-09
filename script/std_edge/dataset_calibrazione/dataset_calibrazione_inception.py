#!/usr/bin/env python3
"""
generate_calibration_dataset.py
================================
Genera il dataset di calibrazione per la quantizzazione INT8 del modello
InceptionTime (o MultiRocketHydra) tramite ST Edge AI / X-CUBE-AI.

ST Edge AI richiede un dataset di calibrazione in formato float32 per
determinare i range di attivazione di ogni layer prima di quantizzare
i pesi da float32 a int8.

Il dataset di calibrazione:
  - NON viene usato per il training
  - NON viene usato per la valutazione
  - Serve SOLO a ST Edge AI per stimare i range dei valori intermedi
  - Deve essere rappresentativo della distribuzione reale dei dati

Formati di output:
  - calibration_data.npz      shape (N, 1, 1000)  float32  ← ST Edge AI (chiave 'input')
  - calibration_data.npy      shape (N, 1, 1000)  float32  ← alternativo
  - calibration_data_flat.npy shape (N, 1000)     float32  ← alternativo flat
  - calibration_info.txt      riepilogo del dataset

Uso:
    python generate_calibration_dataset.py <arc_dataset_train.npz>
    python generate_calibration_dataset.py <arc_dataset_train.npz> --n-per-class 300
    python generate_calibration_dataset.py <arc_dataset_train.npz> --out <cartella>

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
N_PER_CLASS  = 100   # campioni per classe (200 arco + 200 no-arco = 400 totali)
                     # ST Edge AI suggerisce almeno 100-200 campioni totali
RAND_STATE   = 42


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Genera il dataset di calibrazione per la quantizzazione INT8\n"
            "del modello ONNX tramite ST Edge AI / X-CUBE-AI."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        help="Percorso al file arc_dataset_train.npz",
    )
    parser.add_argument(
        "--out", "-o",
        default="./calibration_inceptiontime",
        help="Cartella di output (default: ./calibration_inceptiontime)",
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

    # ── carica dataset ────────────────────────────────────────────────────────
    log.info("Caricamento dataset: %s", args.dataset)
    data = np.load(args.dataset)
    X    = data["X"]   # shape (N, 1000)  float32
    y    = data["y"]   # shape (N,)       int32

    log.info("  X shape:    %s  dtype=%s", X.shape, X.dtype)
    log.info("  y shape:    %s  dtype=%s", y.shape, y.dtype)
    log.info("  label=0:    %d  (no arco)", int((y == 0).sum()))
    log.info("  label=1:    %d  (arco)",    int((y == 1).sum()))

    # ── campionamento bilanciato ──────────────────────────────────────────────
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

    idx_cal = np.concatenate([idx0_cal, idx1_cal])
    rng.shuffle(idx_cal)

    X_cal = X[idx_cal].astype(np.float32)   # shape (N_tot, 1000)
    y_cal = y[idx_cal]

    log.info("")
    log.info("  Campioni selezionati:")
    log.info("    label=0 (no arco): %d", n0)
    log.info("    label=1 (arco):    %d", n1)
    log.info("    Totale:            %d", len(idx_cal))

    # ── statistiche del dataset di calibrazione ───────────────────────────────
    log.info("")
    log.info("  Statistiche X_cal:")
    log.info("    min:    %.4f", float(X_cal.min()))
    log.info("    max:    %.4f", float(X_cal.max()))
    log.info("    mean:   %.4f", float(X_cal.mean()))
    log.info("    std:    %.4f", float(X_cal.std()))

    # shape 3D: (N, 1, 1000) — stesso formato del training InceptionTime
    X_cal_3d = X_cal[:, np.newaxis, :]
    X_cal_4d = X_cal_3d[:, np.newaxis, :]   # (N,1,1,1000)
    # ── formato NPZ per ST Edge AI ← usa questo ───────────────────────────────
    # ST Edge AI accetta .npz con chiave 'input' oppure 'X'
    # shape (N, 1, 1000) float32
    path_npz = os.path.join(args.out, "calibration_data.npz")
    np.savez(path_npz, input=X_cal_4d)

    size_mb_npz = os.path.getsize(path_npz) / 1024 / 1024
    log.info("")
    log.info("  Salvato (formato NPZ — usa questo in ST Edge AI):")
    log.info("    %s", path_npz)
    log.info("    chiavi: 'input', 'X'  shape: %s  dtype: %s  (%.2f MB)",
             X_cal_3d.shape, X_cal_3d.dtype, size_mb_npz)
    data = np.load("./calibration_inceptiontime/calibration_data.npz")
    print(data.files)

    # ── formato NPY (alternativo se ST Edge AI non accetta npz) ──────────────
    path_npy = os.path.join(args.out, "calibration_data.npy")
    np.save(path_npy, X_cal_3d)
    size_mb_npy = os.path.getsize(path_npy) / 1024 / 1024
    log.info("")
    log.info("  Salvato (formato NPY — alternativo):")
    log.info("    %s", path_npy)
    log.info("    shape: %s  dtype: %s  (%.2f MB)",
             X_cal_3d.shape, X_cal_3d.dtype, size_mb_npy)

    # ── formato flat (alcune versioni vecchie di ST Edge AI) ─────────────────
    path_flat = os.path.join(args.out, "calibration_data_flat.npy")
    np.save(path_flat, X_cal)
    log.info("")
    log.info("  Salvato (formato flat 2D — fallback):")
    log.info("    %s", path_flat)
    log.info("    shape: %s  dtype: %s", X_cal.shape, X_cal.dtype)

    # ── salva anche le label (utile per verifica) ─────────────────────────────
    path_labels = os.path.join(args.out, "calibration_labels.npy")
    np.save(path_labels, y_cal)
    log.info("")
    log.info("  Salvato (label — solo per verifica, non serve a ST Edge AI):")
    log.info("    %s", path_labels)

    # ── report testuale ───────────────────────────────────────────────────────
    report_path = os.path.join(args.out, "calibration_info.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("DATASET DI CALIBRAZIONE — ST Edge AI / X-CUBE-AI\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Sorgente dataset:    {args.dataset}\n")
        f.write(f"Campioni per classe: {args.n_per_class}\n")
        f.write(f"Totale campioni:     {len(idx_cal)}\n")
        f.write(f"  label=0 (no arco): {n0}\n")
        f.write(f"  label=1 (arco):    {n1}\n\n")
        f.write("File generati:\n")
        f.write(f"  calibration_data.npz      shape={X_cal_3d.shape}  "
                f"dtype={X_cal_3d.dtype}  ← USA QUESTO in ST Edge AI\n")
        f.write(f"    chiavi: 'input' (principale), 'X' (alternativa), 'y'\n")
        f.write(f"  calibration_data.npy      shape={X_cal_3d.shape}  "
                f"dtype={X_cal_3d.dtype}  ← alternativo se npz non accettato\n")
        f.write(f"  calibration_data_flat.npy shape={X_cal.shape}  "
                f"dtype={X_cal.dtype}  ← fallback 2D\n")
        f.write(f"  calibration_labels.npy    shape={y_cal.shape}  "
                f"← solo per verifica\n\n")
        f.write("Statistiche X:\n")
        f.write(f"  min:  {float(X_cal.min()):.4f}\n")
        f.write(f"  max:  {float(X_cal.max()):.4f}\n")
        f.write(f"  mean: {float(X_cal.mean()):.4f}\n")
        f.write(f"  std:  {float(X_cal.std()):.4f}\n\n")
        f.write("Note:\n")
        f.write("  - I dati sono in float32, normalizzati (I/I_nom)\n")
        f.write("  - ST Edge AI esegue la quantizzazione INT8 internamente\n")
        f.write("  - NON usare questo dataset per training o valutazione\n")
        f.write("  - Selezionati casualmente con seed=42 (riproducibile)\n\n")
        f.write("Come usare in ST Edge AI:\n")
        f.write("  1. Importa il modello ONNX (inceptiontime.onnx)\n")
        f.write("  2. Seleziona quantizzazione INT8\n")
        f.write("  3. Carica calibration_data.npz come dataset di calibrazione\n")
        f.write("     Se chiede la chiave: usa 'input' oppure 'X'\n")
        f.write("  4. Avvia la quantizzazione\n")

    log.info("")
    log.info("=" * 60)
    log.info("RIEPILOGO — Come usare in ST Edge AI:")
    log.info("  1. Importa inceptiontime.onnx")
    log.info("  2. Seleziona quantizzazione INT8")
    log.info("  3. Carica: %s", path_npz)
    log.info("     chiave da usare: 'input'  oppure  'X'")
    log.info("     Se non accettato, prova: %s", path_npy)
    log.info("  4. Avvia la quantizzazione")
    log.info("=" * 60)


if __name__ == "__main__":
    main()