#!/usr/bin/env python3
"""
build_dataset.py
================
Costruisce il dataset di addestramento per il classificatore real-time
di archi elettrici in impianti fotovoltaici DC.

Modalità: FINESTRA SCORREVOLE (Approccio B)
  Ogni file .mat genera multiple finestre temporali con passo STEP_S.
  Le finestre prima di t1 ricevono label=0 (pre-arco).
  Le finestre dopo t1 ricevono label=1 (arco in corso).
  Questo riproduce esattamente il comportamento del classificatore
  in produzione su STM32/ESP32, dove ogni STEP_S viene classificata
  una nuova finestra di WINDOW_S secondi.

Esempio con WINDOW_S=0.10s, STEP_S=0.05s su file da 4s con t1=0.34s:
  finestra  0ms– 100ms  → label=0  (pre-arco)
  finestra 50ms– 150ms  → label=0  (pre-arco)
  ...
  finestra 300ms– 400ms → label=0  (t1=340ms non ancora raggiunto)
  finestra 350ms– 450ms → label=1  (t1 cade dentro questa finestra)
  finestra 400ms– 500ms → label=1  (arco stabile)
  ...

Parametri principali (modificabili):
  WINDOW_S = 0.10   finestra di classificazione [s]
  STEP_S   = 0.05   passo tra finestre consecutive [s]
  MAX_WIN_PER_CLASS  max finestre per classe per file (evita squilibri)

Uso:
    python build_dataset.py <cartella_dataset> [--out <cartella>]
                            [--exclude-low-voltage]

Output:
    arc_dataset.npz          X (n_finestre × n_campioni), y (n_finestre,)
    arc_dataset_meta.csv     Metadati per finestra (file, t_start, label)
    arc_dataset_skipped.csv  File scartati con motivazione

Autori: progetto tesi magistrale — Manutenzione e Affidabilità
Dataset: https://ieee-dataport.org/open-access/photovoltaic-pv-dc-arc-library
"""

import argparse
import csv
import logging
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.io import loadmat

# ── configurazione logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETRI — modifica qui
# ══════════════════════════════════════════════════════════════════════════════

FS_HZ    = 10_000   # Frequenza di campionamento dopo sottocampionamento [Hz]

WINDOW_S = 0.10     # Durata finestra di classificazione [s]
                    # 0.10s = 1.000 campioni  ← real-time (prof)
                    # 0.50s = 5.000 campioni
                    # 1.00s = 10.000 campioni
                    # 2.00s = 20.000 campioni ← offline (run precedenti)

STEP_S   = 0.05     # Passo tra finestre consecutive [s]
                    # 0.05s = 50ms  ← latenza massima di rilevamento

# Max finestre label=0 e label=1 estratte per file
# Limita lo squilibrio e la ridondanza (molte finestre consecutive simili)
MAX_WIN_PER_CLASS = 10

# Parametri rilevamento t1 — calcolati sull'intera registrazione (non sulla finestra)
NOMINAL_S   = 0.20   # Durata segmento per stima corrente nominale [s]
CONFIRM_MS  = 300    # Durata minima sotto soglia per confermare t1 [ms]
THRESH_FRAC = 0.95   # Soglia = nominale × THRESH_FRAC
MIN_NOM_A   = 0.10   # Corrente nominale minima accettabile [A]

# Tensioni da escludere con --exclude-low-voltage
LOW_VOLTAGE_TAGS = ("_0050V_", "_0100V_", "_00.50A_", "_01.00A_")

# ══════════════════════════════════════════════════════════════════════════════
# Costanti derivate
# ══════════════════════════════════════════════════════════════════════════════

WIN_N      = int(WINDOW_S * FS_HZ)
STEP_N     = int(STEP_S   * FS_HZ)
NOMINAL_N  = int(NOMINAL_S * FS_HZ)
CONFIRM_N  = int(CONFIRM_MS / 1000 * FS_HZ)


# ══════════════════════════════════════════════════════════════════════════════
# Rilevamento t1 — calcolato sull'intera registrazione
# ══════════════════════════════════════════════════════════════════════════════

def find_t1(current: np.ndarray) -> tuple:
    """
    Rileva t1 (inizio arco) dall'intera forma d'onda di corrente.

    Stima il nominale sui primi NOMINAL_S secondi.
    t1 = primo campione in cui la corrente rimane sotto
    THRESH_FRAC × nominale per almeno CONFIRM_MS ms consecutivi.

    Returns:
        t1_idx (int | None): indice campione di t1, None se non trovato.
        nominal (float): corrente nominale stimata [A].
    """
    nominal = float(np.median(current[:NOMINAL_N]))
    if nominal < MIN_NOM_A:
        return None, nominal
    threshold = nominal * THRESH_FRAC
    below     = current < threshold
    for i in range(len(below) - CONFIRM_N):
        if np.all(below[i:i + CONFIRM_N]):
            return i, nominal
    return None, nominal


# ══════════════════════════════════════════════════════════════════════════════
# Estrazione finestre scorrevoli da un file
# ══════════════════════════════════════════════════════════════════════════════

def extract_windows(path: str) -> tuple:
    """
    Carica un file .mat ed estrae le finestre scorrevoli con label.

    Per ogni finestra [t_start, t_start + WINDOW_S]:
      - label = 0 se t_start + WINDOW_S <= t1  (finestra interamente pre-arco)
      - label = 1 se t_start >= t1             (finestra interamente in-arco)
      - Le finestre che attraversano t1 vengono scartate (transitorio)

    Applica MAX_WIN_PER_CLASS per bilanciare le classi per file.

    Returns:
        windows (list of np.ndarray): finestre normalizzate float32
        labels  (list of int):        label per ogni finestra
        meta    (list of dict):       metadati per ogni finestra
        skip_info (dict | None):      info se il file è stato scartato
    """
    fname = os.path.basename(path)

    # Carica file
    try:
        mat = loadmat(path, squeeze_me=False)
    except Exception as exc:
        return [], [], [], {"filename": fname, "skip_reason": f"errore lettura: {exc}"}

    if "CurrentData" not in mat:
        return [], [], [], {"filename": fname, "skip_reason": "CurrentData assente"}

    current = mat["CurrentData"].flatten()

    if len(current) < WIN_N + STEP_N:
        return [], [], [], {
            "filename":    fname,
            "skip_reason": f"segnale troppo corto ({len(current)} campioni)",
        }

    # Stima t1 e nominale sull'intera registrazione
    t1_idx, nominal = find_t1(current)

    if nominal < MIN_NOM_A:
        return [], [], [], {
            "filename":    fname,
            "skip_reason": f"nominale troppo basso ({nominal:.4f} A)",
        }

    # Normalizzazione intera registrazione
    current_norm = (current / nominal).astype(np.float32)
    n_total      = len(current_norm)

    # Generazione finestre
    wins_0, wins_1 = [], []   # finestre per classe
    meta_0, meta_1 = [], []

    start = 0
    while start + WIN_N <= n_total:
        end = start + WIN_N

        if t1_idx is None:
            # File senza arco → tutte le finestre sono label=0
            label = 0
        elif end <= t1_idx:
            # Finestra interamente pre-arco
            label = 0
        elif start >= t1_idx:
            # Finestra interamente in-arco
            label = 1
        else:
            # Finestra attraversa t1 → scarta (transitorio ambiguo)
            start += STEP_N
            continue

        win  = current_norm[start:end].copy()
        info = {
            "filename": fname,
            "t_start_s": round(start / FS_HZ, 4),
            "t_end_s":   round(end   / FS_HZ, 4),
            "t1_s":      round(t1_idx / FS_HZ, 4) if t1_idx else None,
            "nominal_A": round(nominal, 4),
            "label":     label,
        }

        if label == 0:
            wins_0.append(win);  meta_0.append(info)
        else:
            wins_1.append(win);  meta_1.append(info)

        start += STEP_N

    # Limita il numero di finestre per classe (evita ridondanza)
    rng = np.random.default_rng(42)

    if len(wins_0) > MAX_WIN_PER_CLASS:
        idx = rng.choice(len(wins_0), MAX_WIN_PER_CLASS, replace=False)
        wins_0 = [wins_0[i] for i in sorted(idx)]
        meta_0 = [meta_0[i] for i in sorted(idx)]

    if len(wins_1) > MAX_WIN_PER_CLASS:
        idx = rng.choice(len(wins_1), MAX_WIN_PER_CLASS, replace=False)
        wins_1 = [wins_1[i] for i in sorted(idx)]
        meta_1 = [meta_1[i] for i in sorted(idx)]

    windows = wins_0 + wins_1
    labels  = [0] * len(wins_0) + [1] * len(wins_1)
    meta    = meta_0 + meta_1

    return windows, labels, meta, None


# ══════════════════════════════════════════════════════════════════════════════
# Scansione cartella
# ══════════════════════════════════════════════════════════════════════════════

def scan_folder(root: str, exclude_low_voltage: bool = False) -> list:
    """Scansiona ricorsivamente la cartella e restituisce i file .mat da usare."""
    mat_files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in sorted(filenames):
            if not fn.lower().endswith(".mat"):
                continue
            if any(kw in fn.lower() for kw in ("recipe", "gap", "settings", "readme")):
                continue
            if exclude_low_voltage and any(tag in fn for tag in LOW_VOLTAGE_TAGS):
                continue
            mat_files.append(os.path.join(dirpath, fn))
    return sorted(mat_files)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Costruisce il dataset con finestra scorrevole per il classificatore\n"
            "real-time di archi elettrici in impianti fotovoltaici DC.\n\n"
            f"Finestra: {WINDOW_S*1000:.0f} ms  "
            f"Passo: {STEP_S*1000:.0f} ms  "
            f"Campioni per finestra: {WIN_N}"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dataset", help="Cartella radice del dataset estratto")
    parser.add_argument("--out", "-o", default="./training_data",
                        help="Cartella di output (default: ./training_data)")
    parser.add_argument(
        "--exclude-low-voltage", action="store_true",
        help="Escludi file a 50V e 100V (archi non sostenuti, labelling inaffidabile)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.dataset):
        log.error("Cartella non trovata: %s", args.dataset)
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)

    log.info("=" * 60)
    log.info("PARAMETRI FINESTRA SCORREVOLE")
    log.info("=" * 60)
    log.info("  Finestra:            %.0f ms  (%d campioni @ %d Hz)",
             WINDOW_S * 1000, WIN_N, FS_HZ)
    log.info("  Passo:               %.0f ms  (%d campioni)",
             STEP_S   * 1000, STEP_N)
    log.info("  Max finestre/classe: %d per file", MAX_WIN_PER_CLASS)
    log.info("  Normalizzazione:     I(t) / I_nominale")
    log.info("  Escludi bassa tens.: %s", args.exclude_low_voltage)

    files = scan_folder(args.dataset, args.exclude_low_voltage)
    if not files:
        log.error("Nessun file .mat trovato in: %s", args.dataset)
        sys.exit(1)
    log.info("  File .mat trovati:   %d", len(files))

    # Estrazione finestre
    X_all, y_all, meta_all, skipped_all = [], [], [], []
    n_no_arc_files = 0

    for i, path in enumerate(files, 1):
        print(f"\r  [{i:5d}/{len(files)}] {os.path.basename(path)[:55]}",
              end="", flush=True)
        windows, labels, meta, skip_info = extract_windows(path)

        if skip_info is not None:
            skipped_all.append(skip_info)
            continue

        if not windows:
            n_no_arc_files += 1
            continue

        X_all.extend(windows)
        y_all.extend(labels)
        meta_all.extend(meta)
    print()

    if not X_all:
        log.error("Nessuna finestra estratta. Controlla WINDOW_S e STEP_S.")
        sys.exit(1)

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.int32)

    # Salvataggio
    npz_path  = os.path.join(args.out, "arc_dataset_new.npz")
    meta_path = os.path.join(args.out, "arc_dataset_meta_new.csv")
    skip_path = os.path.join(args.out, "arc_dataset_skipped_new.csv")

    np.savez_compressed(npz_path, X=X, y=y)

    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["filename", "t_start_s", "t_end_s", "t1_s",
                      "nominal_A", "label"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(meta_all)

    with open(skip_path, "w", newline="", encoding="utf-8") as f:
        w2 = csv.DictWriter(f, fieldnames=["filename", "skip_reason"])
        w2.writeheader()
        w2.writerows(skipped_all)

    # Summary
    n1    = int((y == 1).sum())
    n0    = int((y == 0).sum())
    ratio = n1 / max(n0, 1)

    log.info("")
    log.info("=" * 60)
    log.info("SUMMARY DATASET")
    log.info("=" * 60)
    log.info("  File processati:       %d", len(files) - len(skipped_all))
    log.info("  File scartati:         %d", len(skipped_all))
    log.info("  Finestre totali:       %d", len(y))
    log.info("  label=0 (pre-arco):    %d  (%.1f%%)", n0, 100*n0/len(y))
    log.info("  label=1 (in-arco):     %d  (%.1f%%)", n1, 100*n1/len(y))
    log.info("  Ratio 1:0:             %.2f:1", ratio)
    log.info("  Campioni per finestra: %d  (%.0f ms @ %d Hz)",
             WIN_N, WINDOW_S * 1000, FS_HZ)
    log.info("  Dimensione file:       %.1f MB",
             os.path.getsize(npz_path) / 1024 / 1024)

    if abs(ratio - 1.0) < 0.3:
        log.info("  Bilanciamento: OTTIMO (%.2f:1)", ratio)
    elif ratio < 3:
        log.info("  Bilanciamento: buono (%.2f:1)", ratio)
    elif ratio < 7:
        log.warning("  Bilanciamento: moderato (%.2f:1) — "
                    "train_classifier userà undersampling", ratio)
    else:
        log.warning("  Bilanciamento: squilibrato (%.2f:1) — "
                    "train_classifier userà undersampling", ratio)

    log.info("")
    log.info("Output salvato in: %s", args.out)
    log.info("  %s", npz_path)
    log.info("  %s", meta_path)
    log.info("  %s", skip_path)


if __name__ == "__main__":
    main()