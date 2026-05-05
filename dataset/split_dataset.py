#!/usr/bin/env python3
"""
split_dataset.py
================
Divide fisicamente arc_dataset_new.npz in due file separati:
  - arc_dataset_train.npz
  - arc_dataset_test.npz

Lo split avviene per file sorgente (GroupShuffleSplit) — tutte le finestre
dello stesso file .mat finiscono nello stesso split, eliminando il data leakage.

Uso:
    python split_dataset.py arc_dataset_new.npz arc_dataset_meta_new.csv
    python split_dataset.py arc_dataset_new.npz arc_dataset_meta_new.csv --test-size 0.20
    python split_dataset.py arc_dataset_new.npz arc_dataset_meta_new.csv --out ./split

Output:
    arc_dataset_train.npz   X_train, y_train
    arc_dataset_test.npz    X_test,  y_test
    split_report.txt        riepilogo dello split
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

RAND_STATE = 42

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="Percorso a arc_dataset_new.npz")
parser.add_argument("meta",    help="Percorso a arc_dataset_meta_new.csv")
parser.add_argument("--test-size", type=float, default=0.20,
                    help="Frazione test set (default: 0.20)")
parser.add_argument("--out", default=None,
                    help="Cartella output (default: stessa del dataset)")
parser.add_argument("--seed", type=int, default=RAND_STATE)
args = parser.parse_args()

# ── Validazione input ─────────────────────────────────────────────────────────
for path in [args.dataset, args.meta]:
    if not os.path.isfile(path):
        print(f"ERRORE: file non trovato: {path}")
        sys.exit(1)

out_dir = args.out or os.path.dirname(os.path.abspath(args.dataset))
os.makedirs(out_dir, exist_ok=True)

# ── Carico dataset ────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  split_dataset.py — Train / Test Split per File Sorgente")
print(f"{'='*60}\n")

print(f"[1/4] Carico dataset: {args.dataset}")
data = np.load(args.dataset)
X = data["X"]
y = data["y"]
print(f"      X shape: {X.shape}  |  y shape: {y.shape}")
print(f"      label=0: {(y==0).sum()}  |  label=1: {(y==1).sum()}")

print(f"\n[2/4] Carico metadati: {args.meta}")
meta = pd.read_csv(args.meta)
print(f"      Righe metadati: {len(meta)}")

if len(meta) != len(y):
    print(f"ERRORE: metadati ({len(meta)}) e dataset ({len(y)}) hanno dimensioni diverse.")
    sys.exit(1)

# ── Costruisco i gruppi per file sorgente ─────────────────────────────────────
def exp_key(fn):
    """Riduce il nome file alla chiave dell'esperimento (senza Study001/002)."""
    s = fn.replace("_Raw Data.mat", "").replace(" Data.mat", "")
    idx = s.lower().rfind("_study")
    return s[:idx] if idx > 0 else s

meta["exp_key"] = meta["filename"].apply(exp_key)
unique_keys = {k: i for i, k in enumerate(meta["exp_key"].unique())}
groups = meta["exp_key"].map(unique_keys).values
n_groups = len(unique_keys)

print(f"      File sorgente unici: {n_groups}")
print(f"      Finestre per file (media): {len(y)/n_groups:.1f}")

# ── Split ─────────────────────────────────────────────────────────────────────
print(f"\n[3/4] GroupShuffleSplit  (test={args.test_size*100:.0f}%  seed={args.seed})")

gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, y_train = X[train_idx], y[train_idx]
X_test,  y_test  = X[test_idx],  y[test_idx]

# Verifica leakage
groups_train = set(groups[train_idx])
groups_test  = set(groups[test_idx])
overlap = groups_train & groups_test

if overlap:
    print(f"ERRORE DATA LEAKAGE: {len(overlap)} file presenti in entrambi gli split!")
    sys.exit(1)

print(f"  ✓ Nessun file sorgente condiviso (leakage = 0)")
print(f"\n  Train: {len(y_train):6d} finestre  da {len(groups_train):4d} file "
      f"| label=0: {(y_train==0).sum():5d}  label=1: {(y_train==1).sum():5d}")
print(f"  Test:  {len(y_test):6d} finestre  da {len(groups_test):4d} file "
      f"| label=0: {(y_test==0).sum():5d}  label=1: {(y_test==1).sum():5d}")

# Verifica che le proporzioni siano ragionevoli
actual_test_frac = len(y_test) / len(y)
print(f"\n  Frazione test effettiva: {actual_test_frac*100:.1f}%  "
      f"(richiesta: {args.test_size*100:.0f}%)")

# ── Salvo i file ──────────────────────────────────────────────────────────────
print(f"\n[4/4] Salvo file in: {out_dir}")

train_path = os.path.join(out_dir, "arc_dataset_train.npz")
test_path  = os.path.join(out_dir, "arc_dataset_test.npz")

np.savez_compressed(train_path, X=X_train, y=y_train)
np.savez_compressed(test_path,  X=X_test,  y=y_test)

size_train = os.path.getsize(train_path) / 1024 / 1024
size_test  = os.path.getsize(test_path)  / 1024 / 1024
print(f"  arc_dataset_train.npz  →  {size_train:.1f} MB")
print(f"  arc_dataset_test.npz   →  {size_test:.1f} MB")

# Salvo anche i metadati separati (utili per analisi successive)
meta_train = meta.iloc[train_idx].reset_index(drop=True)
meta_test  = meta.iloc[test_idx].reset_index(drop=True)
meta_train_path = os.path.join(out_dir, "arc_dataset_meta_train.csv")
meta_test_path  = os.path.join(out_dir, "arc_dataset_meta_test.csv")
meta_train.to_csv(meta_train_path, index=False)
meta_test.to_csv(meta_test_path,   index=False)
print(f"  arc_dataset_meta_train.csv")
print(f"  arc_dataset_meta_test.csv")

# ── Report testuale ───────────────────────────────────────────────────────────
report_path = os.path.join(out_dir, "split_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("Dataset Split Report\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Dataset originale : {args.dataset}\n")
    f.write(f"Metadati          : {args.meta}\n")
    f.write(f"Test size         : {args.test_size*100:.0f}%\n")
    f.write(f"Seed              : {args.seed}\n")
    f.write(f"Data leakage      : 0 (GroupShuffleSplit per file sorgente)\n\n")
    f.write(f"Totale finestre   : {len(y)}\n")
    f.write(f"Totale file       : {n_groups}\n\n")
    f.write(f"TRAIN\n")
    f.write(f"  Finestre  : {len(y_train)}\n")
    f.write(f"  File      : {len(groups_train)}\n")
    f.write(f"  label=0   : {(y_train==0).sum()}\n")
    f.write(f"  label=1   : {(y_train==1).sum()}\n\n")
    f.write(f"TEST\n")
    f.write(f"  Finestre  : {len(y_test)}\n")
    f.write(f"  File      : {len(groups_test)}\n")
    f.write(f"  label=0   : {(y_test==0).sum()}\n")
    f.write(f"  label=1   : {(y_test==1).sum()}\n\n")
    f.write("File sorgente nel test set:\n")
    test_files = meta.iloc[test_idx]["exp_key"].unique()
    for fn in sorted(test_files):
        f.write(f"  {fn}\n")
print(f"  split_report.txt")

print(f"\n{'='*60}")
print("  FATTO — usa questi file da ora in poi:")
print(f"  Training  →  arc_dataset_train.npz")
print(f"  Test      →  arc_dataset_test.npz  (mai usare per training)")
print(f"  Confronto float32/int8  →  arc_dataset_test.npz")
print(f"{'='*60}\n")
