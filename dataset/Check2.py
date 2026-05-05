#!/usr/bin/env python3
"""
Check.py
=======================
Verifica se il modello Hydra ha leakage o split corretto

esempio uso:
python Check.py C:\Users\Asus\Desktop\progetto_manutenzione\dataset\dataset_new\arc_dataset_test.npz 
"""

import numpy as np
import argparse
import logging

from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# FEATURE HYDRA (stessa del tuo modello)
# ─────────────────────────────────────────────
def extract_hydra_features(X):
    if X.ndim == 3:
        X = X.mean(axis=1)

    mean = X.mean(axis=1, keepdims=True)
    std  = X.std(axis=1, keepdims=True)
    maxv = X.max(axis=1, keepdims=True)
    minv = X.min(axis=1, keepdims=True)

    return np.concatenate([mean, std, maxv, minv], axis=1)


# ─────────────────────────────────────────────
def simple_model_eval(X_train, y_train, X_test, y_test):
    from sklearn.linear_model import RidgeClassifier

    clf = RidgeClassifier(alpha=1.0)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    return balanced_accuracy_score(y_test, pred)


# ─────────────────────────────────────────────
def detect_duplicates(X):
    """Controlla segnali identici o quasi identici"""
    log.info("\n[CHECK] Duplicati / segnali simili...")

    flat = X.reshape(X.shape[0], -1)

    unique = np.unique(flat, axis=0)

    ratio = len(unique) / len(flat)

    log.info("  campioni totali:   %d", len(flat))
    log.info("  campioni unici:    %d", len(unique))
    log.info("  ratio unicità:     %.4f", ratio)

    if ratio < 0.95:
        log.warning("  ⚠ POSSIBILE DUPLICAZIONE ALTA (leakage risk)")
    else:
        log.info("  ✔ nessun problema evidente")


# ─────────────────────────────────────────────
def repeated_split_test(X, y, groups=None):
    """Test robustezza su split multipli"""
    log.info("\n[CHECK] Robustezza su split multipli...")

    accs = []

    if groups is None:
        splitter = lambda: np.random.permutation(len(y))
        for i in range(5):
            idx = splitter()
            split = int(0.8 * len(y))

            tr, te = idx[:split], idx[split:]

            X_tr = extract_hydra_features(X[tr])
            X_te = extract_hydra_features(X[te])

            acc = simple_model_eval(X_tr, y[tr], X_te, y[te])
            accs.append(acc)

    else:
        gss = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

        for tr, te in gss.split(X, y, groups=groups):
            X_tr = extract_hydra_features(X[tr])
            X_te = extract_hydra_features(X[te])

            acc = simple_model_eval(X_tr, y[tr], X_te, y[te])
            accs.append(acc)

    log.info("  accuracy runs: %s", np.round(accs, 4))
    log.info("  mean accuracy: %.4f", np.mean(accs))
    log.info("  std accuracy:  %.4f", np.std(accs))

    if np.std(accs) > 0.02:
        log.warning("  ⚠ modello instabile (possibile leakage o overfit)")
    else:
        log.info("  ✔ modello stabile")


# ─────────────────────────────────────────────
def class_distribution(y):
    log.info("\n[CHECK] Distribuzione classi...")

    unique, counts = np.unique(y, return_counts=True)

    for u, c in zip(unique, counts):
        log.info("  classe %d: %d (%.2f%%)", u, c, 100*c/len(y))


# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()

    data = np.load(args.dataset)
    X = data["X"]
    y = data["y"]

    log.info("Dataset shape: %s", X.shape)

    # ── CHECK 1: distribuzione classi
    class_distribution(y)

    # ── CHECK 2: duplicati
    detect_duplicates(X)

    # ── CHECK 3: split robusto
    repeated_split_test(X, y)


if __name__ == "__main__":
    main()