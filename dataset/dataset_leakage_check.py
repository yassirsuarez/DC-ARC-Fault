#!/usr/bin/env python3
"""
dataset_leakage_check.py
========================
Controllo completo data leakage indipendente dal modello.

USO:
python dataset_leakage_check.py train.npz test.npz \
    --meta-train arc_dataset_meta_train.csv \
    --meta-test arc_dataset_meta_test.csv
"""

import argparse
import logging
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import RidgeClassifier
from sklearn.utils import shuffle

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ============================================================
# LOAD
# ============================================================
def load_npz(path):
    d = np.load(path)
    return d["X"], d["y"]


def flatten(X):
    return X.reshape(X.shape[0], -1)


# ============================================================
# 1. CLASS DISTRIBUTION
# ============================================================
def class_distribution(y, name):
    log.info(f"\n=== Distribuzione classi ({name}) ===")
    u, c = np.unique(y, return_counts=True)

    for cls, n in zip(u, c):
        log.info(f"classe {cls}: {n} ({100*n/len(y):.2f}%)")


# ============================================================
# 2. EXACT DUPLICATES
# ============================================================
def exact_duplicates(X_train, X_test):
    log.info("\n=== Exact duplicates ===")

    tr = flatten(X_train)
    te = flatten(X_test)

    train_set = set(map(tuple, tr))
    overlap = sum(tuple(x) in train_set for x in te)

    log.info("duplicati train-test: %d", overlap)

    return overlap


# ============================================================
# 3. NEAR DUPLICATES
# ============================================================
def near_duplicates(X_train, X_test, threshold=1e-3, max_samples=200):
    log.info("\n=== Near duplicates ===")

    tr = flatten(X_train[:max_samples])
    te = flatten(X_test[:max_samples])

    D = euclidean_distances(te, tr)
    mins = D.min(axis=1)

    suspicious = np.sum(mins < threshold)

    log.info("min distance avg: %.8f", mins.mean())
    log.info("min distance min: %.8f", mins.min())
    log.info("near duplicates: %d", suspicious)

    return suspicious


# ============================================================
# 4. GROUP LEAKAGE
# ============================================================
def group_leakage(meta_train, meta_test):
    log.info("\n=== Group leakage ===")

    mt = pd.read_csv(meta_train)
    ms = pd.read_csv(meta_test)

    tr = set(
        mt["filename"].str.replace(
            "_Study00[12]_Raw Data.mat", "", regex=True
        )
    )

    te = set(
        ms["filename"].str.replace(
            "_Study00[12]_Raw Data.mat", "", regex=True
        )
    )

    overlap = tr & te

    log.info("file train: %d", len(tr))
    log.info("file test : %d", len(te))
    log.info("in comune : %d", len(overlap))

    if overlap:
        for x in sorted(overlap):
            log.warning("  %s", x)

    return len(overlap)


# ============================================================
# 5. SHUFFLE TEST
# ============================================================
def simple_features(X):
    if X.ndim == 3:
        X = X.mean(axis=1)

    mean = X.mean(axis=1, keepdims=True)
    std  = X.std(axis=1, keepdims=True)
    mx   = X.max(axis=1, keepdims=True)
    mn   = X.min(axis=1, keepdims=True)

    return np.concatenate([mean, std, mx, mn], axis=1)


def shuffle_test(X_train, y_train, X_test, y_test):
    log.info("\n=== Shuffle label test ===")

    Xtr = simple_features(X_train)
    Xte = simple_features(X_test)

    y_fake = shuffle(y_train, random_state=42)

    clf = RidgeClassifier()
    clf.fit(Xtr, y_fake)

    pred = clf.predict(Xte)

    acc = balanced_accuracy_score(y_test, pred)

    log.info("accuracy con label random: %.4f", acc)

    return acc


# ============================================================
# REPORT
# ============================================================
def final_report(exact_dup, near_dup, group_overlap, shuffle_acc):
    log.info("\n" + "="*50)
    log.info("FINAL REPORT")
    log.info("="*50)

    risk = 0

    if exact_dup > 0:
        risk += 2
        log.warning("Exact duplicates trovati")

    if near_dup > 0:
        risk += 2
        log.warning("Near duplicates trovati")

    if group_overlap > 0:
        risk += 3
        log.warning("Group leakage trovato")

    if shuffle_acc > 0.6:
        risk += 3
        log.warning("Shuffle accuracy troppo alta")

    if risk == 0:
        log.info("✅ DATASET CLEAN")
    elif risk <= 3:
        log.info("⚠ RISCHIO BASSO")
    elif risk <= 6:
        log.info("⚠ RISCHIO MEDIO")
    else:
        log.info("🚨 DATA LEAKAGE MOLTO PROBABILE")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("train")
    parser.add_argument("test")
    parser.add_argument("--meta-train", default=None)
    parser.add_argument("--meta-test", default=None)

    args = parser.parse_args()

    X_train, y_train = load_npz(args.train)
    X_test, y_test   = load_npz(args.test)

    log.info("Train: %s", X_train.shape)
    log.info("Test : %s", X_test.shape)

    class_distribution(y_train, "train")
    class_distribution(y_test, "test")

    exact_dup = exact_duplicates(X_train, X_test)
    near_dup  = near_duplicates(X_train, X_test)

    group_overlap = 0
    if args.meta_train and args.meta_test:
        group_overlap = group_leakage(
            args.meta_train,
            args.meta_test
        )

    shuffle_acc = shuffle_test(
        X_train, y_train,
        X_test, y_test
    )

    final_report(
        exact_dup,
        near_dup,
        group_overlap,
        shuffle_acc
    )


if __name__ == "__main__":
    main()