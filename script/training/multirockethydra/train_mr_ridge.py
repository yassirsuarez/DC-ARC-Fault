#!/usr/bin/env python3
"""
MultiRocket + Ridge (STM32-ready)
==================================

Pipeline:
    X → MultiRocket → features → RidgeClassifier → export C

Output:
    ridge_model.pkl
    scaler.json
"""

import argparse
import json
import os
import time
import pickle
import numpy as np

from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from aeon.transformations.collection.convolution_based import MultiRocket

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# -------------------------------------------------------------
# LOAD DATASET
# -------------------------------------------------------------
def load_dataset(path, downsample=4):
    data = np.load(path)
    X = data["X"]
    y = data["y"]

    if X.ndim == 2:
        X = X[:, np.newaxis, :]

    if downsample > 1:
        X = X[:, :, ::downsample]

    return X.astype(np.float32), y.astype(np.int64)


# -------------------------------------------------------------
# FEATURE EXTRACTION
# -------------------------------------------------------------
def extract_features(X_train, X_test):
    print("\n[MultiRocket] extracting features...")

    tr = MultiRocket(
        random_state=RANDOM_STATE,
        n_jobs=1
    )

    t0 = time.time()
    F_train = tr.fit_transform(X_train)
    F_test  = tr.transform(X_test)

    print(f"Done in {time.time() - t0:.2f}s")
    print("Train shape:", F_train.shape)
    print("Test shape :", F_test.shape)

    return F_train, F_test, tr


# -------------------------------------------------------------
# SCALER SIMPLE (STM32 FRIENDLY)
# -------------------------------------------------------------
class SimpleScaler:
    def fit(self, X):
        self.mean = X.mean(axis=0).astype(np.float32)
        self.std  = X.std(axis=0).astype(np.float32)
        self.std[self.std < 1e-8] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean) / self.std


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("train")
    parser.add_argument("test")
    parser.add_argument("--out", default="./results_ridge")
    parser.add_argument("--downsample", type=int, default=4)

    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ---------------------------------------------------------
    print("\nLOAD DATA")
    print("=" * 50)

    X_train, y_train = load_dataset(args.train, args.downsample)
    X_test,  y_test  = load_dataset(args.test, args.downsample)

    print("Train:", X_train.shape)
    print("Test :", X_test.shape)

    # ---------------------------------------------------------
    print("\nFEATURE EXTRACTION")
    print("=" * 50)

    F_train, F_test, transformer = extract_features(X_train, X_test)

    # ---------------------------------------------------------
    print("\nSCALING")
    print("=" * 50)

    scaler = SimpleScaler()
    F_train = scaler.fit(F_train).transform(F_train)
    F_test  = scaler.transform(F_test)

    print("Feature range:", F_train.min(), "→", F_train.max())

    # ---------------------------------------------------------
    print("\nTRAIN RIDGE")
    print("=" * 50)

    model = RidgeClassifier(
        alpha=1.0,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    t0 = time.time()
    model.fit(F_train, y_train)
    print("Training time:", round(time.time() - t0, 2), "s")

    # ---------------------------------------------------------
    print("\nEVALUATION")
    print("=" * 50)

    y_pred = model.predict(F_test)

    # decision function → probabilità
    scores = model.decision_function(F_test)
    probs = 1 / (1 + np.exp(-scores))

    acc = accuracy_score(y_test, y_pred)
    ba  = balanced_accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, probs)

    print(classification_report(y_test, y_pred, digits=4))
    print("Accuracy:", acc)
    print("Balanced Accuracy:", ba)
    print("F1:", f1)
    print("AUC:", auc)
    print(confusion_matrix(y_test, y_pred))

    # ---------------------------------------------------------
    print("\nEXPORT")
    print("=" * 50)

    bundle = {
        "model": model,
        "transformer": transformer,
        "scaler_mean": scaler.mean,
        "scaler_std": scaler.std,
        "metrics": {
            "accuracy": float(acc),
            "balanced_accuracy": float(ba),
            "f1": float(f1),
            "auc": float(auc),
        }
    }

    out_pkl = os.path.join(args.out, "ridge_model.pkl")

    with open(out_pkl, "wb") as f:
        pickle.dump(bundle, f)

    config = {
        "model": "MultiRocket + Ridge (STM32-ready)",
        "n_features": F_train.shape[1],
        "downsample": args.downsample,
        "metrics": bundle["metrics"]
    }

    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ---------------------------------------------------------
    print("\nSTM32 INFO")
    print("=" * 50)

    n_features = F_train.shape[1]
    print("Features:", n_features)
    print("Flash Ridge ~", n_features * 4 / 1024, "KB")


if __name__ == "__main__":
    main()