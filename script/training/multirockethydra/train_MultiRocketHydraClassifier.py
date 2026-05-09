#!/usr/bin/env python3
"""
MultiRocketHydra originale (aeon) per rilevamento archi elettrici DC PV
========================================================================

Pipeline:
    X -> MultiRocketHydraTransformer -> RidgeClassifierCV

REQUISITI:
    pip install aeon scikit-learn numpy joblib

USO:
    python train_original_mrh.py train.npz test.npz
"""
#!/usr/bin/env python3

import argparse
import json
import os
import time
import pickle
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from aeon.classification.convolution_based import MultiRocketHydraClassifier

RANDOM_STATE = 42


# -------------------------------------------------------------
# LOAD + DOWNSAMPLING (CRUCIALE)
# -------------------------------------------------------------
def load_dataset(path, downsample_factor=4):

    data = np.load(path)
    X = data["X"]
    y = data["y"]

    if X.ndim == 2:
        X = X[:, np.newaxis, :]

    # 🔥 DOWNSAMPLING per evitare OOM Hydra
    # da (1000) → (250) se factor=4
    X = X[:, :, ::downsample_factor]

    return X.astype(np.float32), y.astype(np.int64)


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("train")
    parser.add_argument("test")
    parser.add_argument("--out", default="./results_mrh")

    # 🔥 PARAMETRI SICURI (IMPORTANTISSIMO)
    parser.add_argument("--n-kernels", type=int, default=8)
    parser.add_argument("--n-groups", type=int, default=4)

    parser.add_argument("--downsample", type=int, default=4)

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ---------------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------------
    print("=" * 60)
    print("LOAD DATASET")
    print("=" * 60)

    X_train, y_train = load_dataset(args.train, args.downsample)
    X_test, y_test = load_dataset(args.test, args.downsample)

    print("Train:", X_train.shape, y_train.shape)
    print("Test :", X_test.shape, y_test.shape)

    # ---------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------
    print()
    print("=" * 60)
    print("MULTIROCKETHYDRA (SAFE MODE)")
    print("=" * 60)

    clf = MultiRocketHydraClassifier(
        n_kernels=args.n_kernels,
        n_groups=args.n_groups,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=1,   # 🔥 IMPORTANTE: -1 = RAM explosion in Hydra
    )

    t0 = time.time()

    clf.fit(X_train, y_train)

    print("Training time:", round(time.time() - t0, 2), "s")

    # ---------------------------------------------------------
    # PREDICT
    # ---------------------------------------------------------
    print()
    print("=" * 60)
    print("PREDICTION")
    print("=" * 60)

    y_pred = clf.predict(X_test)

    try:
        decision = clf.decision_function(X_test)
        y_score = 1 / (1 + np.exp(-decision))
    except:
        y_score = y_pred

    # ---------------------------------------------------------
    # METRICS
    # ---------------------------------------------------------
    print()
    print("=" * 60)
    print("METRICS")
    print("=" * 60)

    acc = accuracy_score(y_test, y_pred)
    ba = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_score)

    print(classification_report(
        y_test,
        y_pred,
        target_names=["No Arc", "Arc"],
        digits=4
    ))

    print("Accuracy          :", round(acc, 4))
    print("Balanced Accuracy :", round(ba, 4))
    print("F1 Score          :", round(f1, 4))
    print("ROC AUC           :", round(auc, 4))
    print(confusion_matrix(y_test, y_pred))

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------
    bundle = {
        "model": clf,
        "metrics": {
            "accuracy": float(acc),
            "balanced_accuracy": float(ba),
            "f1": float(f1),
            "roc_auc": float(auc),
        }
    }

    out_path = os.path.join(args.out, "mrh_model.pkl")

    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)

    config = {
        "model": "MultiRocketHydra SAFE",
        "n_kernels": args.n_kernels,
        "n_groups": args.n_groups,
        "downsample": args.downsample,
        "metrics": bundle["metrics"]
    }

    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("Saved:", out_path)


if __name__ == "__main__":
    main()