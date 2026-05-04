#!/usr/bin/env python3
"""
train_hydra_real_fixed.py
=========================
Hydra + Ridge CORRETTO per export STM32:

✔ salva modello come dict (NO classi custom → no pickle errors)
✔ salva Hydra separata per ONNX
✔ salva Ridge per C header
"""

import argparse
import os
import numpy as np
import logging
import pickle

import torch
import torch.nn as nn

from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score, f1_score

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# HYDRA FEATURE EXTRACTOR
# ─────────────────────────────────────────────
class HydraFeatureExtractor(nn.Module):
    def __init__(self, n_kernels=32, kernel_sizes=[3, 5, 9], dilations=[1, 2, 4]):
        super().__init__()

        self.convs = nn.ModuleList()

        for k in kernel_sizes:
            for d in dilations:
                self.convs.append(
                    nn.Conv1d(
                        in_channels=1,
                        out_channels=n_kernels,
                        kernel_size=k,
                        dilation=d,
                        padding=(k // 2) * d
                    )
                )

    def forward(self, x):
        x = x.unsqueeze(1)

        feats = []

        for conv in self.convs:
            y = torch.relu(conv(x))

            feats.append(torch.max(y, dim=-1).values)
            feats.append(torch.mean(y, dim=-1))

        return torch.cat(feats, dim=1)


# ─────────────────────────────────────────────
def extract_features(model, X, device):
    X = torch.tensor(X, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        feats = model(X).cpu().numpy()

    return feats


# ─────────────────────────────────────────────
def undersample(X, y, max_per_class=5000):
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    n = min(len(idx0), len(idx1), max_per_class)

    rng = np.random.default_rng(42)
    idx0 = rng.choice(idx0, n, replace=False)
    idx1 = rng.choice(idx1, n, replace=False)

    idx = np.concatenate([idx0, idx1])
    rng.shuffle(idx)

    return X[idx], y[idx]


# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--out", default="results_hydra_real")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ── LOAD DATASET ──
    data = np.load(args.dataset)
    X = data["X"]
    y = data["y"]

    log.info("Dataset: %s", X.shape)

    # ── SPLIT 80/20 ──
    n = len(y)
    idx = np.random.permutation(n)
    split = int(n * 0.8)

    tr_idx, te_idx = idx[:split], idx[split:]

    X_train, y_train = X[tr_idx], y[tr_idx]
    X_test, y_test = X[te_idx], y[te_idx]

    # ── BALANCE ──
    X_train, y_train = undersample(X_train, y_train)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── HYDRA ──
    log.info("Building Hydra feature extractor...")
    hydra = HydraFeatureExtractor().to(device)

    # ── FEATURES ──
    log.info("Extracting TRAIN features...")
    X_train_feat = extract_features(hydra, X_train, device)

    log.info("Extracting TEST features...")
    X_test_feat = extract_features(hydra, X_test, device)

    log.info("Feature shape: %s", X_train_feat.shape)

    # ── TRAIN RIDGE ──
    log.info("Training Ridge classifier...")
    clf = RidgeClassifier(alpha=1.0)
    clf.fit(X_train_feat, y_train)

    # ── EVAL ──
    y_pred = clf.predict(X_test_feat)

    ba = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    log.info("Balanced Accuracy: %.4f", ba)
    log.info("F1: %.4f", f1)

    log.info("\nReport:")
    log.info(classification_report(y_test, y_pred))

    # ─────────────────────────────────────────────
    # SAVE (FIX DEFINITIVO)
    # ─────────────────────────────────────────────

    # ✔ SALVATAGGIO SICURO (NO classi custom → NO pickle error)
    model_bundle = {
        "hydra_state_dict": hydra.state_dict(),
        "ridge": clf
    }

    full_path = os.path.join(args.out, "hydra_bundle.pkl")
    with open(full_path, "wb") as f:
        pickle.dump(model_bundle, f)

    # ✔ Hydra separata per ONNX export
    torch.save(hydra.state_dict(),
               os.path.join(args.out, "hydra_extractor.pt"))

    # ✔ Ridge per C header (opzionale export diretto)
    np.save(os.path.join(args.out, "ridge_coef.npy"), clf.coef_)
    np.save(os.path.join(args.out, "ridge_intercept.npy"), clf.intercept_)

    log.info("\nSaved:")
    log.info("  ✔ hydra_bundle.pkl (dict safe)")
    log.info("  ✔ hydra_extractor.pt")
    log.info("  ✔ ridge_coef.npy")
    log.info("  ✔ ridge_intercept.npy")


if __name__ == "__main__":
    main()