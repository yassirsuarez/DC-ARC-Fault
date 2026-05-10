#!/usr/bin/env python3
"""
train_mcnn.py
=========================
Multi-scale CNN + Ridge per export STM32:

✔ dataset già separato (train/test esterni)
✔ dataset già bilanciato
✔ feature extractor CNN multi-scala
✔ Ridge classifier per semplicità e export
✔ salva modello per analisi/ONNX/edge pipeline

Esempio:
python train_mcnn.py --train train.npz --test test.npz --out results
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
# MULTI-SCALE CNN FEATURE EXTRACTOR (MCNN-like)
# ─────────────────────────────────────────────
class MultiScaleCNNFeatureExtractor(nn.Module):
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
        x = x.unsqueeze(1)  # (B, 1, T)

        feats = []

        for conv in self.convs:
            y = torch.relu(conv(x))

            # statistiche globali
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
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="train npz file")
    parser.add_argument("--test", required=True, help="test npz file")
    parser.add_argument("--out", default="results_mcnn")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ── LOAD DATASET ──
    train = np.load(args.train)
    test = np.load(args.test)

    X_train, y_train = train["X"], train["y"]
    X_test, y_test = test["X"], test["y"]

    log.info("Train: %s | Test: %s", X_train.shape, X_test.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── FEATURE EXTRACTOR ──
    log.info("Building Multi-scale CNN feature extractor...")
    feature_extractor = MultiScaleCNNFeatureExtractor().to(device)

    # ── FEATURE EXTRACTION ──
    log.info("Extracting TRAIN features...")
    X_train_feat = extract_features(feature_extractor, X_train, device)

    log.info("Extracting TEST features...")
    X_test_feat = extract_features(feature_extractor, X_test, device)

    log.info("Feature shape: %s", X_train_feat.shape)

    # ── RIDGE CLASSIFIER ──
    log.info("Training Ridge classifier...")
    clf = RidgeClassifier(alpha=1.0)
    clf.fit(X_train_feat, y_train)

    # ── SANITY CHECK (label shuffle) ──
    from sklearn.utils import shuffle

    y_shuffled = shuffle(y_train, random_state=42)

    clf_shuffle = RidgeClassifier(alpha=1.0)
    clf_shuffle.fit(X_train_feat, y_shuffled)

    pred_shuffle = clf_shuffle.predict(X_test_feat)
    print("SHUFFLE F1:", f1_score(y_test, pred_shuffle))

    # ── EVALUATION ──
    y_pred = clf.predict(X_test_feat)

    ba = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    log.info("Balanced Accuracy: %.4f", ba)
    log.info("F1: %.4f", f1)

    log.info("\nClassification Report:")
    log.info(classification_report(y_test, y_pred))

    # ─────────────────────────────────────────────
    # SAVE MODEL
    # ─────────────────────────────────────────────

    model_bundle = {
        "feature_extractor_state_dict": feature_extractor.state_dict(),
        "ridge": clf
    }

    with open(os.path.join(args.out, "mcnn_bundle.pkl"), "wb") as f:
        pickle.dump(model_bundle, f)

    torch.save(
        feature_extractor.state_dict(),
        os.path.join(args.out, "mcnn_extractor.pt")
    )

    np.save(os.path.join(args.out, "ridge_coef.npy"), clf.coef_)
    np.save(os.path.join(args.out, "ridge_intercept.npy"), clf.intercept_)

    log.info("\nSaved:")
    log.info("  ✔ mcnn_bundle.pkl")
    log.info("  ✔ mcnn_extractor.pt")
    log.info("  ✔ ridge_coef.npy")
    log.info("  ✔ ridge_intercept.npy")


if __name__ == "__main__":
    main()