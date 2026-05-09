#!/usr/bin/env python3
"""
train_mrh_nn.py
===============
Pipeline:
    MultiRocket → PCA(256) → ArcNet → arcnet.onnx

USO:
    python train_mrh_nn.py train.npz test.npz
"""

import argparse
import json
import os
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from aeon.transformations.collection.convolution_based import MultiRocket

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# =============================================================================
# LOAD DATASET
# =============================================================================
def load_dataset(path, downsample=4):
    data = np.load(path)
    X = data["X"]
    y = data["y"]
    if X.ndim == 2:
        X = X[:, np.newaxis, :]
    if downsample > 1:
        X = X[:, :, ::downsample]
    return X.astype(np.float32), y.astype(np.int64)


# =============================================================================
# MULTIROCKET TRANSFORM
# =============================================================================
def fit_transform_rocket(X_train, X_test):
    print("  Fitting MultiRocket...")
    t0 = time.time()

    tr = MultiRocket(
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    F_train = tr.fit_transform(X_train)
    F_test  = tr.transform(X_test)

    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Feature shape train : {F_train.shape}")
    print(f"  Feature shape test  : {F_test.shape}")

    return F_train.astype(np.float32), F_test.astype(np.float32), tr


# =============================================================================
# STANDARD SCALER float32
# =============================================================================
class FeatureScaler:

    def __init__(self):
        self.mean  = None
        self.scale = None

    def fit(self, X):
        self.mean  = X.mean(axis=0).astype(np.float32)
        self.scale = X.std(axis=0).astype(np.float32)
        self.scale[self.scale < 1e-8] = 1.0
        return self

    def transform(self, X):
        return ((X - self.mean) / self.scale).astype(np.float32)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# =============================================================================
# PCA WRAPPER
# =============================================================================
def fit_pca(F_train, F_test, n_components=256):
    print(f"\n  Fitting PCA({n_components})...")
    t0 = time.time()

    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    P_train = pca.fit_transform(F_train).astype(np.float32)
    P_test  = pca.transform(F_test).astype(np.float32)

    var_explained = pca.explained_variance_ratio_.sum() * 100
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Variance explained  : {var_explained:.2f}%")
    print(f"  Shape after PCA     : {P_train.shape}")

    return P_train, P_test, pca


# =============================================================================
# ARCNET — Dense NN leggera per STM32
# =============================================================================
class ArcNet(nn.Module):

    def __init__(self, n_features, hidden=(64, 32), dropout=0.3):
        super().__init__()

        layers = []
        in_dim = n_features

        for h in hidden:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h

        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# =============================================================================
# WEIGHTED SAMPLER
# =============================================================================
def make_sampler(y):
    counts  = np.bincount(y.astype(int))
    weights = 1.0 / counts
    sw = torch.tensor([weights[int(l)] for l in y])
    return WeightedRandomSampler(sw, len(sw))


# =============================================================================
# TRAIN / EVAL
# =============================================================================
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for Xb, yb in loader:
        Xb = Xb.to(device)
        yb = yb.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(Xb)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    logits_all, labels_all = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            logits_all.append(model(Xb.to(device)).cpu())
            labels_all.append(yb)
    logits = torch.cat(logits_all).squeeze(1).numpy()
    labels = torch.cat(labels_all).numpy()
    probs  = 1 / (1 + np.exp(-logits))
    preds  = (probs >= 0.5).astype(int)
    return labels, preds, probs


# =============================================================================
# EXPORT ONNX (solo ArcNet — input = PCA features)
# =============================================================================
def export_onnx(model, n_features, out_dir):
    import onnx

    model.eval()
    device   = next(model.parameters()).device
    dummy    = torch.randn(1, n_features).to(device)
    out_path = os.path.join(out_dir, "arcnet.onnx")

    torch.onnx.export(
        model, dummy, out_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=11,
        do_constant_folding=True,
    )

    # Fix ir_version per ST Edge AI
    m = onnx.load(out_path)
    m.ir_version = 7

    inputs_to_keep = [i for i in m.graph.input if i.name == "input"]
    del m.graph.input[:]
    m.graph.input.extend(inputs_to_keep)

    onnx.save(m, out_path)
    print("ONNX saved:", out_path)
    return out_path


# =============================================================================
# EXPORT SCALER HEADER C
# =============================================================================
def export_scaler_header(scaler, out_dir):
    mean  = scaler.mean
    scale = scaler.scale
    n     = len(mean)

    lines = []
    lines.append("/* scaler.h - AUTO-GENERATED, DO NOT EDIT */")
    lines.append("#ifndef SCALER_H")
    lines.append("#define SCALER_H")
    lines.append(f"#define SCALER_N_FEATURES {n}")
    lines.append("")
    lines.append(f"static const float scaler_mean[{n}] = {{")
    for i in range(0, n, 8):
        chunk = mean[i:i+8]
        lines.append("    " + ", ".join(f"{v:.8f}f" for v in chunk) + ",")
    lines.append("};")
    lines.append("")
    lines.append(f"static const float scaler_scale[{n}] = {{")
    for i in range(0, n, 8):
        chunk = scale[i:i+8]
        lines.append("    " + ", ".join(f"{v:.8f}f" for v in chunk) + ",")
    lines.append("};")
    lines.append("")
    lines.append("static inline void scaler_transform(float* feat, int n)")
    lines.append("{")
    lines.append("    for (int i = 0; i < n; i++)")
    lines.append("        feat[i] = (feat[i] - scaler_mean[i]) / scaler_scale[i];")
    lines.append("}")
    lines.append("#endif /* SCALER_H */")

    path = os.path.join(out_dir, "scaler.h")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Scaler header saved:", path)


# =============================================================================
# EXPORT PCA HEADER C
# =============================================================================
def export_pca_header(pca, scaler_mean, scaler_scale, out_dir):
    """
    Esporta PCA come header C.
    Su STM32: i componenti vanno in Flash esterna (128MB disponibili).
    """
    components = pca.components_.astype(np.float32)  # (n_components, n_features)
    mean_pca   = pca.mean_.astype(np.float32)         # (n_features,)
    n_comp, n_feat = components.shape

    lines = []
    lines.append("/* pca.h - AUTO-GENERATED, DO NOT EDIT */")
    lines.append("/* Mappa in Flash esterna su STM32H7    */")
    lines.append("#ifndef PCA_H")
    lines.append("#define PCA_H")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append(f"#define PCA_N_COMPONENTS {n_comp}")
    lines.append(f"#define PCA_N_FEATURES   {n_feat}")
    lines.append("")

    # Mean scaler (per normalizzare prima di PCA)
    lines.append("/* StandardScaler mean (applicato prima di PCA) */")
    lines.append(f"static const float pca_scaler_mean[{n_feat}] = {{")
    for i in range(0, n_feat, 8):
        chunk = scaler_mean[i:i+8]
        lines.append("    " + ", ".join(f"{v:.8f}f" for v in chunk) + ",")
    lines.append("};")
    lines.append("")

    lines.append("/* StandardScaler scale */")
    lines.append(f"static const float pca_scaler_scale[{n_feat}] = {{")
    for i in range(0, n_feat, 8):
        chunk = scaler_scale[i:i+8]
        lines.append("    " + ", ".join(f"{v:.8f}f" for v in chunk) + ",")
    lines.append("};")
    lines.append("")

    # PCA mean
    lines.append("/* PCA mean */")
    lines.append(f"static const float pca_mean[{n_feat}] = {{")
    for i in range(0, n_feat, 8):
        chunk = mean_pca[i:i+8]
        lines.append("    " + ", ".join(f"{v:.8f}f" for v in chunk) + ",")
    lines.append("};")
    lines.append("")

    # PCA components — flat row-major
    lines.append("/* PCA components (n_components x n_features), row-major */")
    lines.append(f"/* Size: {n_comp * n_feat * 4 / 1024 / 1024:.1f} MB -> Flash esterna */")
    lines.append(f"static const float pca_components[{n_comp}][{n_feat}] = {{")
    for c in range(n_comp):
        row = components[c]
        lines.append(f"    {{ /* component {c} */")
        for i in range(0, n_feat, 8):
            chunk = row[i:i+8]
            lines.append("        " + ", ".join(f"{v:.8f}f" for v in chunk) + ",")
        lines.append("    },")
    lines.append("};")
    lines.append("")

    # Funzione transform inline
    lines.append("""
/* Applica scaler + PCA a un vettore di feature.
   feat_in  : input  (n_features float)
   feat_out : output (n_components float)
   tmp_buf  : buffer temporaneo (n_features float) - alloca in RAM
*/
static inline void pca_transform(
    const float* feat_in,
    float*       feat_out,
    float*       tmp_buf)
{
    /* 1. StandardScaler */
    for (int i = 0; i < PCA_N_FEATURES; i++)
        tmp_buf[i] = (feat_in[i] - pca_scaler_mean[i]) / pca_scaler_scale[i];

    /* 2. Sottrai media PCA */
    for (int i = 0; i < PCA_N_FEATURES; i++)
        tmp_buf[i] -= pca_mean[i];

    /* 3. Proietta sui componenti */
    for (int c = 0; c < PCA_N_COMPONENTS; c++)
    {
        float s = 0.0f;
        for (int i = 0; i < PCA_N_FEATURES; i++)
            s += pca_components[c][i] * tmp_buf[i];
        feat_out[c] = s;
    }
}
""")
    lines.append("#endif /* PCA_H */")

    path = os.path.join(out_dir, "pca.h")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("PCA header saved:", path)

    # Stima dimensioni
    comp_mb  = n_comp * n_feat * 4 / 1024 / 1024
    mean_kb  = n_feat * 4 / 1024
    print(f"  PCA components : {comp_mb:.1f} MB  (Flash esterna)")
    print(f"  PCA mean       : {mean_kb:.1f} KB")


# =============================================================================
# MAIN
# =============================================================================
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("train")
    parser.add_argument("test")
    parser.add_argument("--out",          default="./results_mrh_nn")
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--batch-size",   type=int,   default=64)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--hidden",       type=int,   nargs="+", default=[64, 32])
    parser.add_argument("--dropout",      type=float, default=0.3)
    parser.add_argument("--downsample",   type=int,   default=4)
    parser.add_argument("--pca-components", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("LOAD DATASET")
    print("=" * 60)

    X_train, y_train = load_dataset(args.train, args.downsample)
    X_test,  y_test  = load_dataset(args.test,  args.downsample)
    print("Train:", X_train.shape, y_train.shape)
    print("Test :", X_test.shape,  y_test.shape)

    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MULTIROCKET TRANSFORM")
    print("=" * 60)

    F_train, F_test, transformer = fit_transform_rocket(X_train, X_test)

    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SCALING")
    print("=" * 60)

    scaler  = FeatureScaler()
    F_train = scaler.fit_transform(F_train)
    F_test  = scaler.transform(F_test)
    print(f"Feature range: {F_train.min():.3f} -> {F_train.max():.3f}")

    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"PCA ({args.pca_components} components)")
    print("=" * 60)

    P_train, P_test, pca = fit_pca(F_train, F_test, args.pca_components)
    n_features = P_train.shape[1]

    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAIN ARCNET")
    print("=" * 60)

    train_ds = TensorDataset(
        torch.from_numpy(P_train),
        torch.from_numpy(y_train.astype(np.float32))
    )
    test_ds = TensorDataset(
        torch.from_numpy(P_test),
        torch.from_numpy(y_test.astype(np.float32))
    )

    sampler      = make_sampler(y_train)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    model = ArcNet(
        n_features=n_features,
        hidden=tuple(args.hidden),
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parametri NN : {n_params:,}")
    print(f"Flash stimata: {n_params*4/1024:.1f} KB")

    n_pos      = float(y_train.sum())
    n_neg      = float(len(y_train) - n_pos)
    pos_weight = torch.tensor([n_neg / n_pos]).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1   = 0.0
    best_path = os.path.join(args.out, "best_arcnet.pt")
    t0        = time.time()

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            labels, preds, probs = evaluate(model, test_loader, device)
            f1  = f1_score(labels, preds, zero_division=0)
            auc = roc_auc_score(labels, probs)
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"loss={loss:.4f} | F1={f1:.4f} | AUC={auc:.4f}")
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), best_path)

    print(f"\nTraining: {time.time()-t0:.1f}s — Best F1: {best_f1:.4f}")

    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("METRICS (best model)")
    print("=" * 60)

    model.load_state_dict(torch.load(best_path, map_location=device))
    labels, preds, probs = evaluate(model, test_loader, device)

    acc = accuracy_score(labels, preds)
    ba  = balanced_accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, zero_division=0)
    auc = roc_auc_score(labels, probs)

    print(classification_report(
        labels, preds,
        target_names=["No Arc", "Arc"],
        digits=4
    ))
    print("Accuracy          :", round(acc, 4))
    print("Balanced Accuracy :", round(ba,  4))
    print("F1 Score          :", round(f1,  4))
    print("ROC AUC           :", round(auc, 4))
    print(confusion_matrix(labels, preds))

    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXPORT")
    print("=" * 60)

    export_onnx(model, n_features, args.out)
    export_scaler_header(scaler, args.out)
    export_pca_header(pca, scaler.mean, scaler.scale, args.out)

    # Salva bundle completo
    bundle = {
        "transformer":   transformer,
        "scaler_mean":   scaler.mean,
        "scaler_scale":  scaler.scale,
        "pca":           pca,
        "n_features":    n_features,
        "hidden":        args.hidden,
        "metrics": {
            "accuracy":          float(acc),
            "balanced_accuracy": float(ba),
            "f1":                float(f1),
            "roc_auc":           float(auc),
        }
    }

    with open(os.path.join(args.out, "bundle.pkl"), "wb") as f:
        pickle.dump(bundle, f)

    cfg = {
        "model":          "MultiRocket + PCA + ArcNet",
        "n_features_raw": 49728,
        "pca_components": args.pca_components,
        "n_features_pca": n_features,
        "hidden":         args.hidden,
        "downsample":     args.downsample,
        "n_params_nn":    n_params,
        "flash_kb_nn":    round(n_params * 4 / 1024, 1),
        "metrics":        bundle["metrics"],
    }

    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MEMORY ESTIMATE STM32H7")
    print("=" * 60)

    rocket_c_kb  = (len(transformer.parameter[0]) * 3 * 4) / 1024
    scaler_mb    = (len(scaler.mean) * 2 * 4) / 1024 / 1024
    pca_mb       = (args.pca_components * 49728 * 4) / 1024 / 1024
    nn_kb        = n_params * 4 / 1024
    feat_ram_kb  = 49728 * 4 / 1024
    pca_ram_kb   = args.pca_components * 4 / 1024

    print(f"  Rocket kernels (Flash int) : ~{rocket_c_kb:.1f} KB")
    print(f"  Scaler mean+scale (Flash)  : ~{scaler_mb:.1f} MB  -> Flash esterna")
    print(f"  PCA components (Flash)     : ~{pca_mb:.1f} MB  -> Flash esterna")
    print(f"  ArcNet (Flash int)         : ~{nn_kb:.1f} KB")
    print(f"  Feature buffer (RAM)       : ~{feat_ram_kb:.1f} KB")
    print(f"  PCA output buffer (RAM)    : ~{pca_ram_kb:.1f} KB")
    print()
    print(f"  Flash interna usata        : ~{rocket_c_kb + nn_kb:.1f} KB")
    print(f"  Flash esterna usata        : ~{scaler_mb + pca_mb:.1f} MB")
    print(f"  RAM usata                  : ~{feat_ram_kb + pca_ram_kb:.1f} KB")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"""
Output in: {args.out}/
    arcnet.onnx  -> ST Edge AI Developer Cloud
    scaler.h     -> StandardScaler in C
    pca.h        -> PCA transform in C (Flash esterna)
    bundle.pkl   -> tutto per inference PC
    config.json  -> metriche e configurazione

Prossimi step:
    Step 2: export_rocket_transform.c  (kernel MultiRocket in C)
    Step 3: validate_pipeline.py       (verifica C vs Python)
    Step 4: STM32CubeIDE + ST Edge AI
""")


if __name__ == "__main__":
    main()