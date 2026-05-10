#!/usr/bin/env python3
"""
train_arc_compare.py
=====================
Pipeline unificata per rilevamento archi elettrici DC/PV.

Addestra e confronta tre approcci:
    1. ridge   → MultiRocket + StandardScaler + RidgeClassifier
    2. hydra   → MultiRocketHydra + RidgeClassifierCV (interno)
    3. arcnet  → MultiRocket + StandardScaler + PCA + ArcNet (ONNX export)

Output:
    results/<model>/config.json      metriche e config
    results/<model>/bundle.pkl       bundle completo per inference
    results/arcnet/arcnet.onnx       modello ONNX per ST Edge AI
    results/arcnet/scaler.h          StandardScaler in C
    results/arcnet/pca.h             PCA transform in C
    results/comparison.json          confronto completo (metriche + risorse)

USO:
    # Addestra tutti e tre
    python train_arc_compare.py train.npz test.npz

    # Solo uno specifico
    python train_arc_compare.py train.npz test.npz --models ridge arcnet

    # Parametri custom
    python train_arc_compare.py train.npz test.npz \\
        --downsample 4 \\
        --pca-components 128 \\
        --hidden 64 32 \\
        --epochs 40

REQUISITI:
    pip install aeon scikit-learn numpy torch onnx
"""

import argparse
import json
import os
import time
import pickle
import numpy as np

# ── Torch (opzionale, solo per arcnet) ────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ── Sklearn ────────────────────────────────────────────────────────────────────
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, roc_auc_score,
    classification_report, confusion_matrix,
)

# ── Aeon ───────────────────────────────────────────────────────────────────────
from aeon.transformations.collection.convolution_based import MultiRocket
from aeon.classification.convolution_based import MultiRocketHydraClassifier

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
if TORCH_AVAILABLE:
    torch.manual_seed(RANDOM_STATE)


# =============================================================================
# UTILITIES
# =============================================================================

def banner(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


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
# FEATURE SCALER
# =============================================================================

class FeatureScaler:
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
# MULTIROCKET TRANSFORM (condiviso da ridge e arcnet)
# =============================================================================

def fit_transform_rocket(X_train, X_test):
    print("  Fitting MultiRocket...")
    t0 = time.time()
    tr = MultiRocket(random_state=RANDOM_STATE, n_jobs=1)
    F_train = tr.fit_transform(X_train).astype(np.float32)
    F_test  = tr.transform(X_test).astype(np.float32)
    print(f"  Fatto in {time.time()-t0:.1f}s | shape: {F_train.shape}")
    return F_train, F_test, tr


# =============================================================================
# PCA
# =============================================================================

def fit_pca(F_train, F_test, n_components):
    print(f"  Fitting PCA({n_components})...")
    t0  = time.time()
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    P_train = pca.fit_transform(F_train).astype(np.float32)
    P_test  = pca.transform(F_test).astype(np.float32)
    var = pca.explained_variance_ratio_.sum() * 100
    print(f"  Fatto in {time.time()-t0:.1f}s | varianza spiegata: {var:.1f}%")
    return P_train, P_test, pca


# =============================================================================
# ARCNET — rete neurale leggera per STM32
# =============================================================================

class ArcNet(nn.Module):
    def __init__(self, n_features, hidden=(64, 32), dropout=0.3):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h),
                       nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def make_sampler(y):
    counts  = np.bincount(y.astype(int))
    weights = 1.0 / counts
    sw = torch.tensor([weights[int(l)] for l in y])
    return WeightedRandomSampler(sw, len(sw))


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for Xb, yb in loader:
        Xb = Xb.to(device)
        yb = yb.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(Xb)
    return total / len(loader.dataset)


def evaluate_nn(model, loader, device):
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
# METRICHE COMUNI
# =============================================================================

def compute_metrics(labels, preds, probs):
    return {
        "accuracy":          float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "f1":                float(f1_score(labels, preds, zero_division=0)),
        "roc_auc":           float(roc_auc_score(labels, probs)),
        "confusion_matrix":  confusion_matrix(labels, preds).tolist(),
    }


def print_metrics(metrics, labels, preds):
    print(classification_report(labels, preds,
                                 target_names=["No Arc", "Arc"], digits=4))
    print(f"  Accuracy          : {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy : {metrics['balanced_accuracy']:.4f}")
    print(f"  F1 Score          : {metrics['f1']:.4f}")
    print(f"  ROC AUC           : {metrics['roc_auc']:.4f}")
    print(f"  Confusion Matrix  : {metrics['confusion_matrix']}")


# =============================================================================
# EXPORT (solo arcnet)
# =============================================================================

def export_onnx(model, n_features, out_dir):
    import onnx
    model.eval()
    device   = next(model.parameters()).device
    dummy    = torch.randn(1, n_features).to(device)
    out_path = os.path.join(out_dir, "arcnet.onnx")
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=11, do_constant_folding=True,
    )
    m = onnx.load(out_path)
    m.ir_version = 7
    inputs_to_keep = [i for i in m.graph.input if i.name == "input"]
    del m.graph.input[:]
    m.graph.input.extend(inputs_to_keep)
    onnx.save(m, out_path)
    print("  ONNX salvato:", out_path)


def export_scaler_header(scaler, out_dir):
    mean, scale, n = scaler.mean, scaler.scale, len(scaler.mean)
    lines = [
        "/* scaler.h - AUTO-GENERATED */",
        "#ifndef SCALER_H", "#define SCALER_H",
        f"#define SCALER_N_FEATURES {n}", "",
        f"static const float scaler_mean[{n}] = {{",
    ]
    for i in range(0, n, 8):
        lines.append("    " + ", ".join(f"{v:.8f}f" for v in mean[i:i+8]) + ",")
    lines += ["};", "", f"static const float scaler_scale[{n}] = {{"]
    for i in range(0, n, 8):
        lines.append("    " + ", ".join(f"{v:.8f}f" for v in scale[i:i+8]) + ",")
    lines += ["};", "",
              "static inline void scaler_transform(float* feat, int n) {",
              "    for (int i = 0; i < n; i++)",
              "        feat[i] = (feat[i] - scaler_mean[i]) / scaler_scale[i];",
              "}", "#endif"]
    path = os.path.join(out_dir, "scaler.h")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("  scaler.h salvato:", path)


def export_pca_header(pca, scaler_mean, scaler_scale, out_dir):
    components = pca.components_.astype(np.float32)
    mean_pca   = pca.mean_.astype(np.float32)
    n_comp, n_feat = components.shape
    lines = [
        "/* pca.h - AUTO-GENERATED */",
        "#ifndef PCA_H", "#define PCA_H", "#include <stdint.h>", "",
        f"#define PCA_N_COMPONENTS {n_comp}",
        f"#define PCA_N_FEATURES   {n_feat}", "",
        f"static const float pca_scaler_mean[{n_feat}] = {{",
    ]
    for i in range(0, n_feat, 8):
        lines.append("    " + ", ".join(f"{v:.8f}f" for v in scaler_mean[i:i+8]) + ",")
    lines += ["};", "", f"static const float pca_scaler_scale[{n_feat}] = {{"]
    for i in range(0, n_feat, 8):
        lines.append("    " + ", ".join(f"{v:.8f}f" for v in scaler_scale[i:i+8]) + ",")
    lines += ["};", "", f"static const float pca_mean[{n_feat}] = {{"]
    for i in range(0, n_feat, 8):
        lines.append("    " + ", ".join(f"{v:.8f}f" for v in mean_pca[i:i+8]) + ",")
    lines += ["};", "",
              f"/* PCA components — {n_comp * n_feat * 4 / 1024 / 1024:.1f} MB → Flash esterna */",
              f"static const float pca_components[{n_comp}][{n_feat}] = {{"]
    for c in range(n_comp):
        row = components[c]
        lines.append(f"    {{ /* component {c} */")
        for i in range(0, n_feat, 8):
            lines.append("        " + ", ".join(f"{v:.8f}f" for v in row[i:i+8]) + ",")
        lines.append("    },")
    lines += ["};", "",
              "static inline void pca_transform(const float* in, float* out, float* tmp) {",
              "    for (int i = 0; i < PCA_N_FEATURES; i++)",
              "        tmp[i] = (in[i] - pca_scaler_mean[i]) / pca_scaler_scale[i] - pca_mean[i];",
              "    for (int c = 0; c < PCA_N_COMPONENTS; c++) {",
              "        float s = 0.0f;",
              "        for (int i = 0; i < PCA_N_FEATURES; i++) s += pca_components[c][i] * tmp[i];",
              "        out[c] = s;",
              "    }",
              "}", "#endif"]
    path = os.path.join(out_dir, "pca.h")
    # riga ~308
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("  pca.h salvato:", path)


# =============================================================================
# STIMA RISORSE EMBEDDED
# =============================================================================

def estimate_resources_ridge(transformer, scaler, n_features):
    """Ridge: bisogna tenere in Flash i pesi (n_features float) + scaler."""
    # Ridge ha coef_ shape (1, n_features) per binario
    n_kernels = len(transformer.parameter[0]) if hasattr(transformer, 'parameter') else 0
    rocket_flash_kb  = (n_kernels * 3 * 4) / 1024 if n_kernels else 0
    scaler_flash_kb  = (n_features * 2 * 4) / 1024         # mean + scale
    ridge_flash_kb   = (n_features * 4) / 1024              # coef_ (1 classe binaria)
    feat_ram_kb      = (n_features * 4) / 1024              # buffer feature
    total_flash_kb   = rocket_flash_kb + scaler_flash_kb + ridge_flash_kb
    return {
        "flash_interna_kb": round(total_flash_kb, 1),
        "flash_esterna_mb": 0.0,
        "ram_kb":           round(feat_ram_kb, 1),
        "dettaglio": {
            "rocket_kernels_flash_kb": round(rocket_flash_kb, 1),
            "scaler_flash_kb":         round(scaler_flash_kb, 1),
            "ridge_weights_flash_kb":  round(ridge_flash_kb, 1),
            "feature_buffer_ram_kb":   round(feat_ram_kb, 1),
        }
    }


def estimate_resources_hydra(n_features_hydra):
    """Hydra: feature più numerose, Ridge interno."""
    scaler_flash_kb = (n_features_hydra * 2 * 4) / 1024
    ridge_flash_kb  = (n_features_hydra * 4) / 1024
    feat_ram_kb     = (n_features_hydra * 4) / 1024
    total_flash_kb  = scaler_flash_kb + ridge_flash_kb
    return {
        "flash_interna_kb": round(total_flash_kb, 1),
        "flash_esterna_mb": 0.0,
        "ram_kb":           round(feat_ram_kb, 1),
        "dettaglio": {
            "scaler_flash_kb":        round(scaler_flash_kb, 1),
            "ridge_weights_flash_kb": round(ridge_flash_kb, 1),
            "feature_buffer_ram_kb":  round(feat_ram_kb, 1),
        }
    }


def estimate_resources_arcnet(transformer, scaler, pca, model, n_features_raw, n_pca):
    n_kernels       = len(transformer.parameter[0]) if hasattr(transformer, 'parameter') else 0
    rocket_flash_kb = (n_kernels * 3 * 4) / 1024 if n_kernels else 0
    scaler_flash_mb = (n_features_raw * 2 * 4) / 1024 / 1024
    pca_flash_mb    = (n_pca * n_features_raw * 4) / 1024 / 1024
    n_params        = sum(p.numel() for p in model.parameters())
    nn_flash_kb     = (n_params * 4) / 1024
    feat_ram_kb     = (n_features_raw * 4) / 1024
    pca_ram_kb      = (n_pca * 4) / 1024
    return {
        "flash_interna_kb": round(rocket_flash_kb + nn_flash_kb, 1),
        "flash_esterna_mb": round(scaler_flash_mb + pca_flash_mb, 2),
        "ram_kb":           round(feat_ram_kb + pca_ram_kb, 1),
        "n_params_nn":      n_params,
        "dettaglio": {
            "rocket_kernels_flash_kb": round(rocket_flash_kb, 1),
            "nn_flash_kb":             round(nn_flash_kb, 1),
            "scaler_flash_mb":         round(scaler_flash_mb, 2),
            "pca_flash_mb":            round(pca_flash_mb, 2),
            "feature_buffer_ram_kb":   round(feat_ram_kb, 1),
            "pca_output_ram_kb":       round(pca_ram_kb, 1),
        }
    }


# =============================================================================
# PIPELINE RIDGE
# =============================================================================

def run_ridge(X_train, y_train, X_test, y_test, args, out_dir):
    banner("PIPELINE 1 — MultiRocket + Ridge")
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.time()

    # Feature extraction
    F_train, F_test, transformer = fit_transform_rocket(X_train, X_test)
    n_features = F_train.shape[1]

    # Scaling
    scaler  = FeatureScaler()
    F_train = scaler.fit_transform(F_train)
    F_test  = scaler.transform(F_test)

    # Fit Ridge
    print("  Fitting RidgeClassifier...")
    t0    = time.time()
    model = RidgeClassifier(alpha=1.0, class_weight="balanced",
                             random_state=RANDOM_STATE)
    model.fit(F_train, y_train)
    train_time = time.time() - t0
    print(f"  Fatto in {train_time:.2f}s")

    # Predict
    preds  = model.predict(F_test)
    scores = model.decision_function(F_test)
    probs  = 1 / (1 + np.exp(-scores))

    metrics = compute_metrics(y_test, preds, probs)
    print_metrics(metrics, y_test, preds)

    resources = estimate_resources_ridge(transformer, scaler, n_features)

    total_time = time.time() - t_start
    result = {
        "model":        "MultiRocket + Ridge",
        "n_features":   n_features,
        "downsample":   args.downsample,
        "train_time_s": round(train_time, 2),
        "total_time_s": round(total_time, 2),
        "metrics":      metrics,
        "resources":    resources,
        "stm32_ready":  "parziale",
        "onnx_export":  False,
    }

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(result, f, indent=2)

    bundle = {"model": model, "transformer": transformer,
              "scaler": scaler, "metrics": metrics}
    with open(os.path.join(out_dir, "bundle.pkl"), "wb") as f:
        pickle.dump(bundle, f)

    print(f"\n  Output in: {out_dir}/")
    return result


# =============================================================================
# PIPELINE HYDRA
# =============================================================================

def run_hydra(X_train, y_train, X_test, y_test, args, out_dir):
    banner("PIPELINE 2 — MultiRocketHydra + Ridge")
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.time()

    clf = MultiRocketHydraClassifier(
        n_kernels=args.hydra_kernels,
        n_groups=args.hydra_groups,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    print(f"  Fitting MultiRocketHydraClassifier "
          f"(kernels={args.hydra_kernels}, groups={args.hydra_groups})...")
    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  Fatto in {train_time:.2f}s")

    preds = clf.predict(X_test)
    try:
        decision = clf.decision_function(X_test)
        probs = 1 / (1 + np.exp(-decision))
    except Exception:
        probs = preds.astype(float)

    metrics = compute_metrics(y_test, preds, probs)
    print_metrics(metrics, y_test, preds)

    # Stima feature Hydra
    n_feat_hydra = args.hydra_kernels * args.hydra_groups * 3 * 2  # approssimazione
    resources = estimate_resources_hydra(n_feat_hydra)

    total_time = time.time() - t_start
    result = {
        "model":        "MultiRocketHydra + Ridge",
        "n_kernels":    args.hydra_kernels,
        "n_groups":     args.hydra_groups,
        "downsample":   args.downsample,
        "train_time_s": round(train_time, 2),
        "total_time_s": round(total_time, 2),
        "metrics":      metrics,
        "resources":    resources,
        "stm32_ready":  "no",
        "onnx_export":  False,
    }

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(result, f, indent=2)

    bundle = {"model": clf, "metrics": metrics}
    with open(os.path.join(out_dir, "bundle.pkl"), "wb") as f:
        pickle.dump(bundle, f)

    print(f"\n  Output in: {out_dir}/")
    return result


# =============================================================================
# PIPELINE ARCNET
# =============================================================================

def run_arcnet(X_train, y_train, X_test, y_test, args, out_dir):
    banner("PIPELINE 3 — MultiRocket + PCA + ArcNet")
    if not TORCH_AVAILABLE:
        print("  [SKIP] PyTorch non disponibile. Installa con: pip install torch")
        return None
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Feature extraction
    F_train, F_test, transformer = fit_transform_rocket(X_train, X_test)
    n_features_raw = F_train.shape[1]

    # Scaling
    scaler  = FeatureScaler()
    F_train = scaler.fit_transform(F_train)
    F_test  = scaler.transform(F_test)

    # PCA
    P_train, P_test, pca = fit_pca(F_train, F_test, args.pca_components)
    n_features = P_train.shape[1]

    # Dataset
    train_ds     = TensorDataset(torch.from_numpy(P_train),
                                  torch.from_numpy(y_train.astype(np.float32)))
    test_ds      = TensorDataset(torch.from_numpy(P_test),
                                  torch.from_numpy(y_test.astype(np.float32)))
    sampler      = make_sampler(y_train)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    # Modello
    model = ArcNet(n_features=n_features,
                   hidden=tuple(args.hidden),
                   dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parametri NN : {n_params:,}  |  Flash stimata: {n_params*4/1024:.1f} KB")

    n_pos      = float(y_train.sum())
    n_neg      = float(len(y_train) - n_pos)
    pos_weight = torch.tensor([n_neg / n_pos]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1   = 0.0
    best_path = os.path.join(out_dir, "best_arcnet.pt")

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        if epoch % max(1, args.epochs // 6) == 0 or epoch == 1:
            labels, preds, probs = evaluate_nn(model, test_loader, device)
            f1  = f1_score(labels, preds, zero_division=0)
            auc = roc_auc_score(labels, probs)
            print(f"  Epoch {epoch:3d}/{args.epochs} | "
                  f"loss={loss:.4f} | F1={f1:.4f} | AUC={auc:.4f}")
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), best_path)

    train_time = time.time() - t0
    print(f"\n  Training: {train_time:.1f}s — Best F1: {best_f1:.4f}")

    # Carica best
    model.load_state_dict(torch.load(best_path, map_location=device))
    labels, preds, probs = evaluate_nn(model, test_loader, device)
    metrics = compute_metrics(labels, preds, probs)
    print_metrics(metrics, labels, preds)

    # Export
    print("\n  Esportazione artefatti...")
    try:
        export_onnx(model, n_features, out_dir)
    except ImportError:
        print("  [WARN] onnx non installato, skip export ONNX")
    export_scaler_header(scaler, out_dir)
    export_pca_header(pca, scaler.mean, scaler.scale, out_dir)

    resources = estimate_resources_arcnet(
        transformer, scaler, pca, model, n_features_raw, args.pca_components)

    total_time = time.time() - t_start
    result = {
        "model":          "MultiRocket + PCA + ArcNet",
        "pca_components": args.pca_components,
        "hidden":         args.hidden,
        "dropout":        args.dropout,
        "epochs":         args.epochs,
        "n_params_nn":    n_params,
        "downsample":     args.downsample,
        "train_time_s":   round(train_time, 2),
        "total_time_s":   round(total_time, 2),
        "metrics":        metrics,
        "resources":      resources,
        "stm32_ready":    "si",
        "onnx_export":    True,
    }

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(result, f, indent=2)

    bundle = {
        "transformer": transformer, "scaler": scaler,
        "pca": pca, "n_features": n_features, "metrics": metrics,
    }
    with open(os.path.join(out_dir, "bundle.pkl"), "wb") as f:
        pickle.dump(bundle, f)

    print(f"\n  Output in: {out_dir}/")
    return result


# =============================================================================
# EXPORT CSV + TXT
# =============================================================================

def export_csv(results, out_dir):
    """Genera due CSV: metriche e risorse embedded."""
    import csv

    # ── CSV metriche ──────────────────────────────────────────────────────────
    metric_path = os.path.join(out_dir, "comparison_metrics.csv")
    metric_fields = ["model", "accuracy", "balanced_accuracy", "f1", "roc_auc",
                     "train_time_s", "total_time_s", "onnx_export", "stm32_ready"]

    with open(metric_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=metric_fields)
        w.writeheader()
        for m, r in results.items():
            cm = r["metrics"].get("confusion_matrix", [[0,0],[0,0]])
            w.writerow({
                "model":             r["model"],
                "accuracy":          round(r["metrics"].get("accuracy", 0), 4),
                "balanced_accuracy": round(r["metrics"].get("balanced_accuracy", 0), 4),
                "f1":                round(r["metrics"].get("f1", 0), 4),
                "roc_auc":           round(r["metrics"].get("roc_auc", 0), 4),
                "train_time_s":      r.get("train_time_s", ""),
                "total_time_s":      r.get("total_time_s", ""),
                "onnx_export":       r.get("onnx_export", False),
                "stm32_ready":       r.get("stm32_ready", "?"),
            })
    print(f"  CSV metriche   : {metric_path}")

    # ── CSV confusion matrix (una riga per modello: TN FP FN TP) ─────────────
    cm_path = os.path.join(out_dir, "comparison_confusion.csv")
    with open(cm_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "TN", "FP", "FN", "TP"])
        w.writeheader()
        for m, r in results.items():
            cm = r["metrics"].get("confusion_matrix", [[0,0],[0,0]])
            w.writerow({
                "model": r["model"],
                "TN": cm[0][0], "FP": cm[0][1],
                "FN": cm[1][0], "TP": cm[1][1],
            })
    print(f"  CSV confusion  : {cm_path}")

    # ── CSV risorse embedded ──────────────────────────────────────────────────
    res_path = os.path.join(out_dir, "comparison_resources.csv")
    res_fields = ["model", "flash_interna_kb", "flash_esterna_mb", "ram_kb",
                  "stm32_ready", "onnx_export"]

    with open(res_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=res_fields)
        w.writeheader()
        for m, r in results.items():
            res = r.get("resources", {})
            w.writerow({
                "model":             r["model"],
                "flash_interna_kb":  res.get("flash_interna_kb", 0),
                "flash_esterna_mb":  res.get("flash_esterna_mb", 0),
                "ram_kb":            res.get("ram_kb", 0),
                "stm32_ready":       r.get("stm32_ready", "?"),
                "onnx_export":       r.get("onnx_export", False),
            })
    print(f"  CSV risorse    : {res_path}")


def export_txt_report(results, out_path):
    """Genera un report testuale leggibile con tutte le tabelle."""

    lines = []
    sep   = "=" * 72
    thin  = "-" * 72

    lines += [
        sep,
        "  ARC DETECTION — CONFRONTO MODELLI",
        "  MultiRocket+Ridge  |  MultiRocketHydra+Ridge  |  MultiRocket+PCA+ArcNet",
        sep, "",
    ]

    models     = list(results.keys())
    col_w      = 20
    label_w    = 28

    def header_row(title):
        h = f"  {title:<{label_w}}"
        for m in models:
            h += f"{results[m]['model'][:col_w]:>{col_w}}"
        return h

    def data_row(label, vals, fmt=".4f", best_is_max=True):
        best = max(vals) if best_is_max else min(vals)
        row  = f"  {label:<{label_w}}"
        for v in vals:
            star = " *" if v == best else "  "
            row += f"{v:{fmt}}{star}".rjust(col_w)
        return row

    # ── Tabella 1: Metriche ───────────────────────────────────────────────────
    lines += ["  TABELLA 1 — METRICHE DI PERFORMANCE", thin]
    lines.append(header_row("Metrica"))
    lines.append(thin)

    metric_defs = [
        ("Accuracy",          "accuracy",          True),
        ("Balanced Accuracy", "balanced_accuracy",  True),
        ("F1 Score",          "f1",                 True),
        ("ROC AUC",           "roc_auc",            True),
    ]
    for label, key, best_max in metric_defs:
        vals = [results[m]["metrics"].get(key, 0) for m in models]
        lines.append(data_row(label, vals, ".4f", best_max))

    lines += [thin, "  * = miglior valore per quella metrica", ""]

    # ── Tabella 2: Tempi ──────────────────────────────────────────────────────
    lines += ["  TABELLA 2 — TEMPI", thin]
    lines.append(header_row("Tempo"))
    lines.append(thin)

    time_defs = [
        ("Training (s)",  "train_time_s", False),
        ("Totale (s)",    "total_time_s", False),
    ]
    for label, key, best_max in time_defs:
        vals = [results[m].get(key, 0) for m in models]
        lines.append(data_row(label, vals, ".1f", best_max))

    lines += [thin, ""]

    # ── Tabella 3: Risorse embedded ───────────────────────────────────────────
    lines += ["  TABELLA 3 — RISORSE EMBEDDED (STM32)", thin]
    lines.append(header_row("Risorsa"))
    lines.append(thin)

    res_defs = [
        ("Flash interna (KB)", "flash_interna_kb", False),
        ("Flash esterna (MB)", "flash_esterna_mb", False),
        ("RAM necessaria (KB)","ram_kb",            False),
    ]
    for label, key, best_max in res_defs:
        vals = [results[m]["resources"].get(key, 0) for m in models]
        lines.append(data_row(label, vals, ".1f", best_max))

    lines.append(thin)

    # STM32 ready e ONNX
    stm_row  = f"  {'STM32 ready':<{label_w}}"
    onnx_row = f"  {'ONNX export':<{label_w}}"
    for m in models:
        stm_row  += f"{results[m].get('stm32_ready','?'):>{col_w}}"
        onnx_row += f"{'si' if results[m].get('onnx_export') else 'no':>{col_w}}"
    lines += [stm_row, onnx_row, thin, ""]

    # ── Tabella 4: Confusion matrices ─────────────────────────────────────────
    lines += ["  TABELLA 4 — CONFUSION MATRICES", thin]
    for m in models:
        cm = results[m]["metrics"].get("confusion_matrix", [[0,0],[0,0]])
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        name = results[m]["model"]
        lines += [
            f"  {name}",
            f"    {'':20} {'Pred No Arc':>14} {'Pred Arc':>14}",
            f"    {'Actual No Arc':20} {tn:>14} {fp:>14}",
            f"    {'Actual Arc':20} {fn:>14} {tp:>14}",
            "",
        ]

    # ── Dettaglio risorse per modello ─────────────────────────────────────────
    lines += ["  TABELLA 5 — DETTAGLIO RISORSE PER MODELLO", thin]
    for m in models:
        det  = results[m]["resources"].get("dettaglio", {})
        name = results[m]["model"]
        lines.append(f"  {name}")
        for k, v in det.items():
            lines.append(f"    {k:<35} {v}")
        lines.append("")

    lines += [sep, "  Fine report", sep]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Report TXT     : {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Confronto pipeline arc detection: Ridge / Hydra / ArcNet"
    )
    parser.add_argument("train",  help="Path al file train.npz")
    parser.add_argument("test",   help="Path al file test.npz")
    parser.add_argument("--out",  default="./results", help="Directory output")
    parser.add_argument("--models", nargs="+",
                        choices=["ridge", "hydra", "arcnet"],
                        default=["ridge", "hydra", "arcnet"],
                        help="Modelli da addestrare (default: tutti e tre)")

    # Comuni
    parser.add_argument("--downsample",   type=int,   default=4)

    # ArcNet
    parser.add_argument("--pca-components", type=int,   default=256)
    parser.add_argument("--hidden",          type=int,   nargs="+", default=[64, 32])
    parser.add_argument("--dropout",         type=float, default=0.3)
    parser.add_argument("--epochs",          type=int,   default=30)
    parser.add_argument("--batch-size",      type=int,   default=64)
    parser.add_argument("--lr",              type=float, default=1e-3)

    # Hydra
    parser.add_argument("--hydra-kernels", type=int, default=8)
    parser.add_argument("--hydra-groups",  type=int, default=4)

    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ── Carica dati (una volta sola) ──────────────────────────────────────────
    banner("LOAD DATASET")
    X_train, y_train = load_dataset(args.train, args.downsample)
    X_test,  y_test  = load_dataset(args.test,  args.downsample)
    print(f"  Train: {X_train.shape}  labels: {np.bincount(y_train)}")
    print(f"  Test : {X_test.shape}   labels: {np.bincount(y_test)}")

    # ── Addestramento ─────────────────────────────────────────────────────────
    results = {}

    if "ridge" in args.models:
        results["ridge"] = run_ridge(
            X_train, y_train, X_test, y_test, args,
            os.path.join(args.out, "ridge"))

    if "hydra" in args.models:
        results["hydra"] = run_hydra(
            X_train, y_train, X_test, y_test, args,
            os.path.join(args.out, "hydra"))

    if "arcnet" in args.models:
        r = run_arcnet(
            X_train, y_train, X_test, y_test, args,
            os.path.join(args.out, "arcnet"))
        if r is not None:
            results["arcnet"] = r

    if not results:
        print("Nessun modello addestrato.")
        return

    # ── Confronto finale ──────────────────────────────────────────────────────
    banner("CONFRONTO FINALE")

    metric_labels = {
        "accuracy":          "Accuracy",
        "balanced_accuracy": "Balanced Accuracy",
        "f1":                "F1 Score",
        "roc_auc":           "ROC AUC",
    }

    # Tabella metriche
    col_w = 22
    header_row = f"{'Metrica':<25}" + "".join(
        f"{results[m]['model'][:col_w]:>{col_w}}" for m in results)
    print(header_row)
    print("-" * len(header_row))

    for key, label in metric_labels.items():
        vals = {m: results[m]["metrics"].get(key, 0) for m in results}
        best = max(vals.values())
        row  = f"{label:<25}"
        for m in results:
            v    = vals[m]
            star = " ★" if v == best else "  "
            row += f"{v:.4f}{star}".rjust(col_w)
        print(row)

    # Tabella risorse
    print("\n")
    res_labels = [
        ("flash_interna_kb", "Flash interna (KB)",  True),
        ("flash_esterna_mb", "Flash esterna (MB)",  True),
        ("ram_kb",           "RAM (KB)",             True),
    ]
    print(f"{'Risorsa':<25}" + "".join(
        f"{results[m]['model'][:col_w]:>{col_w}}" for m in results))
    print("-" * len(header_row))
    for key, label, lower in res_labels:
        vals = {m: results[m]["resources"].get(key, 0) for m in results}
        best = min(vals.values())
        row  = f"{label:<25}"
        for m in results:
            v    = vals[m]
            star = " ★" if (lower and v == best) else "  "
            row += f"{v:.1f}{star}".rjust(col_w)
        print(row)

    print()
    for m in results:
        stm = results[m].get("stm32_ready", "?")
        onnx = "✅" if results[m].get("onnx_export") else "❌"
        print(f"  {results[m]['model']:<40} STM32={stm}  ONNX={onnx}")

    # ── Salva JSON confronto ──────────────────────────────────────────────────
    comparison_path = os.path.join(args.out, "comparison.json")
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Confronto JSON: {comparison_path}")


    # ── Genera CSV e TXT ──────────────────────────────────────────────────────
    banner("EXPORT CSV + TXT")
    export_csv(results, args.out)
    txt_path = os.path.join(args.out, "comparison_report.txt")
    export_txt_report(results, txt_path)

    banner("DONE")
    print(f"""
  Output in: {args.out}/
    ridge/                   bundle Ridge
    hydra/                   bundle Hydra
    arcnet/                  bundle ArcNet + ONNX + header C
    comparison.json          dati completi
    comparison_metrics.csv   metriche per modello
    comparison_confusion.csv confusion matrix
    comparison_resources.csv risorse embedded STM32
    comparison_report.txt    report testuale completo

  Prossimi step STM32:
    1. Carica arcnet/arcnet.onnx su ST Edge AI Developer Cloud
    2. Usa arcnet/scaler.h e arcnet/pca.h nel firmware C
    3. Implementa kernel MultiRocket in C
""")

if __name__ == "__main__":
    main()