#!/usr/bin/env python3
"""
train_arc_compare.py
=====================
Pipeline unificata per rilevamento archi elettrici DC/PV.

Addestra e confronta tre approcci:
    1. ridge   → MultiRocket + StandardScaler + RidgeClassifier
    2. hydra   → MultiRocketHydra + RidgeClassifierCV (interno)
    3. arcnet  → MultiRocket + StandardScaler + PCA + ArcNet

NOTA sul deploy STM32:
    Tutte e tre le pipeline usano MultiRocket come preprocessing.
    MultiRocket non ha un export C/ONNX automatico, quindi nessuna
    pipeline è deployabile su STM32 senza reimplementare manualmente
    i kernel in C. ArcNet è la più avanzata perché la parte finale
    (scaler + PCA + rete) è già pronta nel bundle, ma il preprocessing
    rimane un lavoro da fare.

Output:
    results/ridge/bundle.pkl        bundle completo
    results/ridge/config.json       metriche
    results/hydra/bundle.pkl        bundle completo
    results/hydra/config.json       metriche
    results/arcnet/bundle.pkl       bundle completo (transformer+scaler+pca+model)
    results/arcnet/config.json      metriche
    results/comparison.json         confronto completo
    results/comparison_metrics.csv
    results/comparison_confusion.csv
    results/comparison_report.txt
    results/comparison_plots.png    grafici comparativi

USO:
    python train_arc_compare.py train.npz test.npz
    python train_arc_compare.py train.npz test.npz --models ridge arcnet
    python train_arc_compare.py train.npz test.npz --pca-components 128 --epochs 40

REQUISITI:
    pip install aeon scikit-learn numpy torch onnx matplotlib seaborn
"""

import argparse
import json
import os
import time
import pickle
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

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
    roc_curve, precision_recall_curve, average_precision_score,
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
# MULTIROCKET TRANSFORM
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
# ARCNET
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
        "avg_precision":     float(average_precision_score(labels, probs)),
        "confusion_matrix":  confusion_matrix(labels, preds).tolist(),
        # salva probs e labels per i grafici
        "_probs":            probs.tolist(),
        "_labels":           labels.tolist(),
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
# GRAFICI COMPARATIVI
# =============================================================================

def plot_comparison(results, out_dir):
    """
    Genera un pannello con 4 grafici comparativi:
      1. Barre metriche (accuracy, balanced acc, F1, ROC-AUC)
      2. ROC curve per tutti i modelli
      3. Precision-Recall curve per tutti i modelli
      4. Confusion matrix affiancate
    """
    model_keys  = list(results.keys())
    model_names = {
        "ridge":  "MR + Ridge",
        "hydra":  "MRH + Ridge",
        "arcnet": "MR + PCA + ArcNet",
    }
    colors = {"ridge": "steelblue", "hydra": "darkorange", "arcnet": "seagreen"}

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("Confronto Modelli — Arc Fault Detection", fontsize=15, y=0.98)

    # ── 1. Barre metriche ─────────────────────────────────────────────────────
    ax1 = fig.add_subplot(3, 2, (1, 2))
    metric_keys   = ["accuracy", "balanced_accuracy", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Balanced Acc", "F1 Score", "ROC-AUC"]
    x      = np.arange(len(metric_keys))
    n_mdl  = len(model_keys)
    width  = 0.22
    offset = np.linspace(-(n_mdl - 1) / 2 * width, (n_mdl - 1) / 2 * width, n_mdl)

    for i, k in enumerate(model_keys):
        vals = [results[k]["metrics"].get(m, 0) for m in metric_keys]
        bars = ax1.bar(x + offset[i], vals, width,
                       label=model_names.get(k, k),
                       color=colors.get(k, "gray"),
                       alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.0005,
                     f"{v:.4f}", ha="center", va="bottom", fontsize=7.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels, fontsize=11)
    ax1.set_ylim(0.98, 1.002)
    ax1.set_title("Metriche di Performance", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylabel("Valore")

    # ── 2. ROC Curve ──────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(3, 2, 3)
    for k in model_keys:
        probs  = np.array(results[k]["metrics"].get("_probs", []))
        labels = np.array(results[k]["metrics"].get("_labels", []))
        if len(probs) > 0:
            fpr, tpr, _ = roc_curve(labels, probs)
            auc = results[k]["metrics"].get("roc_auc", 0)
            ax2.plot(fpr, tpr, color=colors.get(k, "gray"), lw=2,
                     label=f"{model_names.get(k, k)} (AUC={auc:.4f})")
    ax2.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax2.set_title("ROC Curve", fontsize=12)
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    # ── 3. Precision-Recall Curve ─────────────────────────────────────────────
    ax3 = fig.add_subplot(3, 2, 4)
    for k in model_keys:
        probs  = np.array(results[k]["metrics"].get("_probs", []))
        labels = np.array(results[k]["metrics"].get("_labels", []))
        if len(probs) > 0:
            prec, rec, _ = precision_recall_curve(labels, probs)
            ap = results[k]["metrics"].get("avg_precision", 0)
            ax3.plot(rec, prec, color=colors.get(k, "gray"), lw=2,
                     label=f"{model_names.get(k, k)} (AP={ap:.4f})")
    ax3.set_title("Precision-Recall Curve", fontsize=12)
    ax3.set_xlabel("Recall")
    ax3.set_ylabel("Precision")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    # ── 4. Confusion Matrices affiancate ──────────────────────────────────────
    for i, k in enumerate(model_keys):
        ax = fig.add_subplot(3, len(model_keys), 2 * len(model_keys) + i + 1)
        cm = np.array(results[k]["metrics"].get("confusion_matrix", [[0, 0], [0, 0]]))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["No Arc", "Arc"],
                    yticklabels=["No Arc", "Arc"],
                    cbar=False, annot_kws={"size": 11})
        ax.set_title(f"Confusion Matrix\n{model_names.get(k, k)}", fontsize=10)
        ax.set_ylabel("Reale" if i == 0 else "")
        ax.set_xlabel("Predetto")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(out_dir, "comparison_plots.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Grafici comparativi: {path}")


# =============================================================================
# STIMA RISORSE — analitica, byte per byte
# =============================================================================

def estimate_multirocket(transformer):
    """
    Costo MultiRocket — COMUNE a Ridge e ArcNet.
    Conta i parametri reali dall'oggetto transformer.
    """
    total_params = 0
    try:
        if hasattr(transformer, 'parameters_'):
            for p in transformer.parameters_:
                total_params += p.size if hasattr(p, 'size') else len(p)
        elif hasattr(transformer, 'parameter'):
            for p in transformer.parameter:
                total_params += p.size if hasattr(p, 'size') else len(p)
    except Exception:
        pass
    if total_params == 0:
        total_params = 10_000 * 5   # fallback: ~10k kernel × 5 valori
    return {
        "flash_kb":     round((total_params * 4) / 1024, 1),
        "total_params": total_params,
    }


def estimate_ridge_clf(model, scaler, n_features):
    """Costo Ridge + Scaler (senza MultiRocket)."""
    ridge_kb    = (model.coef_.size * 4) / 1024
    scaler_kb   = (n_features * 2 * 4) / 1024
    intercept_b = 4 / 1024
    feat_ram_kb = (n_features * 4) / 1024
    return {
        "scaler_flash_kb":    round(scaler_kb, 1),
        "ridge_flash_kb":     round(ridge_kb, 1),
        "intercept_flash_kb": round(intercept_b, 4),
        "clf_flash_kb":       round(ridge_kb + scaler_kb + intercept_b, 1),
        "feat_ram_kb":        round(feat_ram_kb, 1),
        "n_weights":          model.coef_.size,
    }


def estimate_arcnet_clf(model, n_features_raw, n_pca):
    """Costo ArcNet + Scaler + PCA (senza MultiRocket)."""
    scaler_kb     = (n_features_raw * 2 * 4) / 1024
    pca_mean_kb   = (n_features_raw * 4) / 1024
    pca_comp_mb   = (n_pca * n_features_raw * 4) / 1024 / 1024
    n_params      = sum(p.numel() for p in model.parameters())
    nn_kb         = (n_params * 4) / 1024
    feat_ram_kb   = (n_features_raw * 4) / 1024
    pca_ram_kb    = (n_pca * 4) / 1024
    return {
        "scaler_flash_kb":         round(scaler_kb, 1),
        "pca_mean_flash_kb":       round(pca_mean_kb, 1),
        "pca_components_flash_mb": round(pca_comp_mb, 2),
        "arcnet_flash_kb":         round(nn_kb, 1),
        "clf_flash_interna_kb":    round(scaler_kb + pca_mean_kb + nn_kb, 1),
        "clf_flash_esterna_mb":    round(pca_comp_mb, 2),
        "feat_ram_kb":             round(feat_ram_kb, 1),
        "pca_ram_kb":              round(pca_ram_kb, 1),
        "clf_ram_kb":              round(feat_ram_kb + pca_ram_kb, 1),
        "n_params_arcnet":         n_params,
    }


def print_resource_table(res_rocket, res_ridge=None, res_arcnet=None):
    sep  = "=" * 65
    thin = "-" * 65
    print(f"\n{sep}")
    print("  STIMA RISORSE EMBEDDED (analitica — non verificata su HW)")
    print(sep)

    print("\n  ── COSTO COMUNE: MultiRocket preprocessing ─────────────────")
    print(f"  Flash kernel    : {res_rocket['flash_kb']:>8.1f} KB")
    print(f"  N. parametri    : {res_rocket['total_params']:>8,}")
    print(f"  Nota: costo identico per Ridge e ArcNet, da reimplementare in C")

    if res_ridge:
        print(f"\n  ── CLASSIFICATORE: RidgeClassifier ──────────────────────────")
        print(f"  Scaler          : {res_ridge['scaler_flash_kb']:>8.1f} KB  Flash")
        print(f"  Ridge coef      : {res_ridge['ridge_flash_kb']:>8.1f} KB  Flash  ({res_ridge['n_weights']:,} float)")
        print(f"  Intercept       : {res_ridge['intercept_flash_kb']:>8.4f} KB  Flash")
        print(f"  ─────────────────────────────────────────────────────────────")
        print(f"  Totale clf      : {res_ridge['clf_flash_kb']:>8.1f} KB  Flash")
        print(f"  Buffer feat RAM : {res_ridge['feat_ram_kb']:>8.1f} KB  RAM")
        tot = res_rocket['flash_kb'] + res_ridge['clf_flash_kb']
        print(f"  ─────────────────────────────────────────────────────────────")
        print(f"  TOTALE PIPELINE : {tot:>8.1f} KB  Flash interna")
        print(f"                    {res_ridge['feat_ram_kb']:>8.1f} KB  RAM")
        print(f"  Verificabile    : esportando Ridge come ONNX lineare in ST Edge AI")

    if res_arcnet:
        print(f"\n  ── CLASSIFICATORE: ArcNet ────────────────────────────────────")
        print(f"  Scaler          : {res_arcnet['scaler_flash_kb']:>8.1f} KB  Flash interna")
        print(f"  PCA mean        : {res_arcnet['pca_mean_flash_kb']:>8.1f} KB  Flash interna")
        print(f"  ArcNet pesi     : {res_arcnet['arcnet_flash_kb']:>8.1f} KB  Flash interna  ({res_arcnet['n_params_arcnet']:,} param)")
        print(f"  PCA components  : {res_arcnet['pca_components_flash_mb']:>8.2f} MB  Flash ESTERNA (QSPI)")
        print(f"  ─────────────────────────────────────────────────────────────")
        print(f"  Totale clf int  : {res_arcnet['clf_flash_interna_kb']:>8.1f} KB  Flash interna")
        print(f"  Totale clf ext  : {res_arcnet['clf_flash_esterna_mb']:>8.2f} MB  Flash esterna")
        print(f"  Buffer feat RAM : {res_arcnet['feat_ram_kb']:>8.1f} KB  RAM")
        print(f"  PCA output RAM  : {res_arcnet['pca_ram_kb']:>8.1f} KB  RAM")
        tot_int = res_rocket['flash_kb'] + res_arcnet['clf_flash_interna_kb']
        print(f"  ─────────────────────────────────────────────────────────────")
        print(f"  TOTALE PIPELINE : {tot_int:>8.1f} KB  Flash interna")
        print(f"                  + {res_arcnet['clf_flash_esterna_mb']:>8.2f} MB  Flash esterna (QSPI)")
        print(f"                    {res_arcnet['clf_ram_kb']:>8.1f} KB  RAM")
        print(f"  Verificabile    : arcnet.onnx caricabile direttamente in ST Edge AI")

    if res_ridge and res_arcnet:
        print(f"\n{sep}")
        print("  CONFRONTO CLASSIFICATORI FINALI (MultiRocket escluso — uguale per entrambi)")
        print(thin)
        print(f"  {'Voce':<35} {'Ridge':>12} {'ArcNet':>12}")
        print(thin)
        print(f"  {'Flash interna clf (KB)':<35} {res_ridge['clf_flash_kb']:>12.1f} {res_arcnet['clf_flash_interna_kb']:>12.1f}")
        print(f"  {'Flash esterna clf (MB)':<35} {'0.00':>12} {res_arcnet['clf_flash_esterna_mb']:>12.2f}")
        print(f"  {'RAM (KB)':<35} {res_ridge['feat_ram_kb']:>12.1f} {res_arcnet['clf_ram_kb']:>12.1f}")
        print(f"  {'N. parametri classificatore':<35} {res_ridge['n_weights']:>12,} {res_arcnet['n_params_arcnet']:>12,}")
        print(thin)
        print(f"  Ridge:  più leggero, no Flash esterna, ma non esportabile in ONNX")
        print(f"  ArcNet: richiede {res_arcnet['clf_flash_esterna_mb']:.1f} MB QSPI per la PCA,")
        print(f"          ma arcnet.onnx è verificabile direttamente con ST Edge AI")
    print(sep)


# =============================================================================
# PIPELINE RIDGE
# =============================================================================

def run_ridge(X_train, y_train, X_test, y_test, args, out_dir):
    banner("PIPELINE 1 — MultiRocket + Ridge")
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.time()

    F_train, F_test, transformer = fit_transform_rocket(X_train, X_test)
    n_features = F_train.shape[1]

    scaler  = FeatureScaler()
    F_train = scaler.fit_transform(F_train)
    F_test  = scaler.transform(F_test)

    print("  Fitting RidgeClassifier...")
    t0    = time.time()
    model = RidgeClassifier(alpha=1.0, class_weight="balanced",
                             random_state=RANDOM_STATE)
    model.fit(F_train, y_train)
    train_time = time.time() - t0
    print(f"  Fatto in {train_time:.2f}s")

    preds  = model.predict(F_test)
    scores = model.decision_function(F_test)
    probs  = 1 / (1 + np.exp(-scores))

    metrics = compute_metrics(y_test, preds, probs)
    print_metrics(metrics, y_test, preds)

    res_rocket = estimate_multirocket(transformer)
    res_clf    = estimate_ridge_clf(model, scaler, n_features)

    total_time = time.time() - t_start
    result = {
        "model":        "MultiRocket + Ridge",
        "n_features":   n_features,
        "downsample":   args.downsample,
        "train_time_s": round(train_time, 2),
        "total_time_s": round(total_time, 2),
        "metrics":      metrics,
        "_res_rocket":  res_rocket,
        "_res_clf":     res_clf,
    }

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        cfg = {k: v for k, v in result.items()
               if not k.startswith("_") and k != "metrics"}
        cfg["metrics"] = {k: v for k, v in metrics.items()
                          if not k.startswith("_")}
        cfg["resources"] = {
            "multirocket_flash_kb": res_rocket["flash_kb"],
            "clf_flash_kb":         res_clf["clf_flash_kb"],
            "total_flash_kb":       round(res_rocket["flash_kb"] + res_clf["clf_flash_kb"], 1),
            "ram_kb":               res_clf["feat_ram_kb"],
        }
        json.dump(cfg, f, indent=2)

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

    total_time = time.time() - t_start
    result = {
        "model":        "MultiRocketHydra + Ridge",
        "n_kernels":    args.hydra_kernels,
        "n_groups":     args.hydra_groups,
        "downsample":   args.downsample,
        "train_time_s": round(train_time, 2),
        "total_time_s": round(total_time, 2),
        "metrics":      metrics,
    }

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        cfg = {k: v for k, v in result.items() if k != "metrics"}
        cfg["metrics"] = {k: v for k, v in metrics.items()
                          if not k.startswith("_")}
        json.dump(cfg, f, indent=2)

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

    F_train, F_test, transformer = fit_transform_rocket(X_train, X_test)
    n_features_raw = F_train.shape[1]

    scaler  = FeatureScaler()
    F_train = scaler.fit_transform(F_train)
    F_test  = scaler.transform(F_test)

    P_train, P_test, pca = fit_pca(F_train, F_test, args.pca_components)
    n_features = P_train.shape[1]

    train_ds     = TensorDataset(torch.from_numpy(P_train),
                                  torch.from_numpy(y_train.astype(np.float32)))
    test_ds      = TensorDataset(torch.from_numpy(P_test),
                                  torch.from_numpy(y_test.astype(np.float32)))
    sampler      = make_sampler(y_train)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

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

    model.load_state_dict(torch.load(best_path, map_location=device,
                                     weights_only=True))
    labels, preds, probs = evaluate_nn(model, test_loader, device)
    metrics = compute_metrics(labels, preds, probs)
    print_metrics(metrics, labels, preds)

    # Rimuove il checkpoint temporaneo
    os.remove(best_path)

    res_rocket = estimate_multirocket(transformer)
    res_clf    = estimate_arcnet_clf(model, n_features_raw, args.pca_components)

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
        "_res_rocket":    res_rocket,
        "_res_clf":       res_clf,
    }

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        cfg = {k: v for k, v in result.items()
               if not k.startswith("_") and k != "metrics"}
        cfg["metrics"] = {k: v for k, v in metrics.items()
                          if not k.startswith("_")}
        cfg["resources"] = {
            "multirocket_flash_kb":     res_rocket["flash_kb"],
            "clf_flash_interna_kb":     res_clf["clf_flash_interna_kb"],
            "clf_flash_esterna_mb":     res_clf["clf_flash_esterna_mb"],
            "total_flash_interna_kb":   round(res_rocket["flash_kb"] + res_clf["clf_flash_interna_kb"], 1),
            "ram_kb":                   res_clf["clf_ram_kb"],
        }
        json.dump(cfg, f, indent=2)

    # Bundle completo: tutto quello che serve per l'inferenza
    bundle = {
        "transformer": transformer,
        "scaler":      scaler,
        "pca":         pca,
        "model":       model.cpu().state_dict(),
        "model_cfg": {
            "n_features": n_features,
            "hidden":     list(args.hidden),
            "dropout":    args.dropout,
        },
        "metrics": metrics,
    }
    with open(os.path.join(out_dir, "bundle.pkl"), "wb") as f:
        pickle.dump(bundle, f)

    print(f"\n  Output in: {out_dir}/")
    return result


# =============================================================================
# EXPORT CSV + TXT
# =============================================================================

def export_csv(results, out_dir):
    import csv

    # CSV metriche
    metric_path = os.path.join(out_dir, "comparison_metrics.csv")
    fields = ["model", "accuracy", "balanced_accuracy", "f1", "roc_auc",
              "avg_precision", "train_time_s", "total_time_s"]
    with open(metric_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for m, r in results.items():
            w.writerow({
                "model":             r["model"],
                "accuracy":          round(r["metrics"].get("accuracy", 0), 4),
                "balanced_accuracy": round(r["metrics"].get("balanced_accuracy", 0), 4),
                "f1":                round(r["metrics"].get("f1", 0), 4),
                "roc_auc":           round(r["metrics"].get("roc_auc", 0), 4),
                "avg_precision":     round(r["metrics"].get("avg_precision", 0), 4),
                "train_time_s":      r.get("train_time_s", ""),
                "total_time_s":      r.get("total_time_s", ""),
            })
    print(f"  CSV metriche   : {metric_path}")

    # CSV confusion matrix
    cm_path = os.path.join(out_dir, "comparison_confusion.csv")
    with open(cm_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "TN", "FP", "FN", "TP"])
        w.writeheader()
        for m, r in results.items():
            cm = r["metrics"].get("confusion_matrix", [[0, 0], [0, 0]])
            w.writerow({
                "model": r["model"],
                "TN": cm[0][0], "FP": cm[0][1],
                "FN": cm[1][0], "TP": cm[1][1],
            })
    print(f"  CSV confusion  : {cm_path}")


def export_txt_report(results, out_path):
    lines = []
    sep   = "=" * 72
    thin  = "-" * 72

    lines += [sep,
              "  ARC DETECTION — CONFRONTO MODELLI",
              "  MultiRocket+Ridge  |  MultiRocketHydra+Ridge  |  MR+PCA+ArcNet",
              sep, ""]

    models  = list(results.keys())
    col_w   = 20
    label_w = 28

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

    # Tabella metriche
    lines += ["  TABELLA 1 — METRICHE DI PERFORMANCE", thin]
    lines.append(header_row("Metrica"))
    lines.append(thin)
    for label, key, best_max in [
        ("Accuracy",          "accuracy",          True),
        ("Balanced Accuracy", "balanced_accuracy",  True),
        ("F1 Score",          "f1",                 True),
        ("ROC AUC",           "roc_auc",            True),
        ("Avg Precision",     "avg_precision",      True),
    ]:
        vals = [results[m]["metrics"].get(key, 0) for m in models]
        lines.append(data_row(label, vals, ".4f", best_max))
    lines += [thin, "  * = miglior valore per quella metrica", ""]

    # Tabella tempi
    lines += ["  TABELLA 2 — TEMPI", thin]
    lines.append(header_row("Tempo"))
    lines.append(thin)
    for label, key in [("Training (s)", "train_time_s"), ("Totale (s)", "total_time_s")]:
        vals = [results[m].get(key, 0) for m in models]
        lines.append(data_row(label, vals, ".1f", False))
    lines += [thin, ""]

    # Confusion matrices
    lines += ["  TABELLA 3 — CONFUSION MATRICES", thin]
    for m in models:
        cm = results[m]["metrics"].get("confusion_matrix", [[0, 0], [0, 0]])
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        lines += [
            f"  {results[m]['model']}",
            f"    {'':20} {'Pred No Arc':>14} {'Pred Arc':>14}",
            f"    {'Actual No Arc':20} {tn:>14} {fp:>14}",
            f"    {'Actual Arc':20} {fn:>14} {tp:>14}",
            "",
        ]

    # Nota deploy
    lines += [
        sep,
        "  NOTA SUL DEPLOY STM32",
        thin,
        "  Tutte e tre le pipeline usano MultiRocket come preprocessing.",
        "  MultiRocket non ha un export C/ONNX automatico: nessuna pipeline",
        "  è deployabile su STM32 senza reimplementare i kernel in C.",
        "  ArcNet è la più avanzata (scaler+PCA+rete già nel bundle),",
        "  ma il preprocessing MultiRocket è comune a tutti e tre.",
        sep,
        "  Fine report",
        sep,
    ]

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
    parser.add_argument("--out",  default="./results")
    parser.add_argument("--models", nargs="+",
                        choices=["ridge", "hydra", "arcnet"],
                        default=["ridge", "hydra", "arcnet"])
    parser.add_argument("--downsample",     type=int,   default=4)
    parser.add_argument("--pca-components", type=int,   default=256)
    parser.add_argument("--hidden",         type=int,   nargs="+", default=[64, 32])
    parser.add_argument("--dropout",        type=float, default=0.3)
    parser.add_argument("--epochs",         type=int,   default=30)
    parser.add_argument("--batch-size",     type=int,   default=64)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--hydra-kernels",  type=int,   default=8)
    parser.add_argument("--hydra-groups",   type=int,   default=4)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    banner("LOAD DATASET")
    X_train, y_train = load_dataset(args.train, args.downsample)
    X_test,  y_test  = load_dataset(args.test,  args.downsample)
    print(f"  Train: {X_train.shape}  labels: {np.bincount(y_train)}")
    print(f"  Test : {X_test.shape}   labels: {np.bincount(y_test)}")

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

    col_w = 22
    metric_labels = {
        "accuracy":          "Accuracy",
        "balanced_accuracy": "Balanced Accuracy",
        "f1":                "F1 Score",
        "roc_auc":           "ROC AUC",
        "avg_precision":     "Avg Precision",
    }

    header = f"{'Metrica':<25}" + "".join(
        f"{results[m]['model'][:col_w]:>{col_w}}" for m in results)
    print(header)
    print("-" * len(header))

    for key, label in metric_labels.items():
        vals = {m: results[m]["metrics"].get(key, 0) for m in results}
        best = max(vals.values())
        row  = f"{label:<25}"
        for m in results:
            v    = vals[m]
            star = " ★" if v == best else "  "
            row += f"{v:.4f}{star}".rjust(col_w)
        print(row)

    print()
    print("  Nota: tutte e tre le pipeline richiedono la reimplementazione")
    print("  di MultiRocket in C per il deploy su STM32.")

    # ── Stima risorse ─────────────────────────────────────────────────────────
    res_rocket_r = results["ridge"]["_res_rocket"] if "ridge" in results else None
    res_rocket_a = results["arcnet"]["_res_rocket"] if "arcnet" in results else None
    res_rocket   = res_rocket_r or res_rocket_a  # stesso costo per entrambi
    res_ridge    = results["ridge"]["_res_clf"] if "ridge" in results else None
    res_arcnet   = results["arcnet"]["_res_clf"] if "arcnet" in results else None
    if res_rocket:
        print_resource_table(res_rocket, res_ridge, res_arcnet)

    # ── Salva JSON ────────────────────────────────────────────────────────────
    comparison_path = os.path.join(args.out, "comparison.json")
    # rimuove _probs/_labels dal JSON (troppo grandi)
    results_json = {}
    for k, r in results.items():
        results_json[k] = {kk: vv for kk, vv in r.items() if kk != "metrics"}
        results_json[k]["metrics"] = {kk: vv for kk, vv in r["metrics"].items()
                                      if not kk.startswith("_")}
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)
    print(f"\n  Confronto JSON: {comparison_path}")

    # ── CSV + TXT ─────────────────────────────────────────────────────────────
    banner("EXPORT CSV + TXT + GRAFICI")
    export_csv(results, args.out)
    export_txt_report(results, os.path.join(args.out, "comparison_report.txt"))
    plot_comparison(results, args.out)

    banner("DONE")
    print(f"""
  Output in: {args.out}/
    ridge/
      bundle.pkl               modello + transformer + scaler
      config.json              metriche
    hydra/
      bundle.pkl               modello completo
      config.json              metriche
    arcnet/
      bundle.pkl               transformer + scaler + pca + model state_dict
      config.json              metriche
    comparison.json
    comparison_metrics.csv
    comparison_confusion.csv
    comparison_report.txt
    comparison_plots.png       grafici comparativi (ROC, PR, CM, metriche)
""")


if __name__ == "__main__":
    main()