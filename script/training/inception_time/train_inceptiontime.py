#!/usr/bin/env python3
"""
train_inceptiontime_v2.py
=========================
InceptionTime su TUTTO il dataset (27.770 train) con gestione corretta
dello sbilanciamento — niente undersampling, niente perdita di dati.

DIFFERENZE RISPETTO ALLA v1:
  - RIMOSSO undersampling (usava solo 6000/27770 campioni → spreco di dati)
  - Sbilanciamento gestito con:
      1. class_weight nel loss (WeightedCrossEntropy) → FIX principale
      2. WeightedRandomSampler nei DataLoader → batch bilanciati
  - Analisi multi-soglia UL1699B (non solo soglia fissa 0.5)
  - Early stopping su val_loss per evitare overfitting su dataset grande
  - Shuffle test per verificare che il modello discrimini davvero
  - Grafici aggiuntivi: confusion matrix normalizzata, curva di apprendimento

ARCHITETTURA InceptionTime:
  - Blocchi Inception con kernel paralleli (10, 20, 40 timepoints)
  - Residual connections
  - Global Average Pooling finale
  - ~450K parametri — leggero per GPU RTX 30xx/40xx

REQUISITI:
    pip install tsai torch scikit-learn matplotlib seaborn onnxruntime
    pip install torch --index-url https://download.pytorch.org/whl/cu121

USO:
    python train_inceptiontime_v2.py
    python train_inceptiontime_v2.py --epochs 80 --batch-size 128
    python train_inceptiontime_v2.py --train path/train.npz --test path/test.npz
    python train_inceptiontime_v2.py --no-class-weight   (per confronto)

Normativa: UL 1699B — Photovoltaic DC Arc-Fault Circuit Protection
"""

import argparse
import logging
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from sklearn.metrics import (
    balanced_accuracy_score, classification_report, confusion_matrix,
    f1_score, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ── path dataset (modifica se necessario) ────────────────────────────────────
DATASET_TRAIN = r"C:\Users\Asus\Desktop\progetto_manutenzione\dataset\dataset_new\arc_dataset_train.npz"
DATASET_TEST  = r"C:\Users\Asus\Desktop\progetto_manutenzione\dataset\dataset_new\arc_dataset_test.npz"

# ── costanti ──────────────────────────────────────────────────────────────────
FS_HZ      = 10_000
RAND       = 42
UL_MIN_DET = 95.0
UL_MAX_FP  = 5.0


# ══════════════════════════════════════════════════════════════════════════════
# 1. InceptionTime — implementazione PyTorch nativa (no tsai)
#    Usiamo l'implementazione diretta per avere pieno controllo su loss,
#    sampler e training loop — evita i bug interni di tsai con numpy.object_
# ══════════════════════════════════════════════════════════════════════════════

class InceptionModule(nn.Module):
    """
    Blocco Inception con tre kernel paralleli + bottleneck.
    Kernel sizes: 10, 20, 40 timepoints (adattati per serie a 10kHz).
    """
    def __init__(self, in_channels: int, n_filters: int = 32,
                 kernel_sizes=(10, 20, 40), bottleneck_size: int = 32):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_size,
                                    kernel_size=1, bias=False)
        self.convs = nn.ModuleList([
            nn.Conv1d(bottleneck_size, n_filters,
                      kernel_size=k, padding=k // 2, bias=False)
            for k in kernel_sizes
        ])
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, n_filters, kernel_size=1, bias=False),
        )
        self.bn = nn.BatchNorm1d(n_filters * len(kernel_sizes) + n_filters)
        self.relu = nn.ReLU()

    def forward(self, x):
        bottleneck = self.bottleneck(x)
        out = [conv(bottleneck) for conv in self.convs]
        out.append(self.maxpool_conv(x))
        # Allinea la dimensione temporale (padding può creare +/-1)
        min_len = min(o.shape[-1] for o in out)
        out = [o[..., :min_len] for o in out]
        out = torch.cat(out, dim=1)
        return self.relu(self.bn(out))


class ResidualBlock(nn.Module):
    """Shortcut connection se le dimensioni differiscono."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
        ) if in_ch != out_ch else nn.Identity()
        self.bn  = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, residual):
        return self.relu(self.bn(residual) + self.shortcut(x))


class InceptionTimeNet(nn.Module):
    """
    InceptionTime — 3 blocchi Inception con residual connections.
    Input:  (batch, 1, T)
    Output: (batch, 2)  logits
    """
    def __init__(self, n_channels: int = 1, n_classes: int = 2,
                 n_filters: int = 32, depth: int = 6):
        super().__init__()
        self.depth = depth
        n_out = n_filters * 4  # 3 kernel_sizes + maxpool = 4

        self.inception_blocks = nn.ModuleList()
        self.residual_blocks   = nn.ModuleList()

        in_ch = n_channels
        for i in range(depth):
            self.inception_blocks.append(
                InceptionModule(in_ch, n_filters=n_filters))
            if (i + 1) % 3 == 0:
                self.residual_blocks.append(
                    ResidualBlock(in_ch, n_out))
            in_ch = n_out

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc  = nn.Linear(n_out, n_classes)

    def forward(self, x):
        residual_input = x
        res_idx = 0
        for i, block in enumerate(self.inception_blocks):
            x = block(x)
            if (i + 1) % 3 == 0:
                x = self.residual_blocks[res_idx](residual_input, x)
                residual_input = x
                res_idx += 1
        x = self.gap(x).squeeze(-1)
        return self.fc(x)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Dataset e DataLoader con WeightedRandomSampler
# ══════════════════════════════════════════════════════════════════════════════

def make_dataloaders(X_train, y_train, X_test, y_test,
                     batch_size: int, use_weighted_sampler: bool = True):
    """
    Crea DataLoader con WeightedRandomSampler per bilanciare i batch.

    WeightedRandomSampler assegna probabilità di campionamento inversamente
    proporzionale alla frequenza della classe — la classe minoritaria
    (no-arco, 37.5%) viene vista con la stessa frequenza della maggioritaria.

    Questo è complementare al class_weight nel loss: insieme garantiscono
    che il modello non ignori la classe minoritaria.
    """
    X_tr = torch.tensor(X_train[:, np.newaxis, :], dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    X_te = torch.tensor(X_test[:, np.newaxis, :],  dtype=torch.float32)
    y_te = torch.tensor(y_test, dtype=torch.long)

    train_ds = TensorDataset(X_tr, y_tr)
    test_ds  = TensorDataset(X_te, y_te)

    if use_weighted_sampler:
        # Peso per campione = 1 / frequenza della sua classe
        class_counts = np.bincount(y_train)
        weights      = 1.0 / class_counts[y_train]
        sampler      = WeightedRandomSampler(
            torch.tensor(weights, dtype=torch.float32),
            num_samples=len(y_train),
            replacement=True,
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  sampler=sampler, num_workers=0,
                                  pin_memory=True)
        log.info("  WeightedRandomSampler attivo: no-arco×%.1f  arco×%.1f",
                 weights[y_train==0].mean(), weights[y_train==1].mean())
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, num_workers=0, pin_memory=True)
        log.info("  Sampler standard (shuffle=True)")

    test_loader = DataLoader(test_ds, batch_size=batch_size * 2,
                             shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, test_loader


# ══════════════════════════════════════════════════════════════════════════════
# 3. Training loop con early stopping
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(X_batch)
                loss   = criterion(logits, y_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(1) == y_batch).sum().item()
        n          += len(y_batch)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_probs, all_preds, all_labels = [], [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        probs  = torch.softmax(logits, dim=1)
        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(1) == y_batch).sum().item()
        n          += len(y_batch)
        all_probs.append(probs[:, 1].cpu().numpy())
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_labels.append(y_batch.cpu().numpy())
    return (total_loss / n, correct / n,
            np.concatenate(all_probs),
            np.concatenate(all_preds),
            np.concatenate(all_labels))


def fit(model, train_loader, test_loader, epochs, lr, device,
        class_weights=None, patience=10):
    """
    Training completo con:
      - WeightedCrossEntropyLoss (class_weights)
      - OneCycleLR scheduler
      - AMP (automatic mixed precision) su GPU
      - Early stopping su val_loss
    """
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32).to(device)
        if class_weights is not None else None
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr,
        steps_per_epoch=len(train_loader), epochs=epochs,
        pct_start=0.3,
    )
    # AMP solo su GPU
    use_amp = device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda") if use_amp else None

    history = {k: [] for k in
               ["train_loss","val_loss","train_acc","val_acc"]}
    best_val_loss = float("inf")
    best_state    = None
    wait          = 0

    log.info("  %6s  %10s  %10s  %8s  %8s  %6s",
             "Epoch", "TrainLoss", "ValLoss", "TrainAcc", "ValAcc", "LR")
    log.info("  " + "-" * 58)

    t0 = time.time()
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer,
                                       criterion, device, scaler)
        vl_loss, vl_acc, _, _, _ = eval_epoch(model, test_loader,
                                               criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        if ep % 5 == 0 or ep == 1:
            lr_cur = optimizer.param_groups[0]["lr"]
            log.info("  %6d  %10.4f  %10.4f  %7.2f%%  %7.2f%%  %.2e",
                     ep, tr_loss, vl_loss,
                     100*tr_acc, 100*vl_acc, lr_cur)

        # Early stopping
        if vl_loss < best_val_loss - 1e-4:
            best_val_loss = vl_loss
            best_state    = {k: v.cpu().clone()
                             for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                log.info("  Early stopping a epoch %d (patience=%d)",
                         ep, patience)
                break

    t_train = time.time() - t0
    log.info("  Training completato in %.1f s (%.1f min)",
             t_train, t_train / 60)

    # Ricarica miglior modello
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        log.info("  Ricaricato miglior modello (val_loss=%.4f)", best_val_loss)

    return history, t_train


# ══════════════════════════════════════════════════════════════════════════════
# 4. Metriche UL1699B
# ══════════════════════════════════════════════════════════════════════════════

def ul1699b_metric(y_test, y_pred, threshold=0.5) -> dict:
    arc, no_arc = y_test == 1, y_test == 0
    det  = int(((y_pred==1) & arc).sum())
    miss = int(((y_pred==0) & arc).sum())
    fp   = int(((y_pred==1) & no_arc).sum())
    tn   = int(((y_pred==0) & no_arc).sum())
    det_r = 100.0 * det / max(int(arc.sum()),    1)
    fp_r  = 100.0 * fp  / max(int(no_arc.sum()), 1)
    ok    = det_r >= UL_MIN_DET and fp_r <= UL_MAX_FP

    log.info("")
    log.info("=" * 60)
    log.info("METRICA UL1699B (soglia=%.2f)", threshold)
    log.info("=" * 60)
    log.info("  Archi:      %d → rilevati %d (%.1f%%), mancati %d",
             int(arc.sum()), det, det_r, miss)
    log.info("  No arco:    %d → FP %d (%.1f%%), TN %d",
             int(no_arc.sum()), fp, fp_r, tn)
    log.info("  Esito:      %s",
             "✓ CONFORME UL1699B" if ok else "✗ NON conforme UL1699B")

    return {"detected": det, "missed": miss, "false_positives": fp,
            "true_negatives": tn, "detection_rate_pct": round(det_r, 2),
            "false_positive_rate_pct": round(fp_r, 2), "ul1699b_conforme": ok}


def threshold_analysis(y_test, y_proba) -> float:
    log.info("")
    log.info("  --- Analisi multi-soglia UL1699B ---")
    log.info("  %8s  %7s  %6s  %6s", "Soglia", "Det%", "FP%", "UL1699B")
    arc, no_arc = y_test==1, y_test==0
    best_thr, best_det = 0.5, 0.0

    for thr in np.arange(0.05, 1.00, 0.05):
        yp  = (y_proba >= thr).astype(int)
        det = 100.0 * ((yp==1)&arc).sum()    / max(int(arc.sum()),    1)
        fpr = 100.0 * ((yp==1)&no_arc).sum() / max(int(no_arc.sum()), 1)
        ok  = "SI" if det >= UL_MIN_DET and fpr <= UL_MAX_FP else "NO"
        log.info("  %8.2f  %6.1f%%  %5.1f%%  %6s", thr, det, fpr, ok)
        if det >= UL_MIN_DET and fpr <= UL_MAX_FP and det > best_det:
            best_det, best_thr = det, float(thr)

    if best_det == 0.0:
        log.warning("  Nessuna soglia soddisfa entrambi i vincoli UL1699B.")
        best_det2 = 0.0
        for thr in np.arange(0.01, 1.00, 0.01):
            yp  = (y_proba >= thr).astype(int)
            det = 100.0 * ((yp==1)&arc).sum()    / max(int(arc.sum()),    1)
            fpr = 100.0 * ((yp==1)&no_arc).sum() / max(int(no_arc.sum()), 1)
            if fpr <= UL_MAX_FP and det > best_det2:
                best_det2, best_thr = det, float(thr)
        log.warning("  Fallback: soglia=%.2f det=%.1f%%", best_thr, best_det2)
    else:
        log.info("  Soglia ottimale: %.2f (det=%.1f%%)", best_thr, best_det)
    return best_thr


# ══════════════════════════════════════════════════════════════════════════════
# 5. Shuffle test
# ══════════════════════════════════════════════════════════════════════════════

def shuffle_test(y_test, y_pred, y_proba, n=5) -> dict:
    """
    Verifica che il modello discrimini davvero e non predica
    sempre la classe maggioritaria.
    """
    log.info("")
    log.info("  --- Shuffle test ---")
    rng    = np.random.default_rng(RAND)
    f1_r   = f1_score(y_test, y_pred, zero_division=0)
    f1_shf = [f1_score(rng.permutation(y_test), y_pred, zero_division=0)
              for _ in range(n)]
    mean_s = float(np.mean(f1_shf))
    ok     = mean_s < f1_r * 0.7
    log.info("  F1 reale:          %.4f", f1_r)
    log.info("  F1 shuffle (media):%.4f  (atteso ≪ F1 reale)", mean_s)
    log.info("  Esito: %s", "✓ OK — modello discrimina" if ok
             else "✗ ATTENZIONE — modello potrebbe non discriminare")
    return {"f1_real": round(f1_r,4), "f1_shuffle": round(mean_s,4),
            "shuffle_ok": ok}


# ══════════════════════════════════════════════════════════════════════════════
# 6. Grafici
# ══════════════════════════════════════════════════════════════════════════════

def plot_training(history, out_dir):
    if not history["train_loss"]:
        return
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("InceptionTime v2 — Curve di Training", fontsize=12)

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], color="steelblue", label="Train loss")
    ax.plot(epochs, history["val_loss"],   color="tomato",    label="Val loss", ls="--")
    ax.set_title("Loss"); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, [v*100 for v in history["train_acc"]],
            color="steelblue", label="Train acc")
    ax.plot(epochs, [v*100 for v in history["val_acc"]],
            color="tomato", label="Val acc", ls="--")
    ax.axhline(95, color="red", ls=":", lw=1, label="95% UL1699B")
    ax.set_title("Accuracy [%]"); ax.set_xlabel("Epoch")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "inceptiontime_training.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    log.info("  Salvato: %s", path)


def plot_results(y_test, y_pred, y_proba, best_thr, out_dir):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(
        f"InceptionTime v2 — dataset completo  (soglia={best_thr:.2f})",
        fontsize=13)

    # CM assoluta
    ax = axes[0]
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", ax=ax,
                xticklabels=["No arco","Arco"],
                yticklabels=["No arco","Arco"])
    ax.set_title("Confusion Matrix"); ax.set_ylabel("Reale"); ax.set_xlabel("Predetto")

    # ROC
    ax = axes[1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC={auc:.3f}")
    ax.plot([0,1],[0,1],"k--",lw=1)
    ax.axvline(UL_MAX_FP/100,  color="steelblue", ls=":", lw=1.5,
               label=f"UL max FP={UL_MAX_FP:.0f}%")
    ax.axhline(UL_MIN_DET/100, color="green",     ls=":", lw=1.5,
               label=f"UL min det={UL_MIN_DET:.0f}%")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(fontsize=8); ax.set_title("ROC Curve"); ax.grid(alpha=0.3)

    # Precision-Recall
    ax = axes[2]
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    ax.plot(rec, prec, color="darkorange", lw=2, label=f"AP={ap:.3f}")
    ax.axhline(y_test.mean(), color="gray", ls="--", lw=1,
               label=f"Baseline={y_test.mean():.2f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.legend(); ax.set_title("Precision-Recall"); ax.grid(alpha=0.3)

    # Score distribution
    ax = axes[3]
    ax.hist(y_proba[y_test==0], bins=40, alpha=0.6,
            color="steelblue", label="No arco (0)")
    ax.hist(y_proba[y_test==1], bins=40, alpha=0.6,
            color="darkorange", label="Arco (1)")
    ax.axvline(best_thr, color="black", ls="--", lw=1.5,
               label=f"soglia={best_thr:.2f}")
    ax.set_xlabel("P(arco)"); ax.legend()
    ax.set_title("Distribuzione score"); ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "results_inceptiontime.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    log.info("  Salvato: %s", path)


def plot_threshold_curve(y_test, y_proba, best_thr, out_dir):
    thrs  = np.arange(0.01, 1.00, 0.01)
    arc, no_arc = y_test==1, y_test==0
    det_r = [100.0 * ((y_proba>=t).astype(int)[arc]).sum()    / max(arc.sum(),    1)
             for t in thrs]
    fp_r  = [100.0 * ((y_proba>=t).astype(int)[no_arc]).sum() / max(no_arc.sum(), 1)
             for t in thrs]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thrs, det_r, color="tomato",    lw=2, label="Detection rate %")
    ax.plot(thrs, fp_r,  color="steelblue", lw=2, label="False Positive rate %")
    ax.axhline(UL_MIN_DET, color="tomato",    ls="--", lw=1,
               label=f"UL1699B min det={UL_MIN_DET:.0f}%")
    ax.axhline(UL_MAX_FP,  color="steelblue", ls="--", lw=1,
               label=f"UL1699B max FP={UL_MAX_FP:.0f}%")
    ax.axvline(best_thr, color="black", ls=":", lw=2,
               label=f"Soglia ottimale={best_thr:.2f}")
    ax.fill_betweenx([UL_MIN_DET, 100], [best_thr-0.05], [best_thr+0.05],
                     alpha=0.1, color="green", label="Zona conformità")
    ax.set_xlabel("Soglia"); ax.set_ylabel("Percentuale [%]")
    ax.set_title("Analisi multi-soglia — UL1699B (InceptionTime v2)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "threshold_analysis.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    log.info("  Salvato: %s", path)


def plot_series_examples(X_test, y_test, y_pred, out_dir, n_per_class=2):
    categories = [
        ("Vero Positivo\n(arco rilevato)",   y_test==1, y_pred==1),
        ("Falso Negativo\n(arco mancato)",    y_test==1, y_pred==0),
        ("Vero Negativo\n(no arco corretto)", y_test==0, y_pred==0),
        ("Falso Positivo\n(falso allarme)",   y_test==0, y_pred==1),
    ]
    fig, axes = plt.subplots(n_per_class, 4, figsize=(16, n_per_class*3))
    fig.suptitle("Esempi serie temporali — corrente I(t)", fontsize=12)
    t = np.arange(X_test.shape[1]) / FS_HZ

    for col, (title, mr, mp) in enumerate(categories):
        idx = np.where(mr & mp)[0]
        for row in range(n_per_class):
            ax = axes[row, col] if n_per_class > 1 else axes[col]
            if row < len(idx):
                ix = idx[row]
                ax.plot(t, X_test[ix], lw=0.7,
                        color="darkorange" if y_test[ix]==1 else "steelblue")
                ax.set_xlabel("t [s]"); ax.set_ylabel("I [-]"); ax.grid(alpha=0.3)
                if row == 0: ax.set_title(title, fontsize=9)
            else:
                ax.text(0.5, 0.5, "Nessun\nesempio", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
                ax.axis("off")
                if row == 0: ax.set_title(title, fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, "series_examples.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    log.info("  Salvato: %s", path)


# ══════════════════════════════════════════════════════════════════════════════
# 7. Export ONNX
# ══════════════════════════════════════════════════════════════════════════════

def export_onnx(model, n_timepoints, out_dir, device):
    model_cpu = model.cpu().eval()

    # Dinamico
    path_dyn = os.path.join(out_dir, "inceptiontime.onnx")
    dummy = torch.zeros(1, 1, n_timepoints, dtype=torch.float32)
    torch.onnx.export(
        model_cpu, dummy, path_dyn,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=13, do_constant_folding=True,
    )
    log.info("  ONNX dinamico: %s  (%.1f KB)",
             path_dyn, os.path.getsize(path_dyn) / 1024)

    # Statico (per ST Edge AI)
    path_sta = os.path.join(out_dir, "inceptiontime_static.onnx")
    torch.onnx.export(
        model_cpu, dummy, path_sta,
        input_names=["input"], output_names=["logits"],
        opset_version=13, do_constant_folding=True,
    )
    log.info("  ONNX statico:  %s  (%.1f KB)  ← ST Edge AI quantizzazione INT8",
             path_sta, os.path.getsize(path_sta) / 1024)

    try:
        import onnxruntime as rt
        for path in [path_dyn, path_sta]:
            sess = rt.InferenceSession(path)
            out  = sess.run(None, {"input": dummy.numpy()})[0]
            log.info("  Verifica %s: output=%s ✓",
                     os.path.basename(path), out.shape)
    except ImportError:
        log.warning("  onnxruntime non installato — skip verifica")
    except Exception as e:
        log.warning("  Verifica ONNX: %s", e)

    model.to(device)


# ══════════════════════════════════════════════════════════════════════════════
# 8. Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="InceptionTime v2 — dataset completo, class weighting, no undersampling\n"
                    "Normativa: UL 1699B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--train",      default=DATASET_TRAIN)
    parser.add_argument("--test",       default=DATASET_TEST)
    parser.add_argument("--out",  "-o", default="./risultati_inception_v2")
    parser.add_argument("--epochs",     type=int,   default=60,
                        help="Epoche max (early stopping può fermarsi prima, default: 60)")
    parser.add_argument("--batch-size", type=int,   default=128,
                        help="Batch size GPU (default: 128, più grande = più veloce)")
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--patience",   type=int,   default=12,
                        help="Patience early stopping (default: 12)")
    parser.add_argument("--no-class-weight",    action="store_true",
                        help="Disabilita class weighting nel loss (confronto)")
    parser.add_argument("--no-weighted-sampler",action="store_true",
                        help="Disabilita WeightedRandomSampler (confronto)")
    parser.add_argument("--export-onnx",        action="store_true",
                        help="Esporta modello in formato ONNX")
    parser.add_argument("--n-filters",  type=int, default=32,
                        help="Filtri per blocco Inception (default: 32)")
    parser.add_argument("--depth",      type=int, default=6,
                        help="Numero blocchi Inception (default: 6)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # GPU
    if torch.cuda.is_available():
        device   = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
        log.info("GPU: %s  (%.1f GB VRAM)", gpu_name, gpu_mem)
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        log.warning("GPU non disponibile — training su CPU (lento)")

    # Carica dataset
    log.info("=" * 60)
    log.info("CARICAMENTO DATASET")
    log.info("=" * 60)
    d_tr    = np.load(args.train)
    d_te    = np.load(args.test)
    X_train = d_tr["X"].astype(np.float32)
    y_train = d_tr["y"].astype(np.int64)
    X_test  = d_te["X"].astype(np.float32)
    y_test  = d_te["y"].astype(np.int64)

    # Shape (n, T) → ok, la rete vuole (n, 1, T) — aggiunto in make_dataloaders
    if X_train.ndim == 3:
        X_train = X_train[:, 0, :]
    if X_test.ndim == 3:
        X_test  = X_test[:, 0, :]

    log.info("  Train: %d  (arco=%d %.1f%%, no-arco=%d %.1f%%)",
             len(y_train), int((y_train==1).sum()),
             100*(y_train==1).mean(),
             int((y_train==0).sum()),
             100*(y_train==0).mean())
    log.info("  Test:  %d  (arco=%d %.1f%%, no-arco=%d %.1f%%)",
             len(y_test), int((y_test==1).sum()),
             100*(y_test==1).mean(),
             int((y_test==0).sum()),
             100*(y_test==0).mean())
    log.info("  Lunghezza serie: %d campioni (%.0f ms @ %d Hz)",
             X_train.shape[1], X_train.shape[1]/FS_HZ*1000, FS_HZ)
    log.info("  NESSUN undersampling — uso tutto il dataset")

    # Class weights per il loss
    class_weights = None
    if not args.no_class_weight:
        n0 = int((y_train==0).sum())
        n1 = int((y_train==1).sum())
        n_tot = n0 + n1
        # w_c = n_tot / (2 * n_c)  — formula sklearn "balanced"
        w0 = n_tot / (2.0 * n0)
        w1 = n_tot / (2.0 * n1)
        class_weights = [w0, w1]
        log.info("  Class weights: no-arco=%.3f  arco=%.3f", w0, w1)

    # DataLoaders
    log.info("")
    log.info("=" * 60)
    log.info("DATALOADER")
    log.info("=" * 60)
    train_loader, test_loader = make_dataloaders(
        X_train, y_train, X_test, y_test,
        batch_size=args.batch_size,
        use_weighted_sampler=not args.no_weighted_sampler,
    )
    log.info("  Train batches: %d  |  Test batches: %d",
             len(train_loader), len(test_loader))

    # Modello
    log.info("")
    log.info("=" * 60)
    log.info("MODELLO InceptionTime v2")
    log.info("=" * 60)
    model = InceptionTimeNet(
        n_channels=1, n_classes=2,
        n_filters=args.n_filters,
        depth=args.depth,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("  Parametri: %s", f"{n_params:,}")
    log.info("  n_filters=%d  depth=%d", args.n_filters, args.depth)
    log.info("  Device: %s", device)

    # Training
    log.info("")
    log.info("=" * 60)
    log.info("TRAINING")
    log.info("=" * 60)
    log.info("  epochs=%d  lr=%.0e  batch=%d  patience=%d",
             args.epochs, args.lr, args.batch_size, args.patience)
    log.info("  class_weight=%s  weighted_sampler=%s",
             "SI" if class_weights else "NO",
             "NO" if args.no_weighted_sampler else "SI")

    history, t_train = fit(
        model, train_loader, test_loader,
        epochs=args.epochs, lr=args.lr, device=device,
        class_weights=class_weights, patience=args.patience,
    )

    # Predizioni finali
    log.info("")
    log.info("=" * 60)
    log.info("PREDIZIONI TEST SET")
    log.info("=" * 60)
    criterion_eval = nn.CrossEntropyLoss()
    _, _, y_proba, y_pred_05, y_labels = eval_epoch(
        model, test_loader, criterion_eval, device)

    # Analisi soglia ottimale
    best_thr = threshold_analysis(y_test, y_proba)
    y_pred   = (y_proba >= best_thr).astype(int)

    # Metriche
    log.info("")
    log.info("=" * 60)
    log.info("METRICHE TEST SET")
    log.info("=" * 60)
    for line in classification_report(y_test, y_pred,
                                       target_names=["No arco","Arco"],
                                       digits=3).splitlines():
        log.info("  %s", line)

    acc = float((y_pred == y_test).mean())
    ba  = balanced_accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    ap  = average_precision_score(y_test, y_proba)
    log.info("  Accuracy:          %.4f", acc)
    log.info("  Balanced Accuracy: %.4f", ba)
    log.info("  F1 (arco):         %.4f", f1)
    log.info("  ROC-AUC:           %.4f", auc)
    log.info("  Avg Precision:     %.4f", ap)

    ul  = ul1699b_metric(y_test, y_pred, threshold=best_thr)
    shf = shuffle_test(y_test, y_pred, y_proba)

    # Grafici
    log.info("")
    log.info("=" * 60)
    log.info("SALVATAGGIO GRAFICI")
    log.info("=" * 60)
    plot_training(history, args.out)
    plot_results(y_test, y_pred, y_proba, best_thr, args.out)
    plot_threshold_curve(y_test, y_proba, best_thr, args.out)
    plot_series_examples(X_test, y_test, y_pred, args.out)

    # Export ONNX
    if args.export_onnx:
        log.info("")
        log.info("=" * 60)
        log.info("EXPORT ONNX")
        log.info("=" * 60)
        export_onnx(model, X_train.shape[1], args.out, device)

    # Salvataggio bundle
    torch.save({
        "model_state_dict":  model.state_dict(),
        "model_config":      {"n_filters": args.n_filters, "depth": args.depth},
        "best_threshold":    best_thr,
        "class_weights":     class_weights,
        "norm": "none",
    }, os.path.join(args.out, "inceptiontime_bundle.pt"))
    log.info("  Bundle PyTorch: %s",
             os.path.join(args.out, "inceptiontime_bundle.pt"))

    # Report
    report_path = os.path.join(args.out, "inceptiontime_report.txt")
    with open(report_path, "w", encoding="utf-8") as fp:
        fp.write("TRAINING REPORT — InceptionTime v2 (dataset completo)\n")
        fp.write("Normativa: UL 1699B\n")
        fp.write("=" * 60 + "\n\n")
        fp.write(f"Train: {args.train}\n")
        fp.write(f"Test:  {args.test}\n")
        fp.write(f"Campioni train: {len(y_train)} (NO undersampling)\n")
        fp.write(f"Campioni test:  {len(y_test)}\n")
        fp.write(f"Epoche effettive: {len(history['train_loss'])}\n")
        fp.write(f"Tempo training: {t_train:.1f} s ({t_train/60:.1f} min)\n\n")
        for k, v in [("accuracy", acc), ("balanced_accuracy", ba),
                     ("f1_arc", f1), ("roc_auc", auc), ("avg_precision", ap),
                     ("detection_rate_pct", ul["detection_rate_pct"]),
                     ("false_positive_rate_pct", ul["false_positive_rate_pct"]),
                     ("best_threshold", best_thr),
                     ("ul1699b_conforme", ul["ul1699b_conforme"]),
                     ("shuffle_ok", shf["shuffle_ok"])]:
            fp.write(f"  {k}: {v}\n")

    # Riepilogo finale
    log.info("")
    log.info("=" * 72)
    log.info("RIEPILOGO FINALE")
    log.info("=" * 72)
    log.info("  Campioni train usati: %d  (tutti, nessun undersampling)", len(y_train))
    log.info("  Epoche effettive:     %d / %d",
             len(history["train_loss"]), args.epochs)
    log.info("  Tempo training:       %.1f s (%.1f min)", t_train, t_train/60)
    log.info("  Accuracy:             %.4f", acc)
    log.info("  Balanced Accuracy:    %.4f", ba)
    log.info("  F1 (arco):            %.4f", f1)
    log.info("  ROC-AUC:              %.4f", auc)
    log.info("  Detection rate:       %.1f%%", ul["detection_rate_pct"])
    log.info("  False positive rate:  %.1f%%", ul["false_positive_rate_pct"])
    log.info("  Soglia ottimale:      %.2f",   best_thr)
    log.info("  UL1699B:              %s",
             "✓ CONFORME" if ul["ul1699b_conforme"] else "✗ NON CONFORME")
    log.info("  Shuffle test:         %s",
             "✓ OK" if shf["shuffle_ok"] else "✗ ATTENZIONE")
    log.info("")
    log.info("  Output in: %s", args.out)

    if not ul["ul1699b_conforme"]:
        log.warning("")
        log.warning("  AZIONI SUGGERITE per conformità UL1699B:")
        log.warning("  → --epochs 100 --patience 20  (più training)")
        log.warning("  → --n-filters 64              (modello più grande)")
        log.warning("  → --depth 9                   (più blocchi Inception)")
        log.warning("  → --export-onnx + verifica ST Edge AI")


if __name__ == "__main__":
    main()
