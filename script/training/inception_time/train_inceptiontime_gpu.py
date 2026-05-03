#!/usr/bin/env python3
"""
train_inceptiontime.py
======================
Addestra InceptionTime sullo stesso dataset e split usati per
MultiRocketHydra e produce un confronto diretto tra i due classificatori.

BACKEND: tsai + PyTorch  (NON aeon+TensorFlow)
  - Compatibile con CUDA 13.x e RTX 4060
  - Training su GPU automatico se disponibile
  - Export ONNX nativo via torch.onnx.export (non serve TensorFlow)

FIX: usa Learner diretto invece di TSClassifier per evitare il bug
     numpy.object_ causato dalla reinizializzazione interna dei dati.

Export ONNX:
  - inceptiontime.onnx        shape dinamica  ← inferenza generale
  - inceptiontime_static.onnx shape fissa     ← ST Edge AI quantizzazione INT8

Uso:
    python train_inceptiontime.py <arc_dataset_new.npz> [--out <cartella>]
                                  [--multirocket-report <training_report.txt>]
                                  [--epochs 50] [--batch-size 64]

Esempio:
    python train_inceptiontime.py arc_dataset_new.npz ^
        --out risultati_inception ^
        --multirocket-report risultati/training_report.txt

Requisiti:
    pip install tsai torch scikit-learn matplotlib seaborn onnxruntime
"""

import argparse
import logging
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # evita conflitto OpenMP su Windows

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch

from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import (
    balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
)

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ── costanti ──────────────────────────────────────────────────────────────────
FS_HZ      = 10_000
TEST_SIZE  = 0.20
RAND       = 42
BATCH_SIZE = 64    # batch grande per sfruttare la GPU (RTX 4060 8GB VRAM)

UL_MIN_DET = 95.0
UL_MAX_FP  = 5.0


# ══════════════════════════════════════════════════════════════════════════════
# Utility — metriche
# ══════════════════════════════════════════════════════════════════════════════

def ul1699b_metric(y_test: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcola e stampa le metriche UL1699B."""
    arc    = y_test == 1
    no_arc = y_test == 0
    det    = int(((y_pred == 1) & arc).sum())
    miss   = int(((y_pred == 0) & arc).sum())
    fp     = int(((y_pred == 1) & no_arc).sum())
    tn     = int(((y_pred == 0) & no_arc).sum())
    det_r  = 100.0 * det / max(int(arc.sum()), 1)
    fp_r   = 100.0 * fp  / max(int(no_arc.sum()), 1)
    ok     = det_r >= UL_MIN_DET and fp_r <= UL_MAX_FP

    log.info("  --- UL1699B (soglia 0.5) ---")
    log.info("  Archi:       %d  → rilevati %d (%.1f%%), mancati %d",
             int(arc.sum()), det, det_r, miss)
    log.info("  Senza arco:  %d  → FP %d (%.1f%%), TN %d",
             int(no_arc.sum()), fp, fp_r, tn)
    log.info("  Esito:       %s", "CONFORME" if ok else "NON conforme")

    return {
        "detected": det, "missed": miss,
        "false_positives": fp, "true_negatives": tn,
        "detection_rate_pct":      round(det_r, 2),
        "false_positive_rate_pct": round(fp_r, 2),
        "ul1699b_conforme":        ok,
    }


def threshold_analysis(y_test: np.ndarray, y_proba: np.ndarray) -> float:
    """Analisi multi-soglia, restituisce la soglia ottimale."""
    log.info("  --- Analisi soglie ---")
    log.info("  %s  %s  %s  %s",
             "Soglia".rjust(8), "Det%".rjust(7),
             "FP%".rjust(6), "UL1699B".rjust(9))
    best_thr = 0.5
    best_det = 0.0
    for thr in np.arange(0.10, 0.55, 0.05):
        yp     = (y_proba >= thr).astype(int)
        arc    = y_test == 1
        no_arc = y_test == 0
        d      = 100 * ((yp == 1) & arc).sum()    / max(arc.sum(), 1)
        f      = 100 * ((yp == 1) & no_arc).sum() / max(no_arc.sum(), 1)
        ok     = "SI" if d >= UL_MIN_DET and f <= UL_MAX_FP else "NO"
        log.info("  %8.2f  %6.1f%%  %5.1f%%  %9s", thr, d, f, ok)
        if d >= UL_MIN_DET and f <= UL_MAX_FP and d > best_det:
            best_det = d
            best_thr = float(thr)
    log.info("  Soglia ottimale: %.2f", best_thr)
    return best_thr


def undersample(X: np.ndarray, y: np.ndarray, max_per_class: int = 3000):
    """Undersampling bilanciato — stessa logica di train_classifier.py."""
    n_min = min((y == 0).sum(), (y == 1).sum(), max_per_class)
    rng   = np.random.default_rng(RAND)
    idx0  = rng.choice(np.where(y == 0)[0], size=n_min, replace=False)
    idx1  = rng.choice(np.where(y == 1)[0], size=n_min, replace=False)
    idx   = np.concatenate([idx0, idx1])
    rng.shuffle(idx)
    log.info("  Undersampling: %d → %d campioni (%d per classe, max=%d)",
             len(y), len(idx), n_min, max_per_class)
    return X[idx], y[idx]


# ══════════════════════════════════════════════════════════════════════════════
# InceptionTime con tsai + PyTorch
# ══════════════════════════════════════════════════════════════════════════════

def train_inceptiontime_tsai(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    epochs:  int,
    batch_size: int,
    device: torch.device,
    out_dir: str,
) -> tuple:
    """
    Addestra InceptionTime con tsai su GPU.

    FIX numpy.object_: usa Learner diretto invece di TSClassifier.
    TSClassifier reinizializza internamente i dati perdendo il dtype,
    Learner accetta i dls già costruiti senza toccarli.

    Shape input: (n_campioni, n_canali=1, n_timepoints)

    Returns:
        y_pred, y_proba, history, learn, t_train
    """
    try:
        from tsai.all import get_ts_dls, InceptionTime, accuracy
        from fastai.learner import Learner
        from fastai.losses import CrossEntropyLossFlat
    except ImportError:
        log.error("tsai/fastai non installati. Eseguire: pip install tsai")
        sys.exit(1)

    # Cast esplicito a tipi supportati da torch — evita numpy.object_
    X_tr = np.array(X_train, dtype=np.float32)
    X_te = np.array(X_test,  dtype=np.float32)
    y_tr = np.array(y_train, dtype=np.int64)
    y_te = np.array(y_test,  dtype=np.int64)

    log.info("  Preparazione DataLoaders tsai...")
    log.info("  X dtype=%s  y dtype=%s", X_tr.dtype, y_tr.dtype)

    # Concatena train+test con split esplicito
    X_all  = np.concatenate([X_tr, X_te], axis=0)
    y_all  = np.concatenate([y_tr, y_te], axis=0)
    splits = (
        list(range(len(X_tr))),
        list(range(len(X_tr), len(X_tr) + len(X_te))),
    )

    dls = get_ts_dls(
        X_all, y_all,
        splits=splits,
        bs=batch_size,
        device=device,
    )

    # Costruisce InceptionTime direttamente — evita TSClassifier
    log.info("  Costruzione modello InceptionTime...")
    n_ch  = X_tr.shape[1]          # numero canali (1)
    n_cls = len(np.unique(y_tr))   # numero classi (2)
    model = InceptionTime(n_ch, n_cls).to(device)

    log.info("  Parametri modello: %s",
             f"{sum(p.numel() for p in model.parameters()):,}")
    log.info("  Device: %s", device)

    # Learner fastai diretto
    learn = Learner(
        dls,
        model,
        loss_func=CrossEntropyLossFlat(),
        metrics=[accuracy],
    )

    log.info("  Training con 1cycle LR policy...")
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    t0 = time.time()
    learn.fit_one_cycle(epochs, 1e-3)
    t_train = time.time() - t0

    # Estrai history dal recorder di fastai
    try:
        for row in learn.recorder.values:
            history["train_loss"].append(float(row[0]))
            history["val_loss"].append(float(row[1]))
            if len(row) > 2:
                history["val_acc"].append(float(row[2]))
    except Exception:
        pass

    log.info("  Training completato in %.1f s (%.1f min)",
             t_train, t_train / 60)

    # Predizione sul validation set (= test set)
    log.info("  Predizione sul test set...")
    try:
        probs, _, preds = learn.get_preds(
            dl=dls.valid, with_decoded=True,
            act=torch.nn.Softmax(dim=1),
        )
        y_proba = probs[:, 1].numpy()
        y_pred  = preds.numpy().astype(int)
    except Exception as e:
        log.warning("  get_preds fallito (%s) — predizione manuale", e)
        model.eval()
        probs_list, preds_list = [], []
        with torch.no_grad():
            for xb, _ in dls.valid:
                out  = model(xb.to(device))
                prob = torch.softmax(out, dim=1)
                probs_list.append(prob.cpu())
                preds_list.append(prob.argmax(dim=1).cpu())
        y_proba = torch.cat(probs_list)[:, 1].numpy()
        y_pred  = torch.cat(preds_list).numpy().astype(int)

    return y_pred, y_proba, history, learn, t_train


# ══════════════════════════════════════════════════════════════════════════════
# Export ONNX — due versioni: dinamica e statica
# ══════════════════════════════════════════════════════════════════════════════

def export_onnx(learn, n_timepoints: int, onnx_path: str,
                device: torch.device) -> bool:
    """
    Esporta ONNX con shape DINAMICA — per inferenza generale e onnxruntime.

    Input:  (batch, 1, n_timepoints)  float32  — batch variabile
    Output: (batch, 2)               float32
    """
    try:
        net   = learn.model.cpu().eval()
        dummy = torch.zeros(1, 1, n_timepoints, dtype=torch.float32)

        torch.onnx.export(
            net,
            dummy,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=13,
            do_constant_folding=True,
        )

        size_kb = os.path.getsize(onnx_path) / 1024
        log.info("  ONNX dinamico: %s  (%.1f KB)", onnx_path, size_kb)
        log.info("  Input:  (batch, 1, %d)  float32", n_timepoints)
        log.info("  Output: (batch, 2)      float32")

        # Verifica con onnxruntime
        try:
            import onnxruntime as rt
            sess     = rt.InferenceSession(onnx_path)
            inp_name = sess.get_inputs()[0].name
            sample   = np.zeros((1, 1, n_timepoints), dtype=np.float32)
            out      = sess.run(None, {inp_name: sample})[0]
            log.info("  Verifica onnxruntime: output shape %s  ✓", out.shape)
        except ImportError:
            log.warning("  onnxruntime non installato — skip verifica")
        except Exception as e:
            log.warning("  Verifica onnxruntime: %s", e)

        learn.model.to(device)
        return True

    except Exception as e:
        log.error("  Export ONNX dinamico fallito: %s", e)
        return False


def export_onnx_static(learn, n_timepoints: int, onnx_path: str,
                        device: torch.device, batch_size: int = 1) -> bool:
    """
    Esporta ONNX con shape FISSA — richiesto da ST Edge AI per quantizzazione INT8.

    ST Edge AI non riesce a inferire le shape con assi dinamici e restituisce
    'list index out of range'. La shape fissa risolve il problema.

    batch_size=1 è il valore corretto per STM32 (un campione alla volta).

    Input fisso:  (1, 1, n_timepoints)  float32
    Output fisso: (1, 2)               float32
    """
    try:
        net   = learn.model.cpu().eval()
        dummy = torch.zeros(batch_size, 1, n_timepoints, dtype=torch.float32)

        torch.onnx.export(
            net,
            dummy,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=13,
            do_constant_folding=True,
            # NON passare dynamic_axes — shape fissa per ST Edge AI
        )

        size_kb = os.path.getsize(onnx_path) / 1024
        log.info("  ONNX statico:  %s  (%.1f KB)", onnx_path, size_kb)
        log.info("  Input fisso:  (%d, 1, %d)  float32", batch_size, n_timepoints)
        log.info("  Output fisso: (%d, 2)      float32", batch_size)
        log.info("  → usa questo file in ST Edge AI per la quantizzazione INT8")

        # Verifica con onnxruntime
        try:
            import onnxruntime as rt
            sess     = rt.InferenceSession(onnx_path)
            inp_name = sess.get_inputs()[0].name
            sample   = np.zeros((batch_size, 1, n_timepoints), dtype=np.float32)
            out      = sess.run(None, {inp_name: sample})[0]
            log.info("  Verifica onnxruntime: output shape %s  ✓", out.shape)
        except ImportError:
            log.warning("  onnxruntime non installato — skip verifica")
        except Exception as e:
            log.warning("  Verifica onnxruntime: %s", e)

        learn.model.to(device)
        return True

    except Exception as e:
        log.error("  Export ONNX statico fallito: %s", e)
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Grafici
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_history(history: dict, out_dir: str):
    """Salva le curve di loss e accuracy durante il training."""
    if not history["train_loss"]:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("InceptionTime (tsai/PyTorch) — Curve di Training", fontsize=12)

    ax = axes[0]
    ax.plot(history["train_loss"], label="Train loss", color="steelblue")
    if history["val_loss"]:
        ax.plot(history["val_loss"], label="Val loss", color="tomato", ls="--")
    ax.set_title("Loss"); ax.set_xlabel("Epoch")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    if history["val_acc"]:
        ax.plot(history["val_acc"], label="Val acc", color="tomato", ls="--")
    ax.axhline(0.95, color="red", ls=":", lw=1, label="95% UL1699B")
    ax.set_title("Accuracy"); ax.set_xlabel("Epoch")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "inceptiontime_training.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    log.info("  Curva training salvata: %s", path)


def plot_results(y_test, y_pred, y_proba, out_dir):
    """Salva confusion matrix, ROC, PR e distribuzione score."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle("Risultati — InceptionTime (tsai/PyTorch)", fontsize=13)

    ax = axes[0]
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", ax=ax,
                xticklabels=["No arco", "Arco"],
                yticklabels=["No arco", "Arco"])
    ax.set_title("Confusion Matrix")
    ax.set_ylabel("Reale"); ax.set_xlabel("Predetto")

    ax = axes[1]
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC={auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.legend()
    ax.set_title("ROC Curve")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.grid(alpha=0.3)

    ax = axes[2]
    if y_proba is not None:
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        ax.plot(rec, prec, color="darkorange", lw=2, label=f"AP={ap:.3f}")
        ax.axhline(y_test.mean(), color="gray", ls="--", lw=1)
        ax.legend()
    ax.set_title("Precision-Recall")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.grid(alpha=0.3)

    ax = axes[3]
    if y_proba is not None:
        ax.hist(y_proba[y_test == 0], bins=30, alpha=0.6,
                color="steelblue", label="No arco")
        ax.hist(y_proba[y_test == 1], bins=30, alpha=0.6,
                color="darkorange", label="Arco")
        ax.axvline(0.5, color="black", ls="--", lw=1, label="soglia=0.5")
        ax.legend()
    ax.set_title("Distribuzione score")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "results_inceptiontime.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    log.info("  Grafico risultati salvato: %s", path)


def plot_comparison(mr: dict, it: dict, out_dir: str):
    """Grafico di confronto MultiRocketHydra vs InceptionTime."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Confronto — MultiRocketHydra vs InceptionTime", fontsize=13)

    colors = ["steelblue", "darkorange"]
    labels = ["MultiRocketHydra", "InceptionTime"]

    ax = axes[0]
    metric_names = ["Bal.Acc", "F1", "AUC", "Det%/100", "1-FP%"]
    x = np.arange(len(metric_names))
    w = 0.35
    for i, r in enumerate([mr, it]):
        vals = [
            r["balanced_accuracy"],
            r["f1_arc"],
            r.get("roc_auc", 0) or 0,
            r["detection_rate_pct"] / 100,
            1 - r["false_positive_rate_pct"] / 100,
        ]
        ax.bar(x + (i - 0.5) * w, vals, w,
               label=labels[i], color=colors[i], alpha=0.85)
    ax.axhline(0.95, color="red", ls="--", lw=1, label="95% UL1699B")
    ax.set_xticks(x); ax.set_xticklabels(metric_names, fontsize=9)
    ax.set_ylim(0.8, 1.02); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Confronto metriche")

    for col, (r, label) in enumerate([(mr, "MultiRocketHydra"),
                                       (it, "InceptionTime")]):
        ax = axes[col + 1]
        if r.get("confusion_matrix") is not None:
            sns.heatmap(r["confusion_matrix"], annot=True, fmt="d",
                        cmap="Blues" if col == 0 else "Oranges", ax=ax,
                        xticklabels=["No arco", "Arco"],
                        yticklabels=["No arco", "Arco"])
            ax.set_title(label)
            ax.set_ylabel("Reale"); ax.set_xlabel("Predetto")
        else:
            ax.text(0.5, 0.5, f"{label}\n(dati non disponibili)",
                    ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")

    plt.tight_layout()
    path = os.path.join(out_dir, "confronto_inception_vs_multirocket.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    log.info("  Grafico confronto salvato: %s", path)


# ══════════════════════════════════════════════════════════════════════════════
# Carica risultati MultiRocketHydra dal report testuale
# ══════════════════════════════════════════════════════════════════════════════

def load_multirocket_results(report_path: str) -> dict | None:
    """Legge training_report.txt e restituisce le metriche come dict."""
    if not report_path or not os.path.isfile(report_path):
        return None
    results = {}
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if ": " in line:
                    k, v = line.split(": ", 1)
                    k = k.strip(); v = v.strip()
                    try:
                        results[k] = float(v)
                    except ValueError:
                        if v == "True":    results[k] = True
                        elif v == "False": results[k] = False
                        else:              results[k] = v
        log.info("Risultati MultiRocketHydra caricati da: %s", report_path)
        return results
    except Exception as e:
        log.warning("Impossibile leggere report MultiRocketHydra: %s", e)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Training InceptionTime (tsai/PyTorch) + confronto MultiRocketHydra",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dataset",
                        help="Percorso al file arc_dataset_new.npz")
    parser.add_argument("--out", "-o", default="./risultati_inception",
                        help="Cartella output (default: ./risultati_inception)")
    parser.add_argument("--multirocket-report", default=None,
                        help="Path al training_report.txt di MultiRocketHydra")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Epoche di training (default: 50)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size GPU (default: {BATCH_SIZE})")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    args = parser.parse_args()

    if not os.path.isfile(args.dataset):
        log.error("File non trovato: %s", args.dataset)
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)

    # ── GPU check ─────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device   = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
        log.info("GPU rilevata: %s  (%.1f GB VRAM)", gpu_name, gpu_mem)
    else:
        device = torch.device("cpu")
        log.warning("Nessuna GPU CUDA disponibile — training su CPU")
        log.warning("Per abilitare GPU:")
        log.warning("  pip install torch --index-url https://download.pytorch.org/whl/cu121")

    # ── carica dataset ────────────────────────────────────────────────────────
    log.info("Caricamento dataset: %s", args.dataset)
    data = np.load(args.dataset)
    X    = data["X"]
    y    = data["y"]
    log.info("  X shape: %s  (%.1f ms @ %d Hz)",
             X.shape, X.shape[1] / FS_HZ * 1000, FS_HZ)
    log.info("  y: arco=%d  no_arco=%d",
             int((y == 1).sum()), int((y == 0).sum()))

    # ── split identico a train_classifier.py ─────────────────────────────────
    meta_path = args.dataset.replace("arc_dataset_new.npz",
                                     "arc_dataset_meta_new.csv")
    groups = None
    if os.path.isfile(meta_path):
        import pandas as pd
        meta = pd.read_csv(meta_path, encoding="latin-1")
        def _key(fn):
            s   = fn.replace("_Raw Data.mat", "").replace(" Data.mat", "")
            idx = s.lower().rfind("_study")
            return s[:idx] if idx > 0 else s
        meta["exp_key"] = meta["filename"].apply(_key)
        unique_keys     = {k: i for i, k in enumerate(meta["exp_key"].unique())}
        groups          = meta["exp_key"].map(unique_keys).values
        log.info("  Gruppi sperimentali: %d", len(unique_keys))

    if groups is not None:
        gss = GroupShuffleSplit(
            n_splits=1, test_size=args.test_size, random_state=RAND
        )
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        log.info("  Split per gruppo sperimentale (%.0f%%/%.0f%%)",
                 (1 - args.test_size) * 100, args.test_size * 100)
    else:
        train_idx, test_idx = train_test_split(
            np.arange(len(y)), test_size=args.test_size,
            random_state=RAND, stratify=y,
        )
        log.warning("  Metadati non trovati — split casuale (rischio data leakage)")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    log.info("  Train: %d  (arco=%d, no=%d)",
             len(y_train), int((y_train==1).sum()), int((y_train==0).sum()))
    log.info("  Test:  %d  (arco=%d, no=%d)",
             len(y_test),  int((y_test==1).sum()),  int((y_test==0).sum()))

    # ── undersampling ─────────────────────────────────────────────────────────
    n_samples_per_series = X_train.shape[1]
    if n_samples_per_series <= 500:
        max_pc = 5000
    elif n_samples_per_series <= 2000:
        max_pc = 3000   # caso attuale: 100ms @ 10kHz = 1000 campioni
    else:
        max_pc = 500
    log.info("  Serie da %d campioni (%.0f ms) → max_per_class=%d",
             n_samples_per_series,
             n_samples_per_series / FS_HZ * 1000,
             max_pc)
    X_train, y_train = undersample(X_train, y_train, max_per_class=max_pc)

    # tsai vuole shape (n, canali, timepoints)
    X_tr = X_train[:, np.newaxis, :].astype(np.float32)
    X_te = X_test[:,  np.newaxis, :].astype(np.float32)

    # ── training ──────────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("TRAINING: InceptionTime (tsai + PyTorch)")
    log.info("=" * 60)
    log.info("  Epoche:     %d", args.epochs)
    log.info("  Batch size: %d", args.batch_size)
    log.info("  Train:      %d campioni", len(y_train))
    log.info("  Test:       %d campioni (intero test set)", len(y_test))
    log.info("  Device:     %s", device)

    y_pred, y_proba, history, learn, t_train = train_inceptiontime_tsai(
        X_tr, y_train, X_te, y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        out_dir=args.out,
    )

    # ── metriche ──────────────────────────────────────────────────────────────
    log.info("")
    log.info("  --- Metriche standard ---")
    report = classification_report(
        y_test, y_pred,
        target_names=["No arco", "Arco"], digits=3,
    )
    for line in report.splitlines():
        log.info("  %s", line)

    ba  = balanced_accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)           if y_proba is not None else None
    ap  = average_precision_score(y_test, y_proba) if y_proba is not None else None

    log.info("  Balanced Accuracy: %.4f", ba)
    log.info("  F1 (arco):         %.4f", f1)
    if auc: log.info("  ROC-AUC:           %.4f", auc)
    if ap:  log.info("  Avg Precision:     %.4f", ap)

    ul      = ul1699b_metric(y_test, y_pred)
    best_th = threshold_analysis(y_test, y_proba) if y_proba is not None else 0.5
    cm      = confusion_matrix(y_test, y_pred)

    # ── grafici ───────────────────────────────────────────────────────────────
    plot_training_history(history, args.out)
    plot_results(y_test, y_pred, y_proba, args.out)

    # ── export ONNX ───────────────────────────────────────────────────────────
    log.info("")
    log.info("  --- Export ONNX ---")
    n_tp = X_tr.shape[-1]

    # 1. ONNX dinamico — per onnxruntime e inferenza batch
    onnx_path = os.path.join(args.out, "inceptiontime.onnx")
    export_onnx(learn, n_timepoints=n_tp, onnx_path=onnx_path, device=device)

    # 2. ONNX statico — per ST Edge AI quantizzazione INT8
    #    Risolve l'errore "list index out of range" causato dagli assi dinamici
    onnx_static_path = os.path.join(args.out, "inceptiontime_static.onnx")
    export_onnx_static(learn, n_timepoints=n_tp,
                       onnx_path=onnx_static_path, device=device, batch_size=1)

    # ── risultati InceptionTime ───────────────────────────────────────────────
    it_result = {
        "model_name":              "InceptionTime",
        "backend":                 "tsai+PyTorch",
        "device":                  str(device),
        "train_time_s":            round(t_train, 1),
        "balanced_accuracy":       round(ba,  4),
        "f1_arc":                  round(f1,  4),
        "roc_auc":                 round(auc, 4) if auc else None,
        "avg_precision":           round(ap,  4) if ap  else None,
        "best_threshold":          round(best_th, 2),
        "confusion_matrix":        cm,
        **ul,
    }

    # ── confronto con MultiRocketHydra ────────────────────────────────────────
    log.info("")
    log.info("=" * 72)
    log.info("CONFRONTO — MultiRocketHydra vs InceptionTime")
    log.info("=" * 72)

    mr_raw    = load_multirocket_results(args.multirocket_report)
    mr_result = None
    if mr_raw:
        mr_result = {
            "model_name":              "MultiRocketHydra",
            "train_time_s":            mr_raw.get("train_time_s", 0),
            "balanced_accuracy":       mr_raw.get("balanced_accuracy", 0),
            "f1_arc":                  mr_raw.get("f1_arc", 0),
            "roc_auc":                 mr_raw.get("roc_auc", None),
            "avg_precision":           mr_raw.get("avg_precision", None),
            "best_threshold":          mr_raw.get("best_threshold", 0.5),
            "detection_rate_pct":      mr_raw.get("detection_rate_pct", 0),
            "false_positive_rate_pct": mr_raw.get("false_positive_rate_pct", 0),
            "ul1699b_conforme":        mr_raw.get("ul1699b_conforme", False),
            "confusion_matrix":        None,
        }

    all_results = [r for r in [mr_result, it_result] if r]
    log.info("  %-20s %8s %7s %7s %7s %6s %7s %9s",
             "Modello", "BA", "F1", "AUC", "Det%", "FP%", "T(s)", "UL1699B")
    log.info("  " + "-" * 72)
    for r in all_results:
        ok = "CONFORME" if r["ul1699b_conforme"] else "NO"
        log.info("  %-20s %8.4f %7.4f %7s %6.1f%% %5.1f%% %7.1f %9s",
                 r["model_name"],
                 r["balanced_accuracy"],
                 r["f1_arc"],
                 f"{r['roc_auc']:.4f}" if r["roc_auc"] else "N/A",
                 r["detection_rate_pct"],
                 r["false_positive_rate_pct"],
                 r["train_time_s"],
                 ok)

    if mr_result:
        plot_comparison(mr_result, it_result, args.out)

    # ── report testuale ───────────────────────────────────────────────────────
    report_path = os.path.join(args.out, "inceptiontime_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("TRAINING REPORT — InceptionTime (tsai + PyTorch)\n")
        f.write("Normativa di riferimento: UL 1699B\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset:    {args.dataset}\n")
        f.write(f"Epoche:     {args.epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Device:     {device}\n\n")
        for r in all_results:
            f.write(f"\n{'='*40}\n{r['model_name']}\n{'='*40}\n")
            for k, v in r.items():
                if k not in ("confusion_matrix", "model_name"):
                    f.write(f"  {k}: {v}\n")

    log.info("")
    log.info("Output in: %s", args.out)
    log.info("  inceptiontime.onnx            ← inferenza generale / onnxruntime")
    log.info("  inceptiontime_static.onnx     ← ST Edge AI quantizzazione INT8")
    log.info("  inceptiontime_training.png")
    log.info("  results_inceptiontime.png")
    if mr_result:
        log.info("  confronto_inception_vs_multirocket.png")
    log.info("  inceptiontime_report.txt")


if __name__ == "__main__":
    main()