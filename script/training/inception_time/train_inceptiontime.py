#!/usr/bin/env python3
"""
train_inceptiontime.py
======================
Addestra InceptionTime su dataset arc fault detection e produce:
  - metriche UL1699B
  - grafici
  - export ONNX (dinamico + statico)
  - dataset di calibrazione per quantizzazione INT8 (ST Edge AI)

BACKEND: tsai + PyTorch  (NON aeon+TensorFlow)
  - Compatibile con CUDA 13.x e RTX 4060
  - Training su GPU automatico se disponibile
  - Export ONNX nativo via torch.onnx.export

FIX: usa Learner diretto invece di TSClassifier per evitare il bug
     numpy.object_ causato dalla reinizializzazione interna dei dati.

Export ONNX:
  - inceptiontime.onnx        shape dinamica  ← inferenza generale

Dataset calibrazione (generato automaticamente al termine del training):
  - calibration_data.npz      shape (N, 1, 1, 1000)  float32  ← ST Edge AI
  - calibration_data.npy      shape (N, 1, 1000)      float32  ← alternativo
  - calibration_data_flat.npy shape (N, 1000)          float32  ← fallback
  - calibration_labels.npy    label per verifica
  - calibration_info.txt      riepilogo

Uso:
    python train_inceptiontime.py [--out <cartella>]
                                  [--epochs 50] [--batch-size 64]
                                  [--n-cal 100]

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
from sklearn.utils.class_weight import compute_class_weight
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

# ── path dataset ──────────────────────────────────────────────────────────────
DATASET_TRAIN = r"C:\Users\Asus\Desktop\progetto_manutenzione\dataset\dataset_new\arc_dataset_train.npz"
DATASET_TEST  = r"C:\Users\Asus\Desktop\progetto_manutenzione\dataset\dataset_new\arc_dataset_test.npz"

# ── costanti ──────────────────────────────────────────────────────────────────
FS_HZ      = 10_000
RAND       = 42
BATCH_SIZE = 64

UL_MIN_DET = 95.0
UL_MAX_FP  = 5.0
N_CAL_DEFAULT = 100   # campioni per classe nel dataset di calibrazione


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
    """Undersampling bilanciato."""
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
    class_weights=None,
) -> tuple:
    """
    Addestra InceptionTime con tsai su GPU.

    FIX numpy.object_: usa Learner diretto invece di TSClassifier.
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

    X_tr = np.array(X_train, dtype=np.float32)
    X_te = np.array(X_test,  dtype=np.float32)
    y_tr = np.array(y_train, dtype=np.int64)
    y_te = np.array(y_test,  dtype=np.int64)

    log.info("  Preparazione DataLoaders tsai...")
    log.info("  X dtype=%s  y dtype=%s", X_tr.dtype, y_tr.dtype)

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

    log.info("  Costruzione modello InceptionTime...")
    n_ch  = X_tr.shape[1]
    n_cls = len(np.unique(y_tr))
    model = InceptionTime(n_ch, n_cls).to(device)

    log.info("  Parametri modello: %s",
             f"{sum(p.numel() for p in model.parameters()):,}")
    log.info("  Device: %s", device)

    learn = Learner(
        dls,
        model,
        loss_func=CrossEntropyLossFlat(weight=class_weights),
        metrics=[accuracy],
    )

    log.info("  Training con 1cycle LR policy...")
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    t0 = time.time()
    learn.fit_one_cycle(epochs, 1e-3)
    t_train = time.time() - t0

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
# Export ONNX
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
            net, dummy, onnx_path,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=13,
            do_constant_folding=True,
        )

        size_kb = os.path.getsize(onnx_path) / 1024
        log.info("  ONNX dinamico: %s  (%.1f KB)", onnx_path, size_kb)
        log.info("  Input:  (batch, 1, %d)  float32", n_timepoints)
        log.info("  Output: (batch, 2)      float32")

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



# ══════════════════════════════════════════════════════════════════════════════
# Dataset di calibrazione per ST Edge AI
# ══════════════════════════════════════════════════════════════════════════════

def generate_calibration_dataset(
    dataset_path: str,
    out_dir: str,
    n_per_class: int = N_CAL_DEFAULT,
):
    """
    Genera il dataset di calibrazione per la quantizzazione INT8 tramite
    ST Edge AI / X-CUBE-AI.

    Formati di output:
      - calibration_data.npz      shape (N, 1, 1, 1000)  float32  ← ST Edge AI
      - calibration_data.npy      shape (N, 1, 1000)      float32  ← alternativo
      - calibration_data_flat.npy shape (N, 1000)          float32  ← fallback
      - calibration_labels.npy    label per verifica
      - calibration_info.txt      riepilogo
    """
    log.info("")
    log.info("=" * 60)
    log.info("DATASET DI CALIBRAZIONE — ST Edge AI")
    log.info("=" * 60)
    log.info("  Sorgente: %s", dataset_path)
    log.info("  Campioni per classe: %d", n_per_class)

    data = np.load(dataset_path)
    X    = data["X"]
    y    = data["y"]

    log.info("  X shape: %s  dtype=%s", X.shape, X.dtype)
    log.info("  label=0 (no arco): %d", int((y == 0).sum()))
    log.info("  label=1 (arco):    %d", int((y == 1).sum()))

    # campionamento bilanciato
    rng  = np.random.default_rng(RAND)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    n0 = min(n_per_class, len(idx0))
    n1 = min(n_per_class, len(idx1))

    if n0 < n_per_class:
        log.warning("  label=0: richiesti %d ma disponibili solo %d", n_per_class, n0)
    if n1 < n_per_class:
        log.warning("  label=1: richiesti %d ma disponibili solo %d", n_per_class, n1)

    idx0_cal = rng.choice(idx0, size=n0, replace=False)
    idx1_cal = rng.choice(idx1, size=n1, replace=False)
    idx_cal  = np.concatenate([idx0_cal, idx1_cal])
    rng.shuffle(idx_cal)

    X_cal = X[idx_cal].astype(np.float32)
    y_cal = y[idx_cal]

    log.info("  Totale campioni selezionati: %d  (label=0: %d, label=1: %d)",
             len(idx_cal), n0, n1)
    log.info("  Statistiche X_cal:  min=%.4f  max=%.4f  mean=%.4f  std=%.4f",
             float(X_cal.min()), float(X_cal.max()),
             float(X_cal.mean()), float(X_cal.std()))

    # shape 3D: (N, 1, T) — formato training InceptionTime
    X_cal_3d = X_cal[:, np.newaxis, :]
    # shape 4D: (N, 1, 1, T) — formato atteso da ST Edge AI
    X_cal_4d = X_cal_3d[:, np.newaxis, :]

    # NPZ con chiave 'input' — da usare in ST Edge AI
    path_npz = os.path.join(out_dir, "calibration_data.npz")
    np.savez(path_npz, input=X_cal_4d)
    size_mb  = os.path.getsize(path_npz) / 1024 / 1024
    log.info("  Salvato NPZ (← USA QUESTO in ST Edge AI):")
    log.info("    %s  shape=%s  (%.2f MB)", path_npz, X_cal_4d.shape, size_mb)

    # NPY 3D alternativo
    path_npy = os.path.join(out_dir, "calibration_data.npy")
    np.save(path_npy, X_cal_3d)
    log.info("  Salvato NPY alternativo:")
    log.info("    %s  shape=%s", path_npy, X_cal_3d.shape)

    # NPY 2D flat — fallback versioni vecchie ST Edge AI
    path_flat = os.path.join(out_dir, "calibration_data_flat.npy")
    np.save(path_flat, X_cal)
    log.info("  Salvato NPY flat fallback:")
    log.info("    %s  shape=%s", path_flat, X_cal.shape)

    # label per verifica
    path_labels = os.path.join(out_dir, "calibration_labels.npy")
    np.save(path_labels, y_cal)
    log.info("  Salvato label (solo verifica): %s", path_labels)

    # report testuale
    report_path = os.path.join(out_dir, "calibration_info.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("DATASET DI CALIBRAZIONE — ST Edge AI / X-CUBE-AI\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Sorgente dataset:    {dataset_path}\n")
        f.write(f"Campioni per classe: {n_per_class}\n")
        f.write(f"Totale campioni:     {len(idx_cal)}\n")
        f.write(f"  label=0 (no arco): {n0}\n")
        f.write(f"  label=1 (arco):    {n1}\n\n")
        f.write("File generati:\n")
        f.write(f"  calibration_data.npz      shape={X_cal_4d.shape}  "
                f"dtype={X_cal_4d.dtype}  ← USA QUESTO in ST Edge AI\n")
        f.write(f"    chiave: 'input'\n")
        f.write(f"  calibration_data.npy      shape={X_cal_3d.shape}  "
                f"dtype={X_cal_3d.dtype}  ← alternativo\n")
        f.write(f"  calibration_data_flat.npy shape={X_cal.shape}  "
                f"dtype={X_cal.dtype}  ← fallback 2D\n")
        f.write(f"  calibration_labels.npy    shape={y_cal.shape}  "
                f"← solo per verifica\n\n")
        f.write("Statistiche X:\n")
        f.write(f"  min:  {float(X_cal.min()):.4f}\n")
        f.write(f"  max:  {float(X_cal.max()):.4f}\n")
        f.write(f"  mean: {float(X_cal.mean()):.4f}\n")
        f.write(f"  std:  {float(X_cal.std()):.4f}\n\n")
        f.write("Come usare in ST Edge AI:\n")
        f.write("  1. Importa il modello ONNX (inceptiontime_static.onnx)\n")
        f.write("  2. Seleziona quantizzazione INT8\n")
        f.write("  3. Carica calibration_data.npz — chiave: 'input'\n")
        f.write("     Se non accettato, prova: calibration_data.npy\n")
        f.write("  4. Avvia la quantizzazione\n")

    log.info("  Report salvato: %s", report_path)


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


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Training InceptionTime (tsai/PyTorch) con metriche UL1699B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--train", default=DATASET_TRAIN,
                        help=f"Path dataset train .npz (default: {DATASET_TRAIN})")
    parser.add_argument("--test",  default=DATASET_TEST,
                        help=f"Path dataset test .npz  (default: {DATASET_TEST})")
    parser.add_argument("--out", "-o", default="./risultati_inception",
                        help="Cartella output (default: ./risultati_inception)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Epoche di training (default: 50)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size GPU (default: {BATCH_SIZE})")
    parser.add_argument("--n-cal", type=int, default=N_CAL_DEFAULT,
                        help=f"Campioni per classe nel dataset di calibrazione "
                             f"(default: {N_CAL_DEFAULT})")
    args = parser.parse_args()

    for path, label in [(args.train, "TRAIN"), (args.test, "TEST")]:
        if not os.path.isfile(path):
            log.error("File %s non trovato: %s", label, path)
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

    # ── carica dataset ────────────────────────────────────────────────────────
    log.info("Caricamento dataset TRAIN: %s", args.train)
    data_tr = np.load(args.train)
    X_train = data_tr["X"]
    y_train = data_tr["y"]

    classes      = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=y_train
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    log.info("Class weights: %s", class_weights)

    log.info("Caricamento dataset TEST: %s", args.test)
    data_te = np.load(args.test)
    X_test  = data_te["X"]
    y_test  = data_te["y"]

    log.info("  Train: %d  (arco=%d, no=%d)",
             len(y_train), int((y_train == 1).sum()), int((y_train == 0).sum()))
    log.info("  Test:  %d  (arco=%d, no=%d)",
             len(y_test),  int((y_test  == 1).sum()), int((y_test  == 0).sum()))

    # ── reshape per tsai: (n, canali, timepoints) ─────────────────────────────
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
    log.info("  Test:       %d campioni", len(y_test))
    log.info("  Device:     %s", device)

    y_pred, y_proba, history, learn, t_train = train_inceptiontime_tsai(
        X_tr, y_train, X_te, y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        out_dir=args.out,
        class_weights=class_weights,
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

    # ── grafici ───────────────────────────────────────────────────────────────
    plot_training_history(history, args.out)
    plot_results(y_test, y_pred, y_proba, args.out)

    # ── export ONNX ───────────────────────────────────────────────────────────
    log.info("")
    log.info("  --- Export ONNX ---")
    n_tp = X_tr.shape[-1]

    onnx_path = os.path.join(args.out, "inceptiontime.onnx")
    export_onnx(learn, n_timepoints=n_tp, onnx_path=onnx_path, device=device)

    # ── dataset di calibrazione ───────────────────────────────────────────────
    generate_calibration_dataset(
        dataset_path=args.train,
        out_dir=args.out,
        n_per_class=args.n_cal,
    )

    # ── report testuale ───────────────────────────────────────────────────────
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
        **ul,
    }

    report_path = os.path.join(args.out, "inceptiontime_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("TRAINING REPORT — InceptionTime (tsai + PyTorch)\n")
        f.write("Normativa di riferimento: UL 1699B\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset train: {args.train}\n")
        f.write(f"Dataset test:  {args.test}\n")
        f.write(f"Epoche:        {args.epochs}\n")
        f.write(f"Batch size:    {args.batch_size}\n")
        f.write(f"Device:        {device}\n\n")
        f.write("=" * 40 + "\n")
        f.write("Risultati\n")
        f.write("=" * 40 + "\n")
        for k, v in it_result.items():
            if k != "model_name":
                f.write(f"  {k}: {v}\n")

    log.info("")
    log.info("=" * 60)
    log.info("OUTPUT: %s", args.out)
    log.info("=" * 60)
    log.info("  inceptiontime.onnx            ← inferenza generale / onnxruntime")
    log.info("  inceptiontime_training.png")
    log.info("  results_inceptiontime.png")
    log.info("  inceptiontime_report.txt")
    log.info("  calibration_data.npz          ← dataset calibrazione ST Edge AI")
    log.info("  calibration_data.npy          ← alternativo")
    log.info("  calibration_data_flat.npy     ← fallback 2D")
    log.info("  calibration_labels.npy        ← label per verifica")
    log.info("  calibration_info.txt          ← riepilogo calibrazione")


if __name__ == "__main__":
    main()