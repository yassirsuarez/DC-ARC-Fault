#!/usr/bin/env python3
"""
train_inceptiontime.py
======================
Addestra InceptionTimeClassifier sullo stesso dataset e split usati per
MultiRocketHydra e produce un confronto diretto tra i due classificatori.

InceptionTime è una rete neurale profonda basata sul modulo Inception di Google,
progettata specificamente per la classificazione di serie temporali.
Vantaggi rispetto a MultiRocketHydra:
  - Esportabile nativamente in ONNX via torch.onnx.export
  - Deployabile su STM32H7 via X-CUBE-AI senza implementazione C custom
  - Architettura profonda che cattura pattern a scale temporali diverse

Il confronto usa esattamente lo stesso GroupShuffleSplit per esperimento
per garantire un confronto equo con MultiRocketHydra.

Uso:
    python train_inceptiontime.py <arc_dataset.npz> [--out <cartella>]
                                  [--multirocket-report <training_report.txt>]
                                  [--epochs 50] [--batch-size 32]

Esempio:
    python train_inceptiontime.py arc_dataset.npz ^
        --out results_inception ^
        --multirocket-report results/training_report.txt

Requisiti:
    pip install aeon torch scikit-learn matplotlib seaborn

Autori: progetto tesi magistrale — Manutenzione e Affidabilità
Normativa di riferimento: UL 1699B
"""

import argparse
import logging
import os
import pickle
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

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
BATCH_SIZE = 16    # batch piccolo per gestire la RAM con serie da 20.000 pt

UL_MIN_DET = 95.0
UL_MAX_FP  = 5.0


# ══════════════════════════════════════════════════════════════════════════════
# Utility
# ══════════════════════════════════════════════════════════════════════════════

def ul1699b(y_test: np.ndarray, y_pred: np.ndarray) -> dict:
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
    log.info("  Archi:        %d  →  rilevati %d (%.1f%%), mancati %d",
             int(arc.sum()), det, det_r, miss)
    log.info("  Senza arco:   %d  →  FP %d (%.1f%%), TN %d",
             int(no_arc.sum()), fp, fp_r, tn)
    log.info("  Esito:        %s", "CONFORME" if ok else "NON conforme")

    return {
        "detected": det, "missed": miss,
        "false_positives": fp, "true_negatives": tn,
        "detection_rate_pct":      round(det_r, 2),
        "false_positive_rate_pct": round(fp_r, 2),
        "ul1699b_conforme":        ok,
    }


def threshold_analysis(y_test: np.ndarray, y_proba: np.ndarray) -> float:
    """Analisi multi-soglia e restituzione della soglia ottimale."""
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


def undersample(X: np.ndarray, y: np.ndarray, max_per_class: int = 300):
    """Undersampling bilanciato — stessa logica di train_classifier.py."""
    n_min = min((y == 0).sum(), (y == 1).sum(), max_per_class)
    rng   = np.random.default_rng(RAND)
    idx0  = rng.choice(np.where(y == 0)[0], size=n_min, replace=False)
    idx1  = rng.choice(np.where(y == 1)[0], size=n_min, replace=False)
    idx   = np.concatenate([idx0, idx1])
    rng.shuffle(idx)
    log.info("  Undersampling: %d → %d campioni (%d per classe)",
             len(y), len(idx), n_min)
    return X[idx], y[idx]


# ══════════════════════════════════════════════════════════════════════════════
# Grafici
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_history(history: dict, out_dir: str):
    """Salva le curve di loss e accuracy durante il training."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("InceptionTime — Curve di Training", fontsize=12)

    ax = axes[0]
    ax.plot(history["train_loss"], label="Train loss", color="steelblue")
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="Val loss", color="tomato", ls="--")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(history["train_acc"], label="Train acc", color="steelblue")
    if "val_acc" in history:
        ax.plot(history["val_acc"], label="Val acc", color="tomato", ls="--")
    ax.axhline(0.95, color="red", ls=":", lw=1, label="95% UL1699B")
    ax.set_title("Accuracy")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "inceptiontime_training.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    log.info("  Curva training salvata: %s", path)


def plot_results(y_test, y_pred, y_proba, out_dir):
    """Salva confusion matrix, ROC, PR e distribuzione score."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle("Risultati — InceptionTime", fontsize=13)

    # Confusion Matrix
    ax = axes[0]
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", ax=ax,
                xticklabels=["No arco", "Arco"],
                yticklabels=["No arco", "Arco"])
    ax.set_title("Confusion Matrix")
    ax.set_ylabel("Reale"); ax.set_xlabel("Predetto")

    # ROC
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

    # PR
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

    # Score distribution
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

    # Barchart metriche
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

    # Confusion matrices affiancate
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
# Export ONNX InceptionTime (Keras → tf2onnx)
# ══════════════════════════════════════════════════════════════════════════════

def _export_inceptiontime_onnx(clf, X_train: np.ndarray, onnx_path: str) -> bool:
    """
    Esporta il modello InceptionTime in formato ONNX.

    InceptionTime usa Keras/TensorFlow internamente, NON PyTorch.
    Il modello Keras addestrato è accessibile tramite clf.model_ dopo il fit.

    Pipeline di export:
      clf.model_  (Keras)
           ↓  tf2onnx.convert.from_keras
      inceptiontime.onnx  →  importabile in STM32Cube.AI / X-CUBE-AI

    Input ONNX:  (batch, 1, n_timepoints)  float32
    Output ONNX: (batch, n_classes)        float32  (probabilità)

    Requisiti:
      pip install tf2onnx tensorflow

    Returns:
        True se export riuscito, False altrimenti.
    """
    # 1. Recupera il modello Keras
    keras_model = getattr(clf, "model_", None)
    if keras_model is None:
        # Prova attributi alternativi usati in versioni diverse di aeon
        for attr in ["model", "_model", "network_", "_network", "training_model_"]:
            keras_model = getattr(clf, attr, None)
            if keras_model is not None:
                log.info("  Modello Keras trovato in: clf.%s", attr)
                break

    if keras_model is None:
        log.warning("  Modello Keras non trovato in nessun attributo noto")
        log.info("  Attributi disponibili nel classificatore:")
        for a in sorted(dir(clf)):
            if not a.startswith("__"):
                try:
                    val = getattr(clf, a)
                    if not callable(val):
                        log.info("    %s: %s", a, type(val).__name__)
                except Exception:
                    pass
        log.warning("  Suggerimento: installa tf2onnx e riprova con aeon aggiornato")
        return False

    log.info("  Modello Keras: %s", keras_model.__class__.__name__)
    n_timepoints = X_train.shape[-1]

    # 2. Export via tf2onnx
    try:
        import tf2onnx
        import tensorflow as tf

        log.info("  tf2onnx versione: %s", tf2onnx.__version__)
        log.info("  TensorFlow versione: %s", tf.__version__)

        # Specifica input: (batch, canali=1, timepoints)
        input_spec = (
            tf.TensorSpec(
                shape=(None, 1, n_timepoints),
                dtype=tf.float32,
                name="input",
            ),
        )

        model_proto, _ = tf2onnx.convert.from_keras(
            keras_model,
            input_signature=input_spec,
            opset=13,
            output_path=onnx_path,
        )

        size_kb = os.path.getsize(onnx_path) / 1024
        log.info("  ONNX salvato: %s  (%.1f KB)", onnx_path, size_kb)
        log.info("  Input:  (batch, 1, %d)  float32", n_timepoints)
        log.info("  Output: (batch, 2)      float32  (prob no-arco, prob arco)")

        # 3. Verifica rapida con onnxruntime
        try:
            import onnxruntime as rt
            sess     = rt.InferenceSession(onnx_path)
            inp_name = sess.get_inputs()[0].name
            sample   = X_train[:2].astype(np.float32)
            out      = sess.run(None, {inp_name: sample})[0]
            log.info("  Verifica onnxruntime: output shape %s  ✓", out.shape)
        except ImportError:
            log.warning("  onnxruntime non installato — skip verifica")
        except Exception as e:
            log.warning("  Verifica onnxruntime fallita: %s", e)

        return True

    except ImportError:
        log.error("  tf2onnx non installato")
        log.error("  Eseguire: pip install tf2onnx tensorflow")
        return False
    except Exception as e:
        log.error("  Export ONNX fallito: %s", e)

        # Fallback: salva come SavedModel (importabile in STM32Cube.AI)
        try:
            saved_dir = onnx_path.replace(".onnx", "_savedmodel")
            keras_model.save(saved_dir)
            log.info("  Fallback: SavedModel salvato in %s", saved_dir)
            log.info("  Usa STM32Cube.AI con formato SavedModel invece di ONNX")
        except Exception as e2:
            log.error("  Fallback SavedModel fallito: %s", e2)

        return False



# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Training InceptionTime e confronto con MultiRocketHydra",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dataset",
                        help="Percorso al file arc_dataset_new.npz")
    parser.add_argument("--out", "-o", default="./results_inception",
                        help="Cartella output (default: ./results_inception)")
    parser.add_argument("--multirocket-report", default=None,
                        help="Path al training_report.txt di MultiRocketHydra")
    parser.add_argument("--epochs", type=int, default=150,
                        help="Epoche di training (default: 50)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    args = parser.parse_args()

    if not os.path.isfile(args.dataset):
        log.error("File non trovato: %s", args.dataset)
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)

    # ── carica dataset ────────────────────────────────────────────────────────
    log.info("Caricamento dataset: %s", args.dataset)
    data = np.load(args.dataset)
    X    = data["X"]
    y    = data["y"]
    log.info("  X shape: %s  (%.1f s @ %d Hz)", X.shape,
             X.shape[1] / FS_HZ, FS_HZ)
    log.info("  y: arco=%d  no_arco=%d",
             int((y == 1).sum()), int((y == 0).sum()))

    # ── split identico a MultiRocketHydra ────────────────────────────────────
    meta_path = args.dataset.replace("arc_dataset_new.npz", "arc_dataset_meta_new.csv")
    groups    = None
    if os.path.isfile(meta_path):
        import pandas as pd
        meta = pd.read_csv(meta_path, encoding='latin-1')
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
        log.info("  Split per gruppo sperimentale")
    else:
        train_idx, test_idx = train_test_split(
            np.arange(len(y)), test_size=args.test_size,
            random_state=RAND, stratify=y
        )
        log.warning("  Metadati non trovati — split casuale")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    log.info("  Train: %d  Test: %d", len(y_train), len(y_test))

    # ── undersampling (stessa logica di train_classifier.py) ─────────────────
    X_train, y_train = undersample(X_train, y_train, max_per_class=300)

    # ── InceptionTime vuole shape (n, n_channels, n_timepoints) ─────────────
    X_tr = X_train[:, np.newaxis, :].astype(np.float32)
    X_te = X_test[:,  np.newaxis, :].astype(np.float32)

    # ── Import InceptionTime ──────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("TRAINING: InceptionTime")
    log.info("=" * 60)
    log.info("  Epoche:     %d", args.epochs)
    log.info("  Batch size: %d", args.batch_size)
    log.info("  Train:      %d campioni", len(y_train))
    log.info("  Test:       %d campioni", len(y_test))

    try:
        from aeon.classification.deep_learning import InceptionTimeClassifier
    except ImportError as e:
        log.error("InceptionTime non disponibile: %s", e)
        log.error("Eseguire: pip install aeon[deep_learning] torch")
        sys.exit(1)

    clf = InceptionTimeClassifier(
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        random_state=RAND,
        verbose=True,
    )

    t0 = time.time()
    clf.fit(X_tr, y_train)
    t_train = time.time() - t0
    log.info("  Training completato in %.1f s (%.1f min)",
             t_train, t_train / 60)

    # ── predizione ────────────────────────────────────────────────────────────
    log.info("  Predizione sul test set (batch=%d)...", args.batch_size)
    y_pred  = clf.predict(X_te)
    try:
        y_proba = clf.predict_proba(X_te)[:, 1]
    except Exception:
        y_proba = None
        log.warning("  predict_proba non disponibile")

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

    ul      = ul1699b(y_test, y_pred)
    best_th = threshold_analysis(y_test, y_proba) if y_proba is not None else 0.5

    cm = confusion_matrix(y_test, y_pred)

    # ── grafici ───────────────────────────────────────────────────────────────
    plot_results(y_test, y_pred, y_proba, args.out)

    # ── salva modello ─────────────────────────────────────────────────────────
    model_path = os.path.join(args.out, "model_inceptiontime.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    log.info("  Modello salvato: %s", model_path)

    # ── export ONNX via tf2onnx ──────────────────────────────────────────────
    # InceptionTime usa Keras/TensorFlow internamente — NON PyTorch.
    # Dopo il fit il modello Keras è in clf.model_
    # Export: Keras → SavedModel → ONNX tramite tf2onnx
    log.info("")
    log.info("  --- Export ONNX (tf2onnx) ---")
    onnx_path = os.path.join(args.out, "inceptiontime.onnx")
    _export_inceptiontime_onnx(clf, X_tr, onnx_path)

    # ── risultati InceptionTime ───────────────────────────────────────────────
    it_result = {
        "model_name":              "InceptionTime",
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

    # Tabella riepilogo
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

    # Grafico confronto
    if mr_result:
        plot_comparison(mr_result, it_result, args.out)

    # ── report testuale ───────────────────────────────────────────────────────
    report_path = os.path.join(args.out, "inceptiontime_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("TRAINING REPORT — InceptionTime\n")
        f.write("Normativa di riferimento: UL 1699B\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset:    {args.dataset}\n")
        f.write(f"Epoche:     {args.epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n\n")
        for r in all_results:
            f.write(f"\n{'='*40}\n{r['model_name']}\n{'='*40}\n")
            for k, v in r.items():
                if k not in ("confusion_matrix", "model_name"):
                    f.write(f"  {k}: {v}\n")

    log.info("")
    log.info("Output in: %s", args.out)
    log.info("  model_inceptiontime.pkl")
    log.info("  inceptiontime.onnx            ← per X-CUBE-AI su STM32H7")
    log.info("  results_inceptiontime.png")
    if mr_result:
        log.info("  confronto_inception_vs_multirocket.png")
    log.info("  inceptiontime_report.txt")


if __name__ == "__main__":
    main()