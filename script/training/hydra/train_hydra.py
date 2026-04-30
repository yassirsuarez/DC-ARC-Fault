#!/usr/bin/env python3
"""
train_hydra.py
==============
Addestramento e valutazione del classificatore di archi elettrici in
impianti fotovoltaici DC, basato su HydraClassifier (aeon).

HydraClassifier usa solo il trasformatore Hydra (dizionario di pattern
con kernel convoluzionali raggruppati) + Ridge Classifier, senza la
componente MultiRocket. Rispetto a MultiRocketHydraClassifier:

  Vantaggi:
    - Feature output ridotto (~12.288 vs ~32.000) → RAM inferenza ~4x inferiore
    - Più adatto al deployment embedded (ESP32-S3, STM32H7)
    - Export ONNX diretto del solo modulo Hydra (PyTorch)
    - Tempo di training simile

  Svantaggi:
    - Accuratezza leggermente inferiore su benchmark UCR
    - Meno feature per archi deboli (drop < 5%)

Pipeline:
  1. Caricamento del dataset (arc_dataset.npz)
  2. GroupShuffleSplit per esperimento (evita data leakage Study001/Study002)
  3. Undersampling della classe maggioritaria (max 300 per classe)
  4. Addestramento HydraClassifier
  5. Predizione a batch per evitare errori di memoria
  6. Calcolo metriche: accuracy, F1, ROC-AUC, Average Precision
  7. Analisi multi-soglia per conformità UL1699B
  8. Salvataggio modello (.pkl), grafici e report testuale

Uso:
    python train_hydra.py <arc_dataset.npz> [--out <cartella>]
                         [--multirocket-report <training_report.txt>]

Esempio:
    python train_hydra.py arc_dataset.npz ^
        --out results_hydra ^
        --multirocket-report results/training_report.txt

Requisiti:
    pip install aeon torch scikit-learn matplotlib seaborn

Autori: progetto tesi magistrale — Manutenzione e Affidabilità
Normativa di riferimento: UL 1699B — Photovoltaic DC Arc-Fault Circuit Protection
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

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# ── configurazione logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ── parametri ─────────────────────────────────────────────────────────────────
FS_HZ      = 10_000   # Frequenza di campionamento [Hz]
TEST_SIZE  = 0.20     # Frazione test set [-]
BATCH_SIZE = 20       # Batch predict (gestione RAM)
RAND_STATE = 42       # Seed riproducibilità

UL_MIN_DETECTION_PCT = 95.0
UL_MAX_FP_PCT        = 5.0


# ══════════════════════════════════════════════════════════════════════════════
# 1. Gestione sbilanciamento classi
# ══════════════════════════════════════════════════════════════════════════════

def analyze_imbalance(y: np.ndarray) -> tuple:
    """
    Analizza lo sbilanciamento delle classi.
    Con ratio > 15 applica undersampling della classe maggioritaria.
    """
    n0    = int((y == 0).sum())
    n1    = int((y == 1).sum())
    tot   = len(y)
    ratio = max(n0, n1) / max(min(n0, n1), 1)

    log.info("=" * 60)
    log.info("ANALISI SBILANCIAMENTO CLASSI")
    log.info("=" * 60)
    log.info("  Totale file:           %d", tot)
    log.info("  label=0 (no arco):     %d  (%.1f%%)", n0, 100 * n0 / tot)
    log.info("  label=1 (arco):        %d  (%.1f%%)", n1, 100 * n1 / tot)
    log.info("  Ratio maggiore/minore: %.1f:1", ratio)
    log.info("  Strategia: undersampling bilanciato (max 300 per classe)")
    return "undersample", n0, n1


def undersample_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_per_class: int = 300,
) -> tuple:
    """
    Bilancia il train set tramite undersampling della classe maggioritaria.

    max_per_class limita i campioni per classe per evitare OOM durante
    il training di Hydra (convoluzione PyTorch su serie lunghe).
    Il test set non viene mai modificato.
    """
    n_min  = min((y_train == 0).sum(), (y_train == 1).sum(), max_per_class)
    rng    = np.random.default_rng(RAND_STATE)
    idx0   = rng.choice(np.where(y_train == 0)[0], size=n_min, replace=False)
    idx1   = rng.choice(np.where(y_train == 1)[0], size=n_min, replace=False)
    idx    = np.concatenate([idx0, idx1])
    rng.shuffle(idx)
    log.info("  Undersampling: %d → %d campioni (%d per classe, max=%d)",
             len(y_train), len(idx), n_min, max_per_class)
    return X_train[idx], y_train[idx]


# ══════════════════════════════════════════════════════════════════════════════
# 2. Metrica UL1699B
# ══════════════════════════════════════════════════════════════════════════════

def ul1699b_metric(y_test: np.ndarray, y_pred: np.ndarray) -> dict:
    """Verifica la conformità UL1699B: detection ≥ 95%, FP ≤ 5%."""
    arc    = y_test == 1
    no_arc = y_test == 0

    detected = int(((y_pred == 1) & arc).sum())
    missed   = int(((y_pred == 0) & arc).sum())
    fp       = int(((y_pred == 1) & no_arc).sum())
    tn       = int(((y_pred == 0) & no_arc).sum())

    det_rate = 100.0 * detected / max(int(arc.sum()), 1)
    fp_rate  = 100.0 * fp       / max(int(no_arc.sum()), 1)
    conforme = det_rate >= UL_MIN_DETECTION_PCT and fp_rate <= UL_MAX_FP_PCT

    log.info("")
    log.info("=" * 60)
    log.info("METRICA UL1699B (soglia decisione: 0.5)")
    log.info("=" * 60)
    log.info("  File con arco nel test:    %d", int(arc.sum()))
    log.info("  Archi rilevati:            %d  (%.1f%%)", detected, det_rate)
    log.info("  Archi mancati:             %d", missed)
    log.info("  File senza arco nel test:  %d", int(no_arc.sum()))
    log.info("  Falsi positivi:            %d  (%.1f%%)", fp, fp_rate)
    log.info("  Veri negativi:             %d", tn)
    log.info("")
    if conforme:
        log.info("  CONFORME UL1699B  "
                 "(detection %.1f%% >= %.0f%%, FP %.1f%% <= %.0f%%)",
                 det_rate, UL_MIN_DETECTION_PCT, fp_rate, UL_MAX_FP_PCT)
    else:
        log.warning("  NON conforme UL1699B")
        if det_rate < UL_MIN_DETECTION_PCT:
            log.warning("    Detection %.1f%% < %.0f%%",
                        det_rate, UL_MIN_DETECTION_PCT)
        if fp_rate > UL_MAX_FP_PCT:
            log.warning("    Falsi positivi %.1f%% > %.0f%%",
                        fp_rate, UL_MAX_FP_PCT)

    return {
        "detected":                detected,
        "missed":                  missed,
        "false_positives":         fp,
        "true_negatives":          tn,
        "detection_rate_pct":      round(det_rate, 2),
        "false_positive_rate_pct": round(fp_rate, 2),
        "ul1699b_conforme":        conforme,
    }


def threshold_analysis(y_test: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Analisi multi-soglia per trovare il punto di operazione ottimale UL1699B.
    Stampa detection rate e FP rate per soglie da 0.10 a 0.50.
    """
    log.info("")
    log.info("  --- Analisi soglie di decisione ---")
    log.info("  %s  %s  %s  %s",
             "Soglia".rjust(8), "Det%".rjust(7),
             "FP%".rjust(6), "UL1699B".rjust(9))
    log.info("  " + "-" * 38)

    best_threshold = 0.5
    best_det       = 0.0

    for thr in np.arange(0.10, 0.55, 0.05):
        yp     = (y_proba >= thr).astype(int)
        arc    = y_test == 1
        no_arc = y_test == 0
        det    = 100.0 * ((yp == 1) & arc).sum()    / max(int(arc.sum()), 1)
        fpr    = 100.0 * ((yp == 1) & no_arc).sum() / max(int(no_arc.sum()), 1)
        ok     = "SI" if det >= UL_MIN_DETECTION_PCT and fpr <= UL_MAX_FP_PCT else "NO"
        log.info("  %8.2f  %6.1f%%  %5.1f%%  %9s", thr, det, fpr, ok)

        if det >= UL_MIN_DETECTION_PCT and fpr <= UL_MAX_FP_PCT and det > best_det:
            best_det       = det
            best_threshold = float(thr)

    log.info("")
    log.info("  Soglia ottimale consigliata: %.2f", best_threshold)
    return best_threshold


# ══════════════════════════════════════════════════════════════════════════════
# 3. Grafici
# ══════════════════════════════════════════════════════════════════════════════

def plot_class_distribution(
    y_train: np.ndarray,
    y_test: np.ndarray,
    out_dir: str,
) -> None:
    """Salva la distribuzione delle classi in train e test set."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Distribuzione classi — train e test set (HydraClassifier)",
                 fontsize=12)
    for ax, y, title in [
        (axes[0], y_train, "Train set"),
        (axes[1], y_test,  "Test set"),
    ]:
        counts = [(y == 0).sum(), (y == 1).sum()]
        bars = ax.bar(
            ["No arco (0)", "Arco (1)"],
            counts,
            color=["steelblue", "tomato"],
            edgecolor="white",
        )
        for bar, c in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{c}\n({100 * c / len(y):.1f}%)",
                ha="center", va="bottom", fontsize=10,
            )
        ax.set_title(title)
        ax.set_ylabel("Numero di file")
        ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(out_dir, "class_distribution_hydra.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    log.info("  Salvato: %s", path)


def plot_results(
    y_test:  np.ndarray,
    y_pred:  np.ndarray,
    y_proba: np.ndarray | None,
    out_dir: str,
) -> None:
    """
    Pannello grafico risultati HydraClassifier:
    Confusion Matrix | ROC Curve | Precision-Recall | Score distribution.
    """
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle("Risultati — HydraClassifier", fontsize=13)

    # Confusion Matrix
    ax = axes[0]
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Purples", ax=ax,
        xticklabels=["No arco", "Arco"],
        yticklabels=["No arco", "Arco"],
    )
    ax.set_title("Confusion Matrix")
    ax.set_ylabel("Reale"); ax.set_xlabel("Predetto")

    # ROC Curve
    ax = axes[1]
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, color="mediumpurple", lw=2,
                label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.legend()
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(alpha=0.3)

    # Precision-Recall
    ax = axes[2]
    if y_proba is not None:
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        ax.plot(rec, prec, color="mediumpurple", lw=2, label=f"AP = {ap:.3f}")
        ax.axhline(
            y_test.mean(), color="gray", ls="--", lw=1,
            label=f"Baseline = {y_test.mean():.2f}",
        )
        ax.legend()
    ax.set_title("Precision-Recall")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.grid(alpha=0.3)

    # Distribuzione score
    ax = axes[3]
    if y_proba is not None:
        ax.hist(y_proba[y_test == 0], bins=30, alpha=0.6,
                color="steelblue", label="No arco (0)")
        ax.hist(y_proba[y_test == 1], bins=30, alpha=0.6,
                color="tomato",    label="Arco (1)")
        ax.axvline(0.5, color="black", ls="--", lw=1, label="soglia = 0.5")
        ax.legend()
    ax.set_title("Distribuzione score")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "results_hydra.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    log.info("  Salvato: %s", path)


def plot_series_examples(
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    out_dir: str,
    n_per_class: int = 2,
) -> None:
    """Esempi di serie temporali per categoria: TP, FN, TN, FP."""
    categories = [
        ("Vero Positivo\n(arco rilevato)",   y_test == 1, y_pred == 1),
        ("Falso Negativo\n(arco mancato)",    y_test == 1, y_pred == 0),
        ("Vero Negativo\n(no arco corretto)", y_test == 0, y_pred == 0),
        ("Falso Positivo\n(falso allarme)",   y_test == 0, y_pred == 1),
    ]
    n_cols = len(categories)
    fig, axes = plt.subplots(n_per_class, n_cols,
                             figsize=(n_cols * 4, n_per_class * 3))
    fig.suptitle(
        "Esempi serie temporali — HydraClassifier — I(t)/I_nom", fontsize=12
    )
    t = np.arange(X_test.shape[1]) / FS_HZ

    for col, (title, mask_real, mask_pred) in enumerate(categories):
        idx = np.where(mask_real & mask_pred)[0]
        for row in range(n_per_class):
            ax = axes[row, col] if n_per_class > 1 else axes[col]
            if row < len(idx):
                ix    = idx[row]
                color = "tomato" if y_test[ix] == 1 else "steelblue"
                ax.plot(t, X_test[ix], lw=0.7, color=color)
                ax.axhline(1.0, color="gray", ls=":", lw=0.8)
                ax.set_ylim(-0.1, 1.3)
                ax.set_xlabel("Tempo [s]")
                ax.set_ylabel("I / I_nom [-]")
                ax.grid(alpha=0.3)
                if row == 0:
                    ax.set_title(title, fontsize=9)
            else:
                ax.text(0.5, 0.5, "Nessun esempio",
                        ha="center", va="center",
                        transform=ax.transAxes, color="gray")
                ax.axis("off")
                if row == 0:
                    ax.set_title(title, fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, "series_examples_hydra.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    log.info("  Salvato: %s", path)


def plot_comparison(
    mr_report_path: str | None,
    hydra_result:   dict,
    out_dir:        str,
) -> None:
    """
    Grafico di confronto HydraClassifier vs MultiRocketHydra.
    Carica le metriche di MultiRocketHydra dal training_report.txt.
    """
    if not mr_report_path or not os.path.isfile(mr_report_path):
        return

    # Leggi report MultiRocketHydra
    mr = {}
    with open(mr_report_path, "r", encoding="utf-8") as f:
        for line in f:
            if ": " in line:
                k, v = line.strip().split(": ", 1)
                try:    mr[k.strip()] = float(v.strip())
                except: mr[k.strip()] = v.strip()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Confronto — MultiRocketHydra vs HydraClassifier", fontsize=13)

    # Barchart metriche
    ax = axes[0]
    names  = ["Bal.Acc", "F1", "AUC", "Det%/100", "1-FP%"]
    x      = np.arange(len(names))
    w      = 0.35
    mr_vals = [
        mr.get("balanced_accuracy", 0),
        mr.get("f1_arc", 0),
        mr.get("roc_auc", 0) or 0,
        mr.get("detection_rate_pct", 0) / 100,
        1 - (mr.get("false_positive_rate_pct", 0) or 0) / 100,
    ]
    hy_vals = [
        hydra_result["balanced_accuracy"],
        hydra_result["f1_arc"],
        hydra_result.get("roc_auc", 0) or 0,
        hydra_result["detection_rate_pct"] / 100,
        1 - hydra_result["false_positive_rate_pct"] / 100,
    ]
    ax.bar(x - w/2, mr_vals, w, label="MultiRocketHydra",
           color="steelblue", alpha=0.85)
    ax.bar(x + w/2, hy_vals, w, label="HydraClassifier",
           color="mediumpurple", alpha=0.85)
    ax.axhline(0.95, color="red", ls="--", lw=1, label="95% UL1699B")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0.8, 1.02); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_title("Confronto metriche")

    # Tabella riepilogo
    ax = axes[1]
    ax.axis("off")
    cols = ["Modello", "BA", "F1", "AUC", "Det%", "FP%", "T(s)", "UL1699B"]
    rows = [
        [
            "MultiRocketHydra",
            f"{mr.get('balanced_accuracy',0):.3f}",
            f"{mr.get('f1_arc',0):.3f}",
            f"{mr.get('roc_auc',0):.3f}" if mr.get("roc_auc") else "N/A",
            f"{mr.get('detection_rate_pct',0):.1f}%",
            f"{mr.get('false_positive_rate_pct',0):.1f}%",
            f"{mr.get('train_time_s',0):.0f}s",
            "✓" if mr.get("ul1699b_conforme") else "✗",
        ],
        [
            "HydraClassifier",
            f"{hydra_result['balanced_accuracy']:.3f}",
            f"{hydra_result['f1_arc']:.3f}",
            f"{hydra_result['roc_auc']:.3f}" if hydra_result.get("roc_auc") else "N/A",
            f"{hydra_result['detection_rate_pct']:.1f}%",
            f"{hydra_result['false_positive_rate_pct']:.1f}%",
            f"{hydra_result['train_time_s']:.0f}s",
            "✓" if hydra_result["ul1699b_conforme"] else "✗",
        ],
    ]
    tbl = ax.table(cellText=rows, colLabels=cols,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    tbl.scale(1.2, 2.0)
    for i, r in enumerate(rows):
        ok = r[-1] == "✓"
        for j in range(len(cols)):
            tbl[i + 1, j].set_facecolor("#d4edda" if ok else "#f8d7da")
    ax.set_title("Riepilogo")

    plt.tight_layout()
    path = os.path.join(out_dir, "confronto_hydra_vs_multirocket.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    log.info("  Salvato: %s", path)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Training e valutazione
# ══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_train: np.ndarray,
    y_test:  np.ndarray,
    out_dir: str,
) -> dict:
    """
    Addestra HydraClassifier e ne valuta le prestazioni sul test set.

    La predizione viene eseguita a batch per evitare OOM su serie lunghe.
    """
    log.info("")
    log.info("=" * 60)
    log.info("TRAINING: HydraClassifier")
    log.info("=" * 60)
    log.info("  Campioni train: %d", len(y_train))
    log.info("  Campioni test:  %d", len(y_test))

    try:
        from aeon.classification.convolution_based import HydraClassifier
    except ImportError as exc:
        log.error("Impossibile importare HydraClassifier: %s", exc)
        log.error("Eseguire: pip install aeon torch")
        sys.exit(1)

    # aeon richiede shape (n_samples, n_channels, n_timepoints)
    X_tr = X_train[:, np.newaxis, :]
    X_te = X_test[:,  np.newaxis, :]

    model = HydraClassifier(
        n_jobs=-1,
        random_state=RAND_STATE,
    )

    t0 = time.time()
    model.fit(X_tr, y_train)
    t_train = time.time() - t0
    log.info("  Training completato in %.1f s (%.1f min)",
             t_train, t_train / 60)

    # Predizione a batch
    log.info("  Predizione sul test set (batch_size=%d)...", BATCH_SIZE)
    y_pred_list, y_proba_list = [], []
    for start in range(0, len(X_te), BATCH_SIZE):
        batch = X_te[start:start + BATCH_SIZE]
        y_pred_list.append(model.predict(batch))
        try:
            y_proba_list.append(model.predict_proba(batch)[:, 1])
        except Exception:
            pass
    y_pred  = np.concatenate(y_pred_list)
    y_proba = np.concatenate(y_proba_list) if y_proba_list else None

    # Metriche standard
    log.info("")
    log.info("  --- Metriche standard ---")
    report = classification_report(
        y_test, y_pred,
        target_names=["No arco", "Arco"],
        digits=3,
    )
    for line in report.splitlines():
        log.info("  %s", line)

    ba  = balanced_accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)           if y_proba is not None else None
    ap  = average_precision_score(y_test, y_proba) if y_proba is not None else None

    log.info("  Balanced Accuracy: %.4f", ba)
    log.info("  F1 (arco):         %.4f", f1)
    if auc is not None: log.info("  ROC-AUC:           %.4f", auc)
    if ap  is not None: log.info("  Avg Precision:     %.4f", ap)

    # UL1699B
    ul = ul1699b_metric(y_test, y_pred)

    best_thr = 0.5
    if y_proba is not None:
        best_thr = threshold_analysis(y_test, y_proba)
        ul["best_threshold"] = best_thr

    # Grafici
    plot_results(y_test, y_pred, y_proba, out_dir)
    plot_series_examples(X_test, y_test, y_pred, out_dir)

    # Salva modello
    model_path = os.path.join(out_dir, "model_hydra.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    log.info("  Modello salvato: %s", model_path)

    return {
        "model_name":              "HydraClassifier",
        "train_time_s":            round(t_train, 1),
        "n_train":                 len(y_train),
        "n_test":                  len(y_test),
        "balanced_accuracy":       round(ba,  4),
        "f1_arc":                  round(f1,  4),
        "roc_auc":                 round(auc, 4) if auc else None,
        "avg_precision":           round(ap,  4) if ap  else None,
        "best_threshold":          round(best_thr, 2),
        **ul,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Training HydraClassifier per rilevamento archi elettrici PV.\n"
            "Versione alleggerita di MultiRocketHydra — stessa Hydra, senza ROCKET.\n"
            "Produce un confronto diretto con MultiRocketHydra se viene fornito\n"
            "il report del training precedente."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        help="Percorso al file arc_dataset.npz",
    )
    parser.add_argument(
        "--out", "-o",
        default="./results_hydra",
        help="Cartella di output (default: ./results_hydra)",
    )
    parser.add_argument(
        "--multirocket-report",
        default=None,
        help="Path al training_report.txt di MultiRocketHydra "
             "per il confronto diretto",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help=f"Frazione test set (default: {TEST_SIZE})",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.dataset):
        log.error("File non trovato: %s", args.dataset)
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)

    # Caricamento dataset
    log.info("Caricamento dataset: %s", args.dataset)
    data = np.load(args.dataset)
    X    = data["X"]
    y    = data["y"]
    log.info("  X shape: %s  (%.1f s per serie a %d Hz)",
             X.shape, X.shape[1] / FS_HZ, FS_HZ)
    log.info("  y shape: %s", y.shape)

    # Analisi sbilanciamento
    strategy, n0, n1 = analyze_imbalance(y)

    # Split per gruppo sperimentale (GroupShuffleSplit)
    meta_path = args.dataset.replace("arc_dataset.npz", "arc_dataset_meta.csv")
    groups    = None
    if os.path.isfile(meta_path):
        import pandas as pd
        meta = pd.read_csv(meta_path)
        def _exp_key(fn):
            s   = fn.replace("_Raw Data.mat", "").replace(" Data.mat", "")
            idx = s.lower().rfind("_study")
            return s[:idx] if idx > 0 else s
        meta["exp_key"] = meta["filename"].apply(_exp_key)
        unique_keys     = {k: i for i, k in enumerate(meta["exp_key"].unique())}
        groups          = meta["exp_key"].map(unique_keys).values
        log.info("  Metadati caricati: %d gruppi sperimentali", len(unique_keys))
        log.info("  (Study001 e Study002 stesso esperimento → stesso split)")

    if groups is not None:
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(
            n_splits=1, test_size=args.test_size, random_state=RAND_STATE
        )
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        log.info("")
        log.info("Split per gruppo sperimentale (%.0f%%/%.0f%%):",
                 (1 - args.test_size) * 100, args.test_size * 100)
    else:
        log.warning("Metadati non trovati — uso split casuale (rischio data leakage)")
        train_idx, test_idx = train_test_split(
            np.arange(len(y)), test_size=args.test_size,
            random_state=RAND_STATE, stratify=y,
        )

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    log.info("  Train: %d  (arco=%d, no=%d)",
             len(y_train), int((y_train==1).sum()), int((y_train==0).sum()))
    log.info("  Test:  %d  (arco=%d, no=%d)",
             len(y_test), int((y_test==1).sum()), int((y_test==0).sum()))

    plot_class_distribution(y_train, y_test, args.out)

    # Undersampling sul solo train set
    X_train, y_train = undersample_train(X_train, y_train, max_per_class=300)

    # Training
    result = train_and_evaluate(X_train, X_test, y_train, y_test, args.out)

    # Confronto con MultiRocketHydra
    if args.multirocket_report:
        plot_comparison(args.multirocket_report, result, args.out)

    # Riepilogo finale
    log.info("")
    log.info("=" * 72)
    log.info("RIEPILOGO FINALE — HydraClassifier")
    log.info("=" * 72)
    log.info("  %-22s %8s %7s %7s %7s %6s %7s %6s %9s",
             "Modello", "Bal.Acc", "F1", "AUC",
             "Det%", "FP%", "Soglia", "T(s)", "UL1699B")
    log.info("  " + "-" * 72)
    ok = "CONFORME" if result["ul1699b_conforme"] else "NO"
    log.info(
        "  %-22s %8.4f %7.4f %7s %6.1f%% %5.1f%% %7.2f %6.1f %9s",
        result["model_name"],
        result["balanced_accuracy"],
        result["f1_arc"],
        f"{result['roc_auc']:.4f}" if result["roc_auc"] else "N/A",
        result["detection_rate_pct"],
        result["false_positive_rate_pct"],
        result["best_threshold"],
        result["train_time_s"],
        ok,
    )

    # Report testuale
    report_path = os.path.join(args.out, "training_report_hydra.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("TRAINING REPORT — HydraClassifier\n")
        f.write("Normativa di riferimento: UL 1699B\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset:        {args.dataset}\n")
        f.write(f"File totali:    {len(y)}\n")
        f.write(f"label=1 arco:   {n1}\n")
        f.write(f"label=0 no:     {n0}\n")
        f.write(f"Train set:      {result['n_train']}\n")
        f.write(f"Test set:       {result['n_test']}\n\n")
        f.write(f"\n{'='*40}\n{result['model_name']}\n{'='*40}\n")
        for k, v in result.items():
            if k != "model_name":
                f.write(f"  {k}: {v}\n")

    log.info("")
    log.info("Report salvato: %s", report_path)
    log.info("Output in: %s", args.out)
    log.info("  model_hydra.pkl")
    log.info("  results_hydra.png")
    log.info("  series_examples_hydra.png")
    log.info("  class_distribution_hydra.png")
    if args.multirocket_report:
        log.info("  confronto_hydra_vs_multirocket.png")
    log.info("  training_report_hydra.txt")


if __name__ == "__main__":
    main()
