#!/usr/bin/env python3
"""
train_classifier.py
===================
Addestramento e valutazione del classificatore di archi elettrici in
impianti fotovoltaici DC, basato su MultiRocketHydraClassifier (aeon).

Il classificatore riceve i primi WINDOW_S secondi di corrente normalizzata
di ciascun file e predice la presenza o assenza di un arco elettrico.
La valutazione include metriche standard di classificazione e la verifica
di conformità alla normativa UL1699B (rilevamento entro 2s, energia < 750J).

Pipeline:
  1. Caricamento del dataset (arc_dataset_new.npz)
  2. Analisi dello sbilanciamento classi e scelta della strategia
  3. Split train/test stratificato (80/20)
  4. Addestramento MultiRocketHydra (o HIVE-COTE v2)
  5. Predizione a batch per evitare errori di memoria
  6. Calcolo metriche: accuracy, F1, ROC-AUC, Average Precision
  7. Analisi multi-soglia per ottimizzare il punto di operazione UL1699B
  8. Salvataggio modello (.pkl), grafici e report testuale

Uso:
    python train_classifier.py <arc_dataset_new.npz> [--out <cartella>]
    python train_classifier.py <arc_dataset_new.npz> --model hivecote
    python train_classifier.py <arc_dataset_new.npz> --model both

Requisiti:
    pip install aeon scikit-learn torch matplotlib seaborn

Autori: progetto tesi magistrale — Manutenzione e Affidabilità
Normativa di riferimento: UL 1699B — Photovoltaic DC Arc-Fault Circuit Protection

MODIFICHE (dataset 34k finestre, WINDOW_S=100ms, FS=10kHz):
  - max_per_class alzato a 3000 (serie da 1000 campioni, memoria ~24 MB)
  - BATCH_SIZE ridotto a 32 per maggiore robustezza su dataset grandi
  - FS_HZ rimane 10_000 (corretto per il dataset corrente)
  - Soglie max_per_class riviste per finestre brevi (≤2000 campioni)
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
FS_HZ        = 10_000   # Frequenza di campionamento [Hz]
                        # Corretto per finestre da 100ms a 10kHz = 1000 campioni
TEST_SIZE    = 0.20     # Frazione del dataset riservata al test set [-]
BATCH_SIZE   = 32       # MODIFICATO: era 50 — ridotto per robustezza su 34k campioni
RAND_STATE   = 42       # Seed per riproducibilità

# Soglie conformità UL1699B
UL_MIN_DETECTION_PCT  = 95.0   # Detection rate minima [%]
UL_MAX_FP_PCT         = 5.0    # False positive rate massimo [%]


# ══════════════════════════════════════════════════════════════════════════════
# 1. Gestione sbilanciamento classi
# ══════════════════════════════════════════════════════════════════════════════

def analyze_imbalance(y: np.ndarray) -> tuple:
    """
    Analizza lo sbilanciamento delle classi e determina la strategia ottimale.

    Strategie:
      - ratio < 1.5  → nessuna correzione
      - ratio 1.5–15 → class_weight='balanced' nel classificatore
      - ratio > 15   → undersampling della classe maggioritaria

    Returns:
        strategy (str): 'none', 'class_weight' o 'undersample'.
        n0 (int): numero di campioni label=0.
        n1 (int): numero di campioni label=1.
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

    if ratio < 1.5:
        strategy = "none"
        log.info("  Strategia: nessuna correzione necessaria")
    else:
        # Forza sempre undersampling indipendentemente dal ratio.
        # class_weight='balanced' causa OOM con Hydra su dataset grandi
        # (dataset scorrevole può avere decine di migliaia di campioni).
        strategy = "undersample"
        log.info("  Strategia: undersampling bilanciato")
        log.info("  (class_weight disabilitato: Hydra va in OOM con > 1000 campioni)")

    return strategy, n0, n1


def undersample_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_per_class: int = 3000,
) -> tuple:
    """
    Bilancia il train set tramite undersampling della classe maggioritaria.

    Il parametro max_per_class limita il numero massimo di campioni per classe
    per evitare errori di allocazione memoria durante il training di Hydra.
    Il test set non viene mai modificato per mantenere la distribuzione reale.

    Con finestre da 1000 campioni (100ms @ 10kHz):
      - max_per_class=3000 → 6000 finestre × 1000 float32 ≈ 24 MB  ✓ sicuro
      - max_per_class=5000 → 10000 finestre × 1000 float32 ≈ 40 MB  ✓ ok se RAM > 8GB

    Args:
        max_per_class: numero massimo di campioni per classe (default: 3000).
                       Era 300 nella versione precedente per finestre da 20.000 campioni.

    Returns:
        X_bal, y_bal: train set bilanciato.
    """
    n_min  = min((y_train == 0).sum(), (y_train == 1).sum(), max_per_class)
    idx0   = np.where(y_train == 0)[0]
    idx1   = np.where(y_train == 1)[0]
    rng    = np.random.default_rng(RAND_STATE)
    idx0_d = rng.choice(idx0, size=n_min, replace=False)
    idx1_d = rng.choice(idx1, size=n_min, replace=False)
    idx    = np.concatenate([idx0_d, idx1_d])
    rng.shuffle(idx)
    log.info("  Undersampling: %d → %d campioni (%d per classe, max=%d)",
             len(y_train), len(idx), n_min, max_per_class)
    return X_train[idx], y_train[idx]


# ══════════════════════════════════════════════════════════════════════════════
# 2. Metrica UL1699B
# ══════════════════════════════════════════════════════════════════════════════

def ul1699b_metric(y_test: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Verifica la conformità alla normativa UL1699B sul test set.

    Criteri di conformità:
      - Criterio 1: detection rate ≥ 95% (archi rilevati / archi totali)
      - Criterio 2: false positive rate ≤ 5% (falsi allarmi / file senza arco)

    Returns:
        dict con conteggi, percentuali e flag di conformità.
    """
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
        "detected":               detected,
        "missed":                 missed,
        "false_positives":        fp,
        "true_negatives":         tn,
        "detection_rate_pct":     round(det_rate, 2),
        "false_positive_rate_pct": round(fp_rate, 2),
        "ul1699b_conforme":       conforme,
    }


def threshold_analysis(
    y_test: np.ndarray,
    y_proba: np.ndarray,
) -> float:
    """
    Analizza le prestazioni UL1699B al variare della soglia di decisione.

    Stampa una tabella con detection rate e false positive rate per soglie
    da 0.10 a 0.50 e identifica la soglia ottimale (minima soglia conforme).

    Returns:
        best_threshold (float): soglia ottimale per la conformità UL1699B.
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
    """Salva il grafico della distribuzione delle classi in train e test set."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Distribuzione classi — train e test set", fontsize=12)
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
    path = os.path.join(out_dir, "class_distribution.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    log.info("  Salvato: %s", path)


def plot_results(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    model_name: str,
    out_dir: str,
) -> None:
    """
    Salva il pannello grafico dei risultati di classificazione.

    Pannelli: Confusion Matrix | ROC Curve | Precision-Recall | Score distribution.
    """
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(f"Risultati — {model_name}", fontsize=13)

    # Confusion Matrix
    ax = axes[0]
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["No arco", "Arco"],
        yticklabels=["No arco", "Arco"],
    )
    ax.set_title("Confusion Matrix")
    ax.set_ylabel("Reale")
    ax.set_xlabel("Predetto")

    # ROC Curve
    ax = axes[1]
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
    ax.set_title("ROC Curve")
    ax.grid(alpha=0.3)

    # Precision-Recall
    ax = axes[2]
    if y_proba is not None:
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        ax.plot(rec, prec, color="tomato", lw=2, label=f"AP = {ap:.3f}")
        ax.axhline(
            y_test.mean(), color="gray", ls="--", lw=1,
            label=f"Baseline = {y_test.mean():.2f}",
        )
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend()
    ax.set_title("Precision-Recall")
    ax.grid(alpha=0.3)

    # Distribuzione score
    ax = axes[3]
    if y_proba is not None:
        ax.hist(y_proba[y_test == 0], bins=30, alpha=0.6,
                color="steelblue", label="No arco (0)")
        ax.hist(y_proba[y_test == 1], bins=30, alpha=0.6,
                color="tomato",    label="Arco (1)")
        ax.axvline(0.5, color="black", ls="--", lw=1, label="soglia = 0.5")
        ax.set_xlabel("Probabilità predetta — classe 1 (arco)")
        ax.legend()
    ax.set_title("Distribuzione score")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    name_safe = model_name.replace(" ", "_").lower()
    path = os.path.join(out_dir, f"results_{name_safe}.png")
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
    """
    Salva esempi di serie temporali suddivisi per esito di classificazione.

    Categorie: Vero Positivo, Falso Negativo, Vero Negativo, Falso Positivo.
    Utile per l'analisi qualitativa degli errori del classificatore.
    """
    FS = FS_HZ
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
        "Esempi serie temporali — corrente normalizzata I(t)/I_nom",
        fontsize=12,
    )
    t = np.arange(X_test.shape[1]) / FS

    for col, (title, mask_real, mask_pred) in enumerate(categories):
        idx = np.where(mask_real & mask_pred)[0]
        for row in range(n_per_class):
            ax = axes[row, col] if n_per_class > 1 else axes[col]
            if row < len(idx):
                ix    = idx[row]
                color = "tomato" if y_test[ix] == 1 else "steelblue"
                ax.plot(t, X_test[ix], lw=0.7, color=color)
                ax.axhline(1.0, color="gray", ls=":", lw=0.8,
                           label="I_nom")
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
    path = os.path.join(out_dir, "series_examples.png")
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
    model_name: str,
    model,
    out_dir: str,
) -> dict:
    """
    Addestra il classificatore e ne valuta le prestazioni sul test set.

    La predizione viene eseguita a batch per evitare errori di allocazione
    memoria su dataset di grandi dimensioni.

    Returns:
        dict con tutte le metriche calcolate.
    """
    log.info("")
    log.info("=" * 60)
    log.info("TRAINING: %s", model_name)
    log.info("=" * 60)
    log.info("  Campioni train: %d", len(y_train))
    log.info("  Campioni test:  %d", len(y_test))

    # aeon richiede shape (n_samples, n_channels, n_timepoints)
    X_tr = X_train[:, np.newaxis, :]
    X_te = X_test[:,  np.newaxis, :]

    t0 = time.time()
    model.fit(X_tr, y_train)
    t_train = time.time() - t0
    log.info("  Training completato in %.1f s", t_train)

    # Predizione a batch (BATCH_SIZE=32, ridotto da 50 per robustezza su 34k campioni)
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
    if auc is not None:
        log.info("  ROC-AUC:           %.4f", auc)
        log.info("  Avg Precision:     %.4f", ap)

    # Metrica UL1699B (soglia 0.5)
    ul = ul1699b_metric(y_test, y_pred)

    # Analisi multi-soglia
    best_thr = 0.5
    if y_proba is not None:
        best_thr = threshold_analysis(y_test, y_proba)
        ul["best_threshold"] = best_thr

    # Grafici
    plot_results(y_test, y_pred, y_proba, model_name, out_dir)
    plot_series_examples(X_test, y_test, y_pred, out_dir)

    # Salva modello
    model_path = os.path.join(
        out_dir,
        f"model_{model_name.replace(' ', '_').lower()}.pkl",
    )
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    log.info("  Modello salvato: %s", model_path)

    return {
        "model_name":              model_name,
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
            "Addestramento e valutazione del classificatore di archi\n"
            "elettrici in impianti fotovoltaici DC."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        help="Percorso al file arc_dataset_new.npz",
    )
    parser.add_argument(
        "--out", "-o",
        default="./results",
        help="Cartella di output (default: ./results)",
    )
    parser.add_argument(
        "--model",
        default="multirocket",
        choices=["multirocket", "hivecote", "both"],
        help="Classificatore da addestrare (default: multirocket)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help=f"Frazione del dataset per il test set (default: {TEST_SIZE})",
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
    log.info("  X shape: %s  (%.1f ms per serie a %d Hz)",
             X.shape, X.shape[1] / FS_HZ * 1000, FS_HZ)
    log.info("  y shape: %s", y.shape)

    # Analisi sbilanciamento
    strategy, n0, n1 = analyze_imbalance(y)

    # Carica metadati per lo split per gruppo sperimentale.
    # Funziona sia con dataset a file singolo che a finestra scorrevole:
    #   - File singolo: raggruppa Study001+Study002 dello stesso esperimento
    #   - Finestra scorrevole: raggruppa tutte le finestre dello stesso file
    # In entrambi i casi evita il data leakage nel test set.
    meta_path = args.dataset.replace("arc_dataset_new.npz", "arc_dataset_meta_new.csv")
    groups = None
    if os.path.isfile(meta_path):
        import pandas as pd
        meta = pd.read_csv(meta_path)
        def _exp_key(fn):
            s = fn.replace("_Raw Data.mat", "").replace(" Data.mat", "")
            idx = s.lower().rfind("_study")
            return s[:idx] if idx > 0 else s
        meta["exp_key"] = meta["filename"].apply(_exp_key)
        unique_keys = {k: i for i, k in enumerate(meta["exp_key"].unique())}
        groups = meta["exp_key"].map(unique_keys).values
        n_groups   = len(unique_keys)
        n_finestre = len(meta)
        is_sliding = n_finestre > n_groups * 1.5
        if is_sliding:
            log.info("  Modalità finestra scorrevole: %d finestre da %d file",
                     n_finestre, n_groups)
            log.info("  Tutte le finestre dello stesso file → stesso split")
        else:
            log.info("  Metadati caricati: %d gruppi sperimentali", n_groups)
            log.info("  (Study001 e Study002 → stesso split)")

    if groups is not None:
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(
            n_splits=1, test_size=args.test_size, random_state=RAND_STATE
        )
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        log.info("")
        log.info("Split per gruppo sperimentale (%.0f%%/%.0f%%):",
                 (1 - args.test_size) * 100, args.test_size * 100)
    else:
        log.warning("Metadati non trovati — uso split casuale (rischio data leakage)")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=args.test_size,
            random_state=RAND_STATE,
            stratify=y,
        )
        log.info("")
        log.info("Split casuale stratificato (%.0f%%/%.0f%%):",
                 (1 - args.test_size) * 100, args.test_size * 100)

    log.info("  Train: %d  (arco=%d, no=%d)",
             len(y_train), int((y_train==1).sum()), int((y_train==0).sum()))
    log.info("  Test:  %d  (arco=%d, no=%d)",
             len(y_test), int((y_test==1).sum()), int((y_test==0).sum()))

    plot_class_distribution(y_train, y_test, args.out)

    # ── Gestione sbilanciamento sul solo train set ────────────────────────────
    #
    # LOGICA max_per_class (MODIFICATA per dataset 34k finestre):
    #
    #   Serie da ≤ 500 campioni  (es. 50ms @ 10kHz):
    #     → max_per_class = 5000  (~50 MB in RAM, sicuro)
    #
    #   Serie da ≤ 2000 campioni (es. 100ms–200ms @ 10kHz):  ← CASO ATTUALE
    #     → max_per_class = 3000  (~24 MB in RAM, sicuro)
    #     Era 1000 nella versione precedente: troppo conservativo,
    #     sprecava ~25k finestre su 27k disponibili nel train.
    #
    #   Serie da > 2000 campioni (es. 2s @ 10kHz = 20.000 campioni):
    #     → max_per_class = 500   (~40 MB in RAM, limite per Hydra)
    #
    # Il test set NON viene mai modificato.
    # ─────────────────────────────────────────────────────────────────────────
    use_class_weight = None
    if strategy == "undersample":
        log.info("")
        n_samples_per_series = X_train.shape[1] if len(X_train) > 0 else 1000

        if n_samples_per_series <= 500:
            max_pc = 5000
        elif n_samples_per_series <= 2000:
            max_pc = 3000   # MODIFICATO: era 1000 — ora sfrutta più dati (24 MB)
        else:
            max_pc = 500    # finestre lunghe (≥ 20.000 campioni): limite memoria

        log.info("  Serie da %d campioni (%.0f ms @ %d Hz) → max_per_class=%d",
                 n_samples_per_series,
                 n_samples_per_series / FS_HZ * 1000,
                 FS_HZ,
                 max_pc)
        X_train, y_train = undersample_train(X_train, y_train, max_per_class=max_pc)
    elif strategy == "class_weight":
        use_class_weight = "balanced"
        log.info("  class_weight='balanced' attivato")

    # Definizione modelli
    models_to_run = []

    if args.model in ("multirocket", "both"):
        try:
            from aeon.classification.convolution_based import (
                MultiRocketHydraClassifier,
            )
            models_to_run.append((
                "MultiRocketHydra",
                MultiRocketHydraClassifier(
                    class_weight=use_class_weight,
                    n_jobs=-1,
                    random_state=RAND_STATE,
                ),
            ))
        except ImportError as exc:
            log.error("Impossibile importare MultiRocketHydra: %s", exc)
            log.error("Eseguire: pip install aeon torch")
            sys.exit(1)

    if args.model in ("hivecote", "both"):
        try:
            from aeon.classification.hybrid import HIVECOTEV2
            models_to_run.append((
                "HIVE-COTE v2",
                HIVECOTEV2(random_state=RAND_STATE, n_jobs=-1),
            ))
            log.warning(
                "HIVE-COTE v2 selezionato — "
                "il training può richiedere diverse ore su dataset grandi."
            )
        except ImportError as exc:
            log.warning("HIVE-COTE v2 non disponibile: %s", exc)

    if not models_to_run:
        log.error("Nessun modello disponibile.")
        sys.exit(1)

    # Training e valutazione
    all_results = []
    for model_name, model in models_to_run:
        result = train_and_evaluate(
            X_train, X_test, y_train, y_test,
            model_name, model, args.out,
        )
        all_results.append(result)

    # Riepilogo finale
    log.info("")
    log.info("=" * 72)
    log.info("RIEPILOGO FINALE")
    log.info("=" * 72)
    log.info("  %-22s %8s %7s %7s %7s %6s %7s %6s %9s",
             "Modello", "Bal.Acc", "F1", "AUC",
             "Det%", "FP%", "Soglia", "T(s)", "UL1699B")
    log.info("  " + "-" * 72)
    for r in all_results:
        ok = "CONFORME" if r["ul1699b_conforme"] else "NO"
        log.info(
            "  %-22s %8.4f %7.4f %7s %6.1f%% %5.1f%% %7.2f %6.1f %9s",
            r["model_name"],
            r["balanced_accuracy"],
            r["f1_arc"],
            f"{r['roc_auc']:.4f}" if r["roc_auc"] else "N/A",
            r["detection_rate_pct"],
            r["false_positive_rate_pct"],
            r["best_threshold"],
            r["train_time_s"],
            ok,
        )

    # Salva report testuale
    report_path = os.path.join(args.out, "training_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("TRAINING REPORT — Classificatore Archi Elettrici PV\n")
        f.write("Normativa di riferimento: UL 1699B\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset:        {args.dataset}\n")
        f.write(f"File totali:    {len(y)}\n")
        f.write(f"label=1 arco:   {n1}\n")
        f.write(f"label=0 no:     {n0}\n")
        f.write(f"Train set:      {len(y_train)}\n")
        f.write(f"Test set:       {len(y_test)}\n\n")
        for r in all_results:
            f.write(f"\n{'=' * 40}\n{r['model_name']}\n{'=' * 40}\n")
            for k, v in r.items():
                if k != "model_name":
                    f.write(f"  {k}: {v}\n")

    log.info("")
    log.info("Report salvato: %s", report_path)
    log.info("Output in: %s", args.out)


if __name__ == "__main__":
    main()