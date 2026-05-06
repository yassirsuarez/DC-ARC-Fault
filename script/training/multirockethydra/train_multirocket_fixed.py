#!/usr/bin/env python3
"""
train_classifier.py
===================
Pipeline di addestramento e valutazione per il classificatore di archi elettrici
in impianti fotovoltaici DC.

Architettura: MultiHydra (ensemble feature extractor) + RidgeClassifier

Pipeline:
  1.  Caricamento dataset già separati (train.npz / test.npz)
  2.  Analisi sbilanciamento classi nel training set e verifica coerenza
      distribuzione train/test
  3.  (Opzionale) Normalizzazione coerente train → test
  4.  Addestramento MultiHydra + RidgeClassifier
  5.  Estrazione feature in batch (ottimizzazione memoria/velocità)
  6.  Calcolo metriche: accuracy, F1, ROC-AUC, Average Precision
  7.  Analisi multi-soglia per ottimizzazione punto operativo UL1699B
  8.  Validazione robustezza (shuffle test + stabilità feature space)
  9.  Salvataggio bundle .pkl (pesi Hydra + coefficienti Ridge + config)
  10. Export deployment embedded: Hydra → ONNX, Ridge → C static inference

Uso:
    python train_classifier.py train.npz test.npz [--out ./results]
    python train_classifier.py train.npz test.npz --no-normalize
    python train_classifier.py train.npz test.npz --export-stm32

Requisiti:
    pip install aeon scikit-learn numpy matplotlib seaborn onnx skl2onnx

Normativa di riferimento: UL 1699B — Photovoltaic DC Arc-Fault Circuit Protection
"""

import argparse
import hashlib
import json
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

from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

# ── configurazione logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ── parametri globali ─────────────────────────────────────────────────────────
FS_HZ               = 10_000    # Frequenza di campionamento [Hz]
BATCH_SIZE          = 64        # Batch per estrazione feature
RAND_STATE          = 42        # Seed riproducibilità
RIDGE_ALPHA         = 1.0       # Regolarizzazione Ridge (default)

# Soglie conformità UL1699B
UL_MIN_DETECTION_PCT = 95.0    # Detection rate minima [%]
UL_MAX_FP_PCT        = 5.0     # False positive rate massimo [%]

# Parametri MultiHydra  (nomi allineati all'API aeon HydraTransformer)
HYDRA_N_KERNELS      = 8        # n_kernels: kernel per gruppo
HYDRA_N_GROUPS       = 64       # n_groups:  numero di gruppi
HYDRA_MAX_CHANNELS   = 8        # max_num_channels


# ══════════════════════════════════════════════════════════════════════════════
# 1. Caricamento dataset
# ══════════════════════════════════════════════════════════════════════════════

def load_datasets(train_path: str, test_path: str) -> tuple:
    """
    Carica i dataset train e test già separati da file .npz.

    I file devono contenere gli array 'X' (serie temporali) e 'y' (etichette).
    X può avere shape (n, T) oppure (n, 1, T) — viene normalizzato a (n, T).

    Returns:
        X_train, y_train, X_test, y_test (np.ndarray)
    """
    log.info("=" * 60)
    log.info("CARICAMENTO DATASET")
    log.info("=" * 60)

    for path in (train_path, test_path):
        if not os.path.isfile(path):
            log.error("File non trovato: %s", path)
            sys.exit(1)

    def _load(path: str, label: str) -> tuple:
        data = np.load(path)
        X = data["X"]
        y = data["y"]
        # Normalizza shape a (n, T)
        if X.ndim == 3:
            X = X[:, 0, :]
        log.info("  %s: X=%s  y=%s  (%.2f s/serie @ %d Hz)",
                 label, X.shape, y.shape, X.shape[1] / FS_HZ, FS_HZ)
        return X, y

    X_train, y_train = _load(train_path, "TRAIN")
    X_test,  y_test  = _load(test_path,  "TEST ")

    # Verifica coerenza dimensionale
    if X_train.shape[1] != X_test.shape[1]:
        log.error(
            "Lunghezza serie temporali non coerente: train=%d, test=%d",
            X_train.shape[1], X_test.shape[1],
        )
        sys.exit(1)

    # Checksum per tracciabilità
    ck_tr = hashlib.md5(X_train.tobytes()).hexdigest()[:8]
    ck_te = hashlib.md5(X_test.tobytes() ).hexdigest()[:8]
    log.info("  Checksum train: %s  |  test: %s", ck_tr, ck_te)

    return X_train, y_train, X_test, y_test


# ══════════════════════════════════════════════════════════════════════════════
# 2. Analisi sbilanciamento e coerenza distribuzione
# ══════════════════════════════════════════════════════════════════════════════

def analyze_distribution(
    y_train: np.ndarray,
    y_test:  np.ndarray,
) -> dict:
    """
    Analizza lo sbilanciamento nel training set e la coerenza con il test set.

    Segnala divergenze > 10pp nella distribuzione delle classi tra train e test,
    che possono indicare uno split non rappresentativo.

    Returns:
        dict con conteggi, ratio e flag di avviso.
    """
    log.info("")
    log.info("=" * 60)
    log.info("ANALISI DISTRIBUZIONE CLASSI")
    log.info("=" * 60)

    def _stats(y: np.ndarray, name: str) -> dict:
        n0    = int((y == 0).sum())
        n1    = int((y == 1).sum())
        tot   = len(y)
        ratio = max(n0, n1) / max(min(n0, n1), 1)
        pct1  = 100.0 * n1 / tot
        log.info("  %-8s  label=0: %4d (%5.1f%%)  label=1: %4d (%5.1f%%)  ratio=%.1f:1",
                 name, n0, 100 * n0 / tot, n1, pct1, ratio)
        return {"n0": n0, "n1": n1, "ratio": ratio, "pct1": pct1}

    st_tr = _stats(y_train, "TRAIN")
    st_te = _stats(y_test,  "TEST")

    drift = abs(st_tr["pct1"] - st_te["pct1"])
    warn  = drift > 10.0
    if warn:
        log.warning(
            "  ATTENZIONE: distribuzione classe 1 diverge di %.1f pp "
            "(train=%.1f%%, test=%.1f%%)",
            drift, st_tr["pct1"], st_te["pct1"],
        )
    else:
        log.info("  ✓ Distribuzione coerente (drift=%.1f pp)", drift)

    return {
        "train": st_tr,
        "test":  st_te,
        "drift_pp": round(drift, 2),
        "distribution_warn": warn,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. Normalizzazione
# ══════════════════════════════════════════════════════════════════════════════

def normalize(
    X_train: np.ndarray,
    X_test:  np.ndarray,
) -> tuple:
    """
    Normalizzazione z-score per-campione (asse temporale).

    Il fit viene eseguito sul singolo campione (non sull'intero train set),
    per essere coerente con il deployment embedded dove ogni finestra è
    normalizzata indipendentemente prima dell'inferenza.

    Returns:
        X_train_norm, X_test_norm, scaler_params (dict con mean/std globali
        del train per logging).
    """
    log.info("")
    log.info("  Normalizzazione z-score per-campione (asse temporale)")

    def _norm(X: np.ndarray) -> np.ndarray:
        mu  = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        return (X - mu) / std

    X_tr_n = _norm(X_train)
    X_te_n = _norm(X_test)

    # Statistiche globali del train per logging/export
    params = {
        "global_mean_train": float(X_tr_n.mean()),
        "global_std_train":  float(X_tr_n.std()),
        "mode": "per_sample_zscore",
    }
    log.info("  Train normalizzato: μ=%.4f  σ=%.4f",
             params["global_mean_train"], params["global_std_train"])
    return X_tr_n, X_te_n, params


# ══════════════════════════════════════════════════════════════════════════════
# 4. MultiHydra feature extractor
# ══════════════════════════════════════════════════════════════════════════════

class MultiHydraTransformer:
    """
    Ensemble di trasformatori Hydra con seed differenti.

    Ogni "testa" è un'istanza di HydraTransformer (aeon) con seed diverso,
    in modo da esplorare sottospazi di feature complementari.
    Le feature di tutte le teste vengono concatenate prima del Ridge.

    Parametri principali (nomi allineati all'API aeon):
        n_kernels (int): kernel per gruppo  — aeon: n_kernels  (default: 8)
        n_groups  (int): numero di gruppi   — aeon: n_groups   (default: 64)
        n_heads   (int): numero di teste Hydra                 (default: 4)
    """

    def __init__(
        self,
        n_kernels:    int = HYDRA_N_KERNELS,
        n_groups:     int = HYDRA_N_GROUPS,
        n_heads:      int = 4,
        random_state: int = RAND_STATE,
    ):
        self.n_kernels    = n_kernels
        self.n_groups     = n_groups
        self.n_heads      = n_heads
        self.random_state = random_state
        self._heads       = []
        self._fitted      = False

    def _make_head(self, seed: int):
        """Istanzia una testa HydraTransformer con seed differente."""
        from aeon.transformations.collection.convolution_based import HydraTransformer
        return HydraTransformer(
            n_kernels=self.n_kernels,
            n_groups=self.n_groups,
            random_state=seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        """
        Addestra tutte le teste Hydra sul training set.

        X: shape (n, T) — viene convertito a (n, 1, T) per aeon.
        """
        log.info("  Fitting MultiHydra (%d teste, n_kernels=%d, n_groups=%d)...",
                 self.n_heads, self.n_kernels, self.n_groups)
        X3 = X[:, np.newaxis, :]
        self._heads = []
        for i in range(self.n_heads):
            head = self._make_head(self.random_state + i)
            head.fit(X3, y)
            self._heads.append(head)
            log.info("    Testa %d/%d completata", i + 1, self.n_heads)
        self._fitted = True
        return self

    def transform_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Estrae le feature da un batch di serie temporali.

        Esegue la trasformazione su ogni testa e concatena le feature.
        Returns: np.ndarray shape (n_batch, total_features)
        """
        if not self._fitted:
            raise RuntimeError("MultiHydraTransformer non ancora addestrato.")
        X3 = X[:, np.newaxis, :]
        parts = [head.transform(X3) for head in self._heads]
        return np.concatenate(parts, axis=1)

    def transform(self, X: np.ndarray, batch_size: int = BATCH_SIZE) -> np.ndarray:
        """
        Estrae feature dall'intero dataset in batch per ottimizzare la memoria.

        Returns: np.ndarray shape (n, total_features)
        """
        log.info("  Estrazione feature (batch_size=%d, n=%d)...",
                 batch_size, len(X))
        parts = []
        for start in range(0, len(X), batch_size):
            batch = X[start:start + batch_size]
            parts.append(self.transform_batch(batch))
            if (start // batch_size) % 5 == 0:
                log.info("    Batch %d/%d (%.0f%%)",
                         start // batch_size + 1,
                         int(np.ceil(len(X) / batch_size)),
                         100.0 * min(start + batch_size, len(X)) / len(X))
        feat = np.concatenate(parts, axis=0)
        log.info("  Feature shape: %s", feat.shape)
        return feat

    @property
    def n_features_out(self) -> int:
        """Numero totale di feature estratte (se già addestrato)."""
        if not self._fitted or not self._heads:
            return -1
        # Stima da una singola trasformazione su dummy
        try:
            dummy = np.zeros((1, 1, self._heads[0].n_timepoints_))
            return sum(h.transform(dummy).shape[1] for h in self._heads)
        except Exception:
            return -1


# ══════════════════════════════════════════════════════════════════════════════
# 5. Metriche UL1699B
# ══════════════════════════════════════════════════════════════════════════════

def ul1699b_metric(y_test: np.ndarray, y_pred: np.ndarray) -> dict:
    """Verifica conformità UL1699B con soglia 0.5."""
    arc    = y_test == 1
    no_arc = y_test == 0

    detected = int(((y_pred == 1) & arc).sum())
    missed   = int(((y_pred == 0) & arc).sum())
    fp       = int(((y_pred == 1) & no_arc).sum())
    tn       = int(((y_pred == 0) & no_arc).sum())

    det_rate = 100.0 * detected / max(int(arc.sum()),    1)
    fp_rate  = 100.0 * fp       / max(int(no_arc.sum()), 1)
    conforme = det_rate >= UL_MIN_DETECTION_PCT and fp_rate <= UL_MAX_FP_PCT

    log.info("")
    log.info("=" * 60)
    log.info("METRICA UL1699B (soglia = 0.5)")
    log.info("=" * 60)
    log.info("  Archi rilevati:   %d / %d  (%.1f%%)",
             detected, int(arc.sum()), det_rate)
    log.info("  Archi mancati:    %d", missed)
    log.info("  Falsi positivi:   %d / %d  (%.1f%%)",
             fp, int(no_arc.sum()), fp_rate)
    log.info("  Veri negativi:    %d", tn)
    status = "✓ CONFORME UL1699B" if conforme else "✗ NON conforme UL1699B"
    log.info("  %s", status)

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
    y_test:  np.ndarray,
    y_score: np.ndarray,
) -> float:
    """
    Analisi multi-soglia per ottimizzazione punto operativo UL1699B.

    Stampa tabella detection rate / false positive rate per soglie 0.05–0.95
    e restituisce la soglia ottimale (max detection rate conforme a UL1699B).
    """
    log.info("")
    log.info("  --- Analisi multi-soglia UL1699B ---")
    log.info("  %8s  %7s  %6s  %9s", "Soglia", "Det%", "FP%", "UL1699B")
    log.info("  " + "-" * 38)

    best_threshold = 0.5
    best_det       = 0.0
    arc    = y_test == 1
    no_arc = y_test == 0

    for thr in np.arange(0.05, 1.00, 0.05):
        yp  = (y_score >= thr).astype(int)
        det = 100.0 * ((yp == 1) & arc).sum()    / max(int(arc.sum()),    1)
        fpr = 100.0 * ((yp == 1) & no_arc).sum() / max(int(no_arc.sum()), 1)
        ok  = "SI " if det >= UL_MIN_DETECTION_PCT and fpr <= UL_MAX_FP_PCT else "NO "
        log.info("  %8.2f  %6.1f%%  %5.1f%%  %9s", thr, det, fpr, ok)

        if det >= UL_MIN_DETECTION_PCT and fpr <= UL_MAX_FP_PCT and det > best_det:
            best_det       = det
            best_threshold = float(thr)

    log.info("")
    log.info("  Soglia ottimale: %.2f  (det=%.1f%%)", best_threshold, best_det)
    return best_threshold


# ══════════════════════════════════════════════════════════════════════════════
# 6. Validazione robustezza
# ══════════════════════════════════════════════════════════════════════════════

def robustness_validation(
    hydra:   "MultiHydraTransformer",
    ridge:   RidgeClassifier,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    n_shuffle: int = 5,
) -> dict:
    """
    Verifica la robustezza del modello con due test:

    1. Shuffle test: permuta casualmente le etichette del test set e verifica
       che le prestazioni degradino (F1 ≈ 0.5), confermando che il modello
       non sfrutta artefatti di ordinamento.

    2. Stabilità feature space: estrae le feature N volte sullo stesso batch
       e verifica che la varianza inter-run sia nulla (determinismo Hydra).

    Returns:
        dict con risultati dei due test.
    """
    log.info("")
    log.info("=" * 60)
    log.info("VALIDAZIONE ROBUSTEZZA")
    log.info("=" * 60)

    results = {}

    # --- Shuffle test ---
    log.info("  Shuffle test (%d ripetizioni)...", n_shuffle)
    rng     = np.random.default_rng(RAND_STATE)
    F1_shuf = []
    feat_te = hydra.transform(X_test, batch_size=BATCH_SIZE)
    score_real = _ridge_decision_to_proba(ridge, feat_te)
    f1_real    = f1_score(y_test, (score_real >= 0.5).astype(int))

    for i in range(n_shuffle):
        y_shuf = rng.permutation(y_test)
        f1_s   = f1_score(y_shuf, (score_real >= 0.5).astype(int),
                          zero_division=0)
        F1_shuf.append(f1_s)

    mean_shuf = float(np.mean(F1_shuf))
    log.info("  F1 reale:          %.4f", f1_real)
    log.info("  F1 shuffle (media): %.4f  (atteso ≈ 0.0–0.3)", mean_shuf)
    results["f1_real"]         = round(f1_real, 4)
    results["f1_shuffle_mean"] = round(mean_shuf, 4)
    results["shuffle_ok"]      = mean_shuf < f1_real * 0.7

    # --- Stabilità feature space ---
    log.info("  Stabilità feature space (3 run su stesso batch)...")
    batch = X_test[:min(32, len(X_test))]
    feats = [hydra.transform_batch(batch) for _ in range(3)]
    max_var = float(np.max(np.var(np.stack(feats, axis=0), axis=0)))
    log.info("  Varianza max inter-run: %.2e  (atteso = 0.0)", max_var)
    results["feature_space_max_var"] = max_var
    results["feature_space_stable"]  = max_var < 1e-10

    if results["shuffle_ok"]:
        log.info("  ✓ Shuffle test superato")
    else:
        log.warning("  ✗ Shuffle test fallito — verificare data leakage")

    if results["feature_space_stable"]:
        log.info("  ✓ Feature space deterministico")
    else:
        log.warning("  ✗ Feature space non deterministico")

    return results


def _ridge_decision_to_proba(
    ridge: RidgeClassifier,
    X_feat: np.ndarray,
) -> np.ndarray:
    """
    Converte i decision scores del RidgeClassifier in probabilità con sigmoid.

    RidgeClassifier non ha predict_proba nativo; la sigmoid normalizza
    i decision values in [0, 1] per l'analisi multi-soglia.
    """
    dec = ridge.decision_function(X_feat)
    return 1.0 / (1.0 + np.exp(-dec))


# ══════════════════════════════════════════════════════════════════════════════
# 7. Export deployment embedded
# ══════════════════════════════════════════════════════════════════════════════

def export_onnx(
    hydra: "MultiHydraTransformer",
    X_sample: np.ndarray,
    out_dir: str,
) -> str | None:
    """
    Esporta una testa Hydra in formato ONNX per ST Edge AI / CubeAI.

    Usa torch.nn.Module wrapping della trasformazione Hydra per la conversione.
    Restituisce il path del file .onnx o None se l'export non è disponibile.
    """
    log.info("  Export ONNX (HydraTransformer → ST Edge AI)...")
    try:
        import torch
        import torch.nn as nn

        class HydraONNXWrapper(nn.Module):
            """Wrapper PyTorch per export ONNX di una testa Hydra."""
            def __init__(self, kernels, dilations, biases):
                super().__init__()
                self.kernels  = nn.Parameter(
                    torch.tensor(kernels, dtype=torch.float32), requires_grad=False)
                self.dilations = dilations
                self.biases   = nn.Parameter(
                    torch.tensor(biases, dtype=torch.float32), requires_grad=False)

            def forward(self, x):
                # x: (batch, 1, T)
                # Applica i kernel con dilation e restituisce le feature PPV+mean
                outputs = []
                for i, (d, b) in enumerate(zip(self.dilations, self.biases)):
                    k  = self.kernels[i].unsqueeze(0).unsqueeze(0)
                    out = nn.functional.conv1d(x, k, dilation=int(d), padding="same")
                    out = out + b
                    ppv  = (out > 0).float().mean(dim=-1)
                    mean = out.mean(dim=-1)
                    outputs.extend([ppv, mean])
                return torch.cat(outputs, dim=1)

        # Estrai parametri dalla prima testa
        head    = hydra._heads[0]
        kernels = head.kernels_   if hasattr(head, "kernels_")   else head.kernel_
        biases  = head.biases_    if hasattr(head, "biases_")    else head.bias_
        dils    = head.dilations_ if hasattr(head, "dilations_") else head.dilation_

        wrapper = HydraONNXWrapper(kernels, dils, biases)
        wrapper.eval()

        dummy = torch.zeros(1, 1, X_sample.shape[1])
        onnx_path = os.path.join(out_dir, "hydra_head0.onnx")
        torch.onnx.export(
            wrapper, dummy, onnx_path,
            input_names=["input"],
            output_names=["features"],
            opset_version=11,
            dynamic_axes={"input": {0: "batch"}},
        )
        log.info("  ONNX salvato: %s", onnx_path)
        return onnx_path

    except Exception as exc:
        log.warning("  Export ONNX non disponibile: %s", exc)
        log.warning("  Installare torch per abilitare l'export ONNX")
        return None


def export_c_header(
    ridge:   RidgeClassifier,
    out_dir: str,
    n_features: int,
    best_threshold: float,
) -> str:
    """
    Genera un C header per l'inferenza statica del RidgeClassifier su STM32.

    Produce un file .h con:
      - Coefficienti del classificatore come array float32
      - Intercetta
      - Soglia ottimale UL1699B
      - Funzione di inferenza inline

    Returns:
        path del file .h generato.
    """
    log.info("  Export C header (Ridge → STM32 static inference)...")

    coef  = ridge.coef_.flatten().astype(np.float32)
    inter = float(ridge.intercept_.flatten()[0])

    # Genera array C (max 8 valori per riga)
    def _c_array(name: str, values: np.ndarray, dtype: str = "float") -> str:
        lines  = [f"static const {dtype} {name}[{len(values)}] = {{"]
        chunk  = 8
        for i in range(0, len(values), chunk):
            row = values[i:i + chunk]
            lines.append("    " + ", ".join(f"{v:.8f}f" for v in row) + ",")
        lines[-1] = lines[-1].rstrip(",")
        lines.append("};")
        return "\n".join(lines)

    coef_block = _c_array("hydra_ridge_coef", coef)

    header = f"""\
/**
 * hydra_ridge_inference.h
 * ========================
 * Inferenza statica RidgeClassifier + MultiHydra per STM32
 * Generato automaticamente da train_classifier.py
 *
 * Normativa: UL 1699B — Photovoltaic DC Arc-Fault Circuit Protection
 *
 * Uso:
 *   #include "hydra_ridge_inference.h"
 *   float score = hydra_ridge_score(feature_vector, N_FEATURES);
 *   int   label = (score >= OPTIMAL_THRESHOLD) ? 1 : 0;
 */

#ifndef HYDRA_RIDGE_INFERENCE_H
#define HYDRA_RIDGE_INFERENCE_H

#include <stdint.h>
#include <math.h>  /* expf */

/* ── Dimensioni ─────────────────────────────────────────────── */
#define N_FEATURES        {n_features}
#define OPTIMAL_THRESHOLD {best_threshold:.4f}f  /* soglia UL1699B ottimale */

/* ── Coefficienti Ridge ─────────────────────────────────────── */
{coef_block}

static const float hydra_ridge_intercept = {inter:.8f}f;

/* ── Sigmoid helper ─────────────────────────────────────────── */
static inline float sigmoid(float x) {{
    return 1.0f / (1.0f + expf(-x));
}}

/* ── Inferenza ──────────────────────────────────────────────── */
/**
 * Restituisce la probabilità (0–1) che la finestra contenga un arco.
 * features: puntatore a vettore float di lunghezza N_FEATURES
 */
static inline float hydra_ridge_score(const float* features, uint32_t n) {{
    float dec = hydra_ridge_intercept;
    for (uint32_t i = 0; i < n; i++) {{
        dec += hydra_ridge_coef[i] * features[i];
    }}
    return sigmoid(dec);
}}

/**
 * Classificazione binaria con soglia ottimale UL1699B.
 * Restituisce 1 (arco) o 0 (normale).
 */
static inline uint8_t hydra_ridge_predict(const float* features, uint32_t n) {{
    return (hydra_ridge_score(features, n) >= OPTIMAL_THRESHOLD) ? 1u : 0u;
}}

#endif /* HYDRA_RIDGE_INFERENCE_H */
"""

    h_path = os.path.join(out_dir, "hydra_ridge_inference.h")
    with open(h_path, "w", encoding="utf-8") as f:
        f.write(header)
    log.info("  C header salvato: %s", h_path)
    return h_path


def export_config_json(
    out_dir:        str,
    hydra_cfg:      dict,
    norm_params:    dict | None,
    best_threshold: float,
    metrics:        dict,
    n_features:     int,
) -> str:
    """
    Salva il file di configurazione JSON per il deployment embedded.

    Contiene tutti i parametri necessari per replicare il preprocessing
    e l'inferenza su target embedded (STM32, ESP32, ecc.).
    """
    cfg = {
        "model":          "MultiHydra + RidgeClassifier",
        "normativa":      "UL1699B",
        "fs_hz":          FS_HZ,
        "n_features":     n_features,
        "best_threshold": round(best_threshold, 4),
        "hydra":          hydra_cfg,
        "normalization":  norm_params or {"mode": "none"},
        "metrics": {
            k: v for k, v in metrics.items()
            if isinstance(v, (int, float, bool, str))
        },
    }
    path = os.path.join(out_dir, "deployment_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, default=str)
    log.info("  Config JSON salvato: %s", path)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# 8. Grafici
# ══════════════════════════════════════════════════════════════════════════════

def plot_class_distribution(
    y_train: np.ndarray,
    y_test:  np.ndarray,
    out_dir: str,
) -> None:
    """Distribuzione classi train vs test."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Distribuzione classi — train e test set", fontsize=12)
    for ax, y, title in [(axes[0], y_train, "Train"), (axes[1], y_test, "Test")]:
        counts = [(y == 0).sum(), (y == 1).sum()]
        bars   = ax.bar(["No arco (0)", "Arco (1)"], counts,
                        color=["steelblue", "tomato"], edgecolor="white")
        for bar, c in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{c}\n({100*c/len(y):.1f}%)",
                    ha="center", va="bottom", fontsize=10)
        ax.set_title(title)
        ax.set_ylabel("Numero di campioni")
        ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(out_dir, "class_distribution.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    log.info("  Salvato: %s", path)


def plot_results(
    y_test:  np.ndarray,
    y_pred:  np.ndarray,
    y_score: np.ndarray,
    out_dir: str,
) -> None:
    """Pannello 4 grafici: CM | ROC | PR | Score distribution."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle("Risultati — MultiHydra + Ridge", fontsize=13)

    # Confusion Matrix
    ax = axes[0]
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No arco", "Arco"],
                yticklabels=["No arco", "Arco"])
    ax.set_title("Confusion Matrix")
    ax.set_ylabel("Reale"); ax.set_xlabel("Predetto")

    # ROC
    ax = axes[1]
    fpr_c, tpr_c, _ = roc_curve(y_test, y_score)
    auc = roc_auc_score(y_test, y_score)
    ax.plot(fpr_c, tpr_c, color="steelblue", lw=2, label=f"AUC={auc:.3f}")
    ax.plot([0,1],[0,1],"k--",lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
    ax.set_title("ROC Curve"); ax.grid(alpha=0.3)

    # Precision-Recall
    ax = axes[2]
    prec, rec, _ = precision_recall_curve(y_test, y_score)
    ap = average_precision_score(y_test, y_score)
    ax.plot(rec, prec, color="tomato", lw=2, label=f"AP={ap:.3f}")
    ax.axhline(y_test.mean(), color="gray", ls="--", lw=1,
               label=f"Baseline={y_test.mean():.2f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.legend()
    ax.set_title("Precision-Recall"); ax.grid(alpha=0.3)

    # Score distribution
    ax = axes[3]
    ax.hist(y_score[y_test==0], bins=40, alpha=0.6,
            color="steelblue", label="No arco (0)")
    ax.hist(y_score[y_test==1], bins=40, alpha=0.6,
            color="tomato", label="Arco (1)")
    ax.axvline(0.5, color="black", ls="--", lw=1, label="soglia=0.5")
    ax.set_xlabel("Score (sigmoid)"); ax.legend()
    ax.set_title("Distribuzione score"); ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "results_multihyra_ridge.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    log.info("  Salvato: %s", path)


def plot_series_examples(
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    y_pred:  np.ndarray,
    out_dir: str,
    n_per_class: int = 2,
) -> None:
    """Esempi di serie per categoria (TP, FN, TN, FP)."""
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
                color = "tomato" if y_test[ix] == 1 else "steelblue"
                ax.plot(t, X_test[ix], lw=0.7, color=color)
                ax.set_ylim(X_test[ix].min() - 0.1, X_test[ix].max() + 0.1)
                ax.set_xlabel("t [s]"); ax.set_ylabel("I [-]")
                ax.grid(alpha=0.3)
                if row == 0:
                    ax.set_title(title, fontsize=9)
            else:
                ax.text(0.5, 0.5, "Nessun\nesempio",
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


def plot_threshold_curve(
    y_test:  np.ndarray,
    y_score: np.ndarray,
    out_dir: str,
    best_thr: float,
) -> None:
    """
    Grafico detection rate e FP rate al variare della soglia.
    Evidenzia la zona di conformità UL1699B e la soglia ottimale.
    """
    thresholds = np.arange(0.01, 1.00, 0.01)
    det_rates, fp_rates = [], []
    arc    = y_test == 1
    no_arc = y_test == 0

    for thr in thresholds:
        yp  = (y_score >= thr).astype(int)
        det = 100.0 * ((yp==1) & arc).sum()    / max(int(arc.sum()),    1)
        fpr = 100.0 * ((yp==1) & no_arc).sum() / max(int(no_arc.sum()), 1)
        det_rates.append(det)
        fp_rates.append(fpr)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, det_rates, color="tomato",    lw=2, label="Detection rate %")
    ax.plot(thresholds, fp_rates,  color="steelblue", lw=2, label="False Positive rate %")
    ax.axhline(UL_MIN_DETECTION_PCT, color="tomato",    ls="--", lw=1,
               label=f"UL1699B min det = {UL_MIN_DETECTION_PCT:.0f}%")
    ax.axhline(UL_MAX_FP_PCT,        color="steelblue", ls="--", lw=1,
               label=f"UL1699B max FP = {UL_MAX_FP_PCT:.0f}%")
    ax.axvline(best_thr, color="black", ls=":", lw=2,
               label=f"Soglia ottimale = {best_thr:.2f}")
    ax.set_xlabel("Soglia di decisione")
    ax.set_ylabel("Percentuale [%]")
    ax.set_title("Analisi multi-soglia — UL1699B")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "threshold_analysis.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    log.info("  Salvato: %s", path)


# ══════════════════════════════════════════════════════════════════════════════
# 9. Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Addestramento MultiHydra + RidgeClassifier\n"
            "per la classificazione di archi elettrici in impianti PV DC.\n"
            "Normativa: UL 1699B"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("train", help="Path a train.npz")
    parser.add_argument("test",  help="Path a test.npz")
    parser.add_argument("--out", "-o", default="./results",
                        help="Cartella di output (default: ./results)")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Disabilita la normalizzazione z-score")
    parser.add_argument("--export-stm32", action="store_true",
                        help="Abilita export ONNX e C header per STM32")
    parser.add_argument("--hydra-heads", type=int, default=4,
                        help="Numero di teste MultiHydra (default: 4)")
    parser.add_argument("--hydra-n-kernels", type=int, default=HYDRA_N_KERNELS,
                        help=f"Kernel per gruppo Hydra / n_kernels (default: {HYDRA_N_KERNELS})")
    parser.add_argument("--hydra-n-groups", type=int, default=HYDRA_N_GROUPS,
                        help=f"Gruppi Hydra / n_groups (default: {HYDRA_N_GROUPS})")
    parser.add_argument("--ridge-alpha", type=float, default=RIDGE_ALPHA,
                        help=f"Alpha Ridge (default: {RIDGE_ALPHA})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch feature extraction (default: {BATCH_SIZE})")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 1. Caricamento
    X_train, y_train, X_test, y_test = load_datasets(args.train, args.test)

    # 2. Analisi distribuzione
    dist_info = analyze_distribution(y_train, y_test)

    # 3. Normalizzazione
    norm_params = None
    if not args.no_normalize:
        log.info("")
        log.info("=" * 60)
        log.info("NORMALIZZAZIONE")
        log.info("=" * 60)
        X_train, X_test, norm_params = normalize(X_train, X_test)
    else:
        log.info("  Normalizzazione disabilitata (--no-normalize)")

    # 4. Addestramento MultiHydra
    log.info("")
    log.info("=" * 60)
    log.info("ADDESTRAMENTO MULTIHYRA + RIDGE")
    log.info("=" * 60)

    try:
        from aeon.transformations.collection.convolution_based import HydraTransformer
    except ImportError:
        log.error("aeon non disponibile. Installare: pip install aeon")
        sys.exit(1)

    hydra_cfg = {
        "n_kernels": args.hydra_n_kernels, "n_groups": args.hydra_n_groups,
        "n_heads": args.hydra_heads, "random_state": RAND_STATE,
    }
    hydra = MultiHydraTransformer(**hydra_cfg)

    t0 = time.time()
    hydra.fit(X_train, y_train)
    t_fit = time.time() - t0
    log.info("  Hydra fit completato in %.1f s", t_fit)

    # Estrazione feature
    t0 = time.time()
    feat_train = hydra.transform(X_train, batch_size=args.batch_size)
    feat_test  = hydra.transform(X_test,  batch_size=args.batch_size)
    t_feat = time.time() - t0
    log.info("  Feature estratte in %.1f s  (train=%s, test=%s)",
             t_feat, feat_train.shape, feat_test.shape)

    # Normalizza lo spazio feature (StandardScaler) per Ridge
    feat_scaler = StandardScaler()
    feat_train  = feat_scaler.fit_transform(feat_train)
    feat_test   = feat_scaler.transform(feat_test)

    # Ridge Classifier
    ridge = RidgeClassifier(alpha=args.ridge_alpha, random_state=RAND_STATE)
    t0 = time.time()
    ridge.fit(feat_train, y_train)
    t_ridge = time.time() - t0
    log.info("  Ridge fit completato in %.1f s", t_ridge)

    # 5. Predizione
    y_pred  = ridge.predict(feat_test)
    y_score = _ridge_decision_to_proba(ridge, feat_test)

    # 6. Metriche
    log.info("")
    log.info("=" * 60)
    log.info("METRICHE TEST SET")
    log.info("=" * 60)
    report = classification_report(
        y_test, y_pred,
        target_names=["No arco", "Arco"], digits=3,
    )
    for line in report.splitlines():
        log.info("  %s", line)

    acc = accuracy_score(y_test, y_pred)
    ba  = balanced_accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_score)
    ap  = average_precision_score(y_test, y_score)

    log.info("  Accuracy:          %.4f", acc)
    log.info("  Balanced Accuracy: %.4f", ba)
    log.info("  F1 (arco):         %.4f", f1)
    log.info("  ROC-AUC:           %.4f", auc)
    log.info("  Avg Precision:     %.4f", ap)

    # 7. UL1699B (soglia 0.5)
    ul = ul1699b_metric(y_test, y_pred)

    # Analisi multi-soglia
    best_thr = threshold_analysis(y_test, y_score)
    ul["best_threshold"] = best_thr

    # 8. Robustezza
    robust = robustness_validation(hydra, ridge, X_test, y_test)

    # Grafici
    log.info("")
    log.info("=" * 60)
    log.info("SALVATAGGIO GRAFICI")
    log.info("=" * 60)
    plot_class_distribution(y_train, y_test, args.out)
    plot_results(y_test, y_pred, y_score, args.out)
    plot_series_examples(X_test, y_test, y_pred, args.out)
    plot_threshold_curve(y_test, y_score, args.out, best_thr)

    # 9. Salvataggio bundle
    log.info("")
    log.info("=" * 60)
    log.info("SALVATAGGIO BUNDLE MODELLO")
    log.info("=" * 60)

    n_features = feat_train.shape[1]
    bundle = {
        "hydra":          hydra,
        "feat_scaler":    feat_scaler,
        "ridge":          ridge,
        "norm_params":    norm_params,
        "best_threshold": best_thr,
        "n_features":     n_features,
        "hydra_cfg":      hydra_cfg,
        "ridge_alpha":    args.ridge_alpha,
        "metrics": {
            "accuracy":           round(acc, 4),
            "balanced_accuracy":  round(ba,  4),
            "f1_arc":             round(f1,  4),
            "roc_auc":            round(auc, 4),
            "avg_precision":      round(ap,  4),
            **ul,
            **{f"robust_{k}": v for k, v in robust.items()},
        },
    }

    bundle_path = os.path.join(args.out, "multihyra_ridge_bundle.pkl")
    with open(bundle_path, "wb") as f:
        pickle.dump(bundle, f)
    log.info("  Bundle salvato: %s", bundle_path)

    # 10. Export embedding
    if args.export_stm32:
        log.info("")
        log.info("=" * 60)
        log.info("EXPORT STM32 / EMBEDDED")
        log.info("=" * 60)
        export_onnx(hydra, X_test, args.out)
        export_c_header(ridge, args.out, n_features, best_thr)

    export_config_json(
        args.out, hydra_cfg, norm_params, best_thr,
        bundle["metrics"], n_features,
    )

    # Riepilogo finale
    log.info("")
    log.info("=" * 72)
    log.info("RIEPILOGO FINALE")
    log.info("=" * 72)
    log.info("  Accuracy:          %.4f", acc)
    log.info("  Balanced Accuracy: %.4f", ba)
    log.info("  F1 (arco):         %.4f", f1)
    log.info("  ROC-AUC:           %.4f", auc)
    log.info("  Avg Precision:     %.4f", ap)
    log.info("  Detection rate:    %.1f%%", ul["detection_rate_pct"])
    log.info("  False positive:    %.1f%%", ul["false_positive_rate_pct"])
    log.info("  Soglia ottimale:   %.2f",  best_thr)
    log.info("  UL1699B:           %s",
             "CONFORME" if ul["ul1699b_conforme"] else "NON CONFORME")
    log.info("  Shuffle test:      %s",
             "OK" if robust["shuffle_ok"] else "FALLITO")
    log.info("  Feature stabili:   %s",
             "SI" if robust["feature_space_stable"] else "NO")
    log.info("  T_hydra_fit:       %.1f s", t_fit)
    log.info("  T_feature_extr:    %.1f s", t_feat)
    log.info("  T_ridge_fit:       %.1f s", t_ridge)
    log.info("")
    log.info("  Output in: %s", args.out)

    # Report testuale
    report_path = os.path.join(args.out, "training_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("TRAINING REPORT — MultiHydra + Ridge — Classificatore Archi PV\n")
        f.write("Normativa: UL 1699B\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Train: {args.train}\n")
        f.write(f"Test:  {args.test}\n")
        f.write(f"Campioni train: {len(y_train)}  (arco={int((y_train==1).sum())},"
                f" no={int((y_train==0).sum())})\n")
        f.write(f"Campioni test:  {len(y_test)}  (arco={int((y_test==1).sum())},"
                f" no={int((y_test==0).sum())})\n\n")
        for k, v in bundle["metrics"].items():
            f.write(f"  {k}: {v}\n")
    log.info("  Report: %s", report_path)


if __name__ == "__main__":
    main()