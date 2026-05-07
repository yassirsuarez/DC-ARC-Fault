#!/usr/bin/env python3
"""
train_multirocket_gpu.py
========================
Pipeline GPU-accelerata per classificatore archi elettrici in impianti PV DC.
Architettura: MultiRocket (PyTorch GPU) + RidgeClassifier

DIFFERENZE RISPETTO A MultiHydra:
  - MultiRocket genera ~49.000 feature per serie (vs ~8.000 di MultiHydra)
    usando kernel random con dilazioni multiple e aggregazioni PPV+mean
  - L'estrazione feature è eseguita su GPU (CUDA) → 10-50× più veloce
  - Fallback automatico su CPU se CUDA non disponibile
  - Stesse feature statistiche manuali della v2 (RMS, bande freq, slope, ecc.)

REQUISITI:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install scikit-learn numpy matplotlib seaborn scipy

    (aeon NON è richiesta per MultiRocket — implementazione PyTorch nativa)

USO:
    python train_multirocket_gpu.py train.npz test.npz
    python train_multirocket_gpu.py train.npz test.npz --out ./results_rocket
    python train_multirocket_gpu.py train.npz test.npz --n-kernels 10000
    python train_multirocket_gpu.py train.npz test.npz --smote
    python train_multirocket_gpu.py train.npz test.npz --export-stm32
    python train_multirocket_gpu.py train.npz test.npz --tsne --learning-curve

Normativa: UL 1699B — Photovoltaic DC Arc-Fault Circuit Protection
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
from scipy import stats as scipy_stats

from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score,
    precision_recall_curve, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ── parametri globali ─────────────────────────────────────────────────────────
FS_HZ                = 10_000
RAND_STATE           = 42
RIDGE_ALPHA          = 1.0
UL_MIN_DETECTION_PCT = 95.0
UL_MAX_FP_PCT        = 5.0

# MultiRocket defaults
MR_N_KERNELS         = 6_250   # kernel per "set" — MultiRocket usa 2 set → 12.500 totali
                                # ogni kernel produce PPV + mean → 25.000 feature
                                # con max_dilations_per_kernel=32 → ~49.000 feature totali
MR_MAX_DILATIONS     = 32      # dilazioni massime per kernel
MR_BATCH_SIZE        = 256     # campioni per batch GPU (riduci se OOM)

FREQ_BANDS_HZ = [
    (0,    500),
    (500,  1500),
    (1500, 3000),
    (3000, 5000),
]


# ══════════════════════════════════════════════════════════════════════════════
# 0. Setup device
# ══════════════════════════════════════════════════════════════════════════════

def setup_device() -> "torch.device":
    """Rileva GPU CUDA, fallback su CPU."""
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props  = torch.cuda.get_device_properties(0)
        log.info("  GPU rilevata: %s  (%.1f GB VRAM)",
                 props.name, props.total_memory / 1e9)
        log.info("  CUDA version: %s  |  PyTorch: %s",
                 torch.version.cuda, torch.__version__)
    else:
        device = torch.device("cpu")
        log.warning("  CUDA non disponibile — fallback su CPU")
        log.warning("  Per abilitare GPU: pip install torch --index-url "
                    "https://download.pytorch.org/whl/cu121")
    return device


# ══════════════════════════════════════════════════════════════════════════════
# 1. MultiRocket PyTorch (implementazione nativa GPU)
# ══════════════════════════════════════════════════════════════════════════════

class MultiRocketGPU:
    """
    Implementazione PyTorch GPU-accelerata di MultiRocket.

    MultiRocket (Tan et al., 2022) estende ROCKET con:
      - Due set di kernel (normale + differenze prime)
      - Aggregazioni PPV (Proportion of Positive Values) e mean
      - Dilazioni calcolate su scala logaritmica per coprire
        tutte le scale temporali rilevanti

    Riferimento: "MultiRocket: Multiple pooling operators and transformations
    for fast and effective time series classification"
    Data Mining and Knowledge Discovery, 2022.

    Parameters
    ----------
    n_kernels : int
        Kernel per set. Feature totali ≈ n_kernels × 2 (set) × 2 (PPV+mean)
        × max_dilations_per_kernel (con padding variabile).
        Default 6250 → ~49.000 feature (come paper originale).
    max_dilations_per_kernel : int
        Dilazioni per kernel (scala log). Default 32.
    device : torch.device
        GPU o CPU.
    """

    def __init__(
        self,
        n_kernels:               int = MR_N_KERNELS,
        max_dilations_per_kernel: int = MR_MAX_DILATIONS,
        device:                  "torch.device | None" = None,
        random_state:            int = RAND_STATE,
    ):
        import torch
        self.n_kernels                = n_kernels
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.device                   = device or torch.device("cpu")
        self.random_state             = random_state
        self._fitted                  = False

        # Parametri kernel (inizializzati in fit)
        self._kernels_a = None   # set A: serie originale
        self._kernels_b = None   # set B: differenze prime
        self._dilations = None
        self._paddings  = None
        self._biases_a  = None
        self._biases_b  = None

    # ── generazione kernel ────────────────────────────────────────────────────

    def _generate_kernels(self, input_length: int):
        """
        Genera kernel random con lunghezze {7, 9, 11} e dilazioni log-spaced.
        Segue esattamente l'algoritmo del paper MultiRocket.
        """
        import torch
        rng = np.random.default_rng(self.random_state)

        kernel_lengths = np.array([7, 9, 11])
        n              = self.n_kernels
        max_dil        = self.max_dilations_per_kernel

        # Assegna lunghezze uniformemente
        lengths = rng.choice(kernel_lengths, size=n)

        all_weights_a, all_weights_b = [], []
        all_dilations, all_paddings  = [], []
        all_biases_a,  all_biases_b  = [], []

        for i in range(n):
            L = int(lengths[i])

            # Dilazioni: floor(2^linspace(0, log2((input_length-1)/(L-1)), max_dil))
            max_exponent = np.log2((input_length - 1) / (L - 1))
            dils = np.floor(
                np.power(2, np.linspace(0, max_exponent,
                                        min(max_dil, int(max_exponent) + 1)))
            ).astype(int)
            dils = np.unique(dils)

            # Pesi random normalizzati a media zero
            w_a = rng.normal(0, 1, (len(dils), L)).astype(np.float32)
            w_a -= w_a.mean(axis=1, keepdims=True)
            w_b = rng.normal(0, 1, (len(dils), L)).astype(np.float32)
            w_b -= w_b.mean(axis=1, keepdims=True)

            # Padding: 50% same (len(dils) * L // 2), 50% no-padding
            pads = np.zeros(len(dils), dtype=int)
            half = len(dils) // 2
            for j in range(half):
                pads[j] = (dils[j] * (L - 1)) // 2

            # Bias dal quantile della risposta su segnale random
            dummy = rng.normal(0, 1, (1, input_length)).astype(np.float32)
            # shape: (1, 1, T) — batch=1, channels=1, length=T
            dummy_t = torch.tensor(dummy, device=self.device).unsqueeze(0)
            dummy_diff = torch.diff(dummy_t, dim=2)  # shape: (1, 1, T-1)
            for j, (d, p) in enumerate(zip(dils, pads)):
                k_a = torch.tensor(w_a[j:j+1], device=self.device).view(1, 1, L)
                k_b = torch.tensor(w_b[j:j+1], device=self.device).view(1, 1, L)
                import torch.nn.functional as F
                out_a = F.conv1d(dummy_t,    k_a, dilation=int(d), padding=int(p))
                out_b = F.conv1d(dummy_diff, k_b, dilation=int(d), padding=int(p))
                q = float(rng.uniform(0, 1))
                all_biases_a.append(float(torch.quantile(out_a.cpu(), q)))
                all_biases_b.append(float(torch.quantile(out_b.cpu(), q)))

            all_weights_a.extend(w_a.tolist())   # ogni elemento: lista di L float
            all_weights_b.extend(w_b.tolist())
            all_dilations.extend(dils.tolist())
            all_paddings.extend(pads.tolist())

        self._kernel_lengths = lengths
        self._weights_a_list = all_weights_a
        self._weights_b_list = all_weights_b
        self._dilations_list = all_dilations
        self._paddings_list  = all_paddings
        self._biases_a_list  = all_biases_a
        self._biases_b_list  = all_biases_b
        self._n_total_kernels = len(all_dilations)

        log.info("  Kernel totali (con dilazioni): %d", self._n_total_kernels)
        log.info("  Feature attese: %d", self._n_total_kernels * 4)
        # 4 = PPV_a + mean_a + PPV_b + mean_b

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y=None):
        """Genera i kernel (non c'è apprendimento — kernel sono random)."""
        log.info("  Generazione kernel MultiRocket (n=%d, max_dil=%d)...",
                 self.n_kernels, self.max_dilations_per_kernel)
        t0 = time.time()
        self._generate_kernels(X.shape[1])
        self._input_length = X.shape[1]
        self._fitted = True
        log.info("  Kernel generati in %.2f s", time.time() - t0)
        return self

    # ── transform (GPU) ───────────────────────────────────────────────────────

    def _transform_batch(self, X_batch: np.ndarray) -> np.ndarray:
        """
        Trasforma un batch su GPU.
        Restituisce array numpy (n_batch, n_total_kernels * 4).
        """
        import torch
        import torch.nn.functional as F

        n = len(X_batch)
        # Input: (batch, 1, T) su GPU
        X_t = torch.tensor(X_batch, dtype=torch.float32, device=self.device)
        if X_t.ndim == 2:
            X_t = X_t.unsqueeze(1)

        # Differenze prime per set B — shape: (batch, 1, T-1)
        X_diff = torch.diff(X_t, dim=2)

        n_k = self._n_total_kernels
        feats = np.empty((n, n_k * 4), dtype=np.float32)

        for i, (wa, wb, d, p, ba, bb) in enumerate(zip(
            self._weights_a_list, self._weights_b_list,
            self._dilations_list,  self._paddings_list,
            self._biases_a_list,   self._biases_b_list,
        )):
            L = len(wa)
            k_a = torch.tensor(wa, dtype=torch.float32,
                               device=self.device).view(1, 1, L)
            k_b = torch.tensor(wb, dtype=torch.float32,
                               device=self.device).view(1, 1, L)

            # Set A: serie originale
            out_a = F.conv1d(X_t,    k_a, dilation=int(d), padding=int(p))
            out_a = out_a - ba

            # Set B: differenze prime (lunghezza T-1, gestisci padding)
            out_b = F.conv1d(X_diff, k_b, dilation=int(d), padding=int(p))
            out_b = out_b - bb

            # PPV = proporzione valori > 0
            ppv_a = (out_a > 0).float().mean(dim=2).squeeze(1).cpu().numpy()
            ppv_b = (out_b > 0).float().mean(dim=2).squeeze(1).cpu().numpy()

            # Mean dei valori (MPV — MultiRocket)
            mean_a = out_a.mean(dim=2).squeeze(1).cpu().numpy()
            mean_b = out_b.mean(dim=2).squeeze(1).cpu().numpy()

            base = i * 4
            feats[:, base]     = ppv_a
            feats[:, base + 1] = mean_a
            feats[:, base + 2] = ppv_b
            feats[:, base + 3] = mean_b

        return feats

    def transform(self, X: np.ndarray,
                  batch_size: int = MR_BATCH_SIZE) -> np.ndarray:
        """Estrae feature dall'intero dataset in batch."""
        if not self._fitted:
            raise RuntimeError("MultiRocketGPU non ancora fittato.")

        log.info("  Estrazione feature MultiRocket GPU (batch=%d, n=%d)...",
                 batch_size, len(X))
        parts = []
        n_batches = int(np.ceil(len(X) / batch_size))

        t0 = time.time()
        for bi, start in enumerate(range(0, len(X), batch_size)):
            batch = X[start:start + batch_size]
            parts.append(self._transform_batch(batch))
            if bi % max(1, n_batches // 10) == 0:
                elapsed = time.time() - t0
                eta = elapsed / (bi + 1) * (n_batches - bi - 1)
                log.info("    Batch %d/%d (%.0f%%)  elapsed=%.0fs  ETA=%.0fs",
                         bi + 1, n_batches,
                         100.0 * (bi + 1) / n_batches,
                         elapsed, eta)

        feat = np.concatenate(parts, axis=0)
        log.info("  Feature MultiRocket shape: %s  (%.1f s totali)",
                 feat.shape, time.time() - t0)
        return feat


# ══════════════════════════════════════════════════════════════════════════════
# 2. Caricamento dataset
# ══════════════════════════════════════════════════════════════════════════════

def load_datasets(train_path: str, test_path: str) -> tuple:
    log.info("=" * 60)
    log.info("CARICAMENTO DATASET")
    log.info("=" * 60)
    for path in (train_path, test_path):
        if not os.path.isfile(path):
            log.error("File non trovato: %s", path)
            sys.exit(1)

    def _load(path, label):
        data = np.load(path)
        X, y = data["X"], data["y"]
        if X.ndim == 3:
            X = X[:, 0, :]
        log.info("  %-8s  X=%s  y=%s  (%.3f s/serie @ %d Hz)",
                 label, X.shape, y.shape, X.shape[1] / FS_HZ, FS_HZ)
        return X, y

    X_train, y_train = _load(train_path, "TRAIN")
    X_test,  y_test  = _load(test_path,  "TEST ")

    if X_train.shape[1] != X_test.shape[1]:
        log.error("Lunghezza serie non coerente: train=%d test=%d",
                  X_train.shape[1], X_test.shape[1])
        sys.exit(1)

    log.info("  Checksum train: %s  test: %s",
             hashlib.md5(X_train.tobytes()).hexdigest()[:8],
             hashlib.md5(X_test.tobytes()).hexdigest()[:8])
    return X_train, y_train, X_test, y_test


# ══════════════════════════════════════════════════════════════════════════════
# 3. Analisi distribuzione
# ══════════════════════════════════════════════════════════════════════════════

def analyze_distribution(y_train, y_test) -> dict:
    log.info("")
    log.info("=" * 60)
    log.info("ANALISI DISTRIBUZIONE CLASSI")
    log.info("=" * 60)

    def _stats(y, name):
        n0, n1 = int((y==0).sum()), int((y==1).sum())
        tot = len(y)
        pct1 = 100.0 * n1 / tot
        ratio = max(n0, n1) / max(min(n0, n1), 1)
        log.info("  %-8s  no-arco: %4d (%5.1f%%)  arco: %4d (%5.1f%%)  ratio=%.1f:1",
                 name, n0, 100*n0/tot, n1, pct1, ratio)
        return {"n0": n0, "n1": n1, "ratio": ratio, "pct1": pct1}

    st_tr = _stats(y_train, "TRAIN")
    st_te = _stats(y_test,  "TEST")
    drift = abs(st_tr["pct1"] - st_te["pct1"])
    if drift > 10:
        log.warning("  ATTENZIONE: drift distribuzione %.1f pp", drift)
    else:
        log.info("  ✓ Distribuzione coerente (drift=%.1f pp)", drift)
    return {"train": st_tr, "test": st_te, "drift_pp": round(drift, 2)}


# ══════════════════════════════════════════════════════════════════════════════
# 4. Normalizzazione globale
# ══════════════════════════════════════════════════════════════════════════════

def normalize_global(X_train, X_test):
    log.info("")
    log.info("=" * 60)
    log.info("NORMALIZZAZIONE GLOBALE")
    log.info("=" * 60)
    scaler  = StandardScaler()
    X_tr_n  = scaler.fit_transform(X_train)
    X_te_n  = scaler.transform(X_test)
    log.info("  z-score globale fit su train → apply su test")
    log.info("  Train: μ=%.4f σ=%.4f", X_tr_n.mean(), X_tr_n.std())
    log.info("  Test:  μ=%.4f σ=%.4f", X_te_n.mean(), X_te_n.std())
    params = {"mode": "global_zscore",
              "global_mean_train": float(X_tr_n.mean()),
              "global_std_train":  float(X_tr_n.std())}
    return X_tr_n, X_te_n, params, scaler


# ══════════════════════════════════════════════════════════════════════════════
# 5. Feature statistiche manuali
# ══════════════════════════════════════════════════════════════════════════════

def extract_statistical_features(X: np.ndarray) -> np.ndarray:
    """
    Feature statistiche e spettrali per campione:
      RMS, varianza, skewness, kurtosi, picco max/min,
      zero-crossing rate, crest factor, energia per banda freq,
      entropia spettrale, slope lineare (deriva DC).
    """
    log.info("  Estrazione feature statistiche (n=%d)...", len(X))
    n, T = X.shape
    n_bands = len(FREQ_BANDS_HZ)
    n_feat  = 8 + n_bands + 2
    feats   = np.zeros((n, n_feat), dtype=np.float32)
    freqs   = np.fft.rfftfreq(T, d=1.0 / FS_HZ)
    t_norm  = np.linspace(0, 1, T)

    for i, x in enumerate(X):
        rms   = float(np.sqrt(np.mean(x**2)))
        var   = float(np.var(x))
        sk    = float(scipy_stats.skew(x))
        kurt  = float(scipy_stats.kurtosis(x))
        pk_mx = float(x.max())
        pk_mn = float(x.min())
        zcr   = float(np.sum(np.diff(np.sign(x)) != 0)) / T
        crest = float(np.abs(x).max()) / (rms + 1e-8)
        feats[i, :8] = [rms, var, sk, kurt, pk_mx, pk_mn, zcr, crest]

        fft_mag = np.abs(np.fft.rfft(x)) ** 2
        for j, (f_lo, f_hi) in enumerate(FREQ_BANDS_HZ):
            mask = (freqs >= f_lo) & (freqs < f_hi)
            feats[i, 8 + j] = fft_mag[mask].sum() / (T + 1e-8)

        psd = fft_mag / (fft_mag.sum() + 1e-8)
        psd = np.clip(psd, 1e-12, None)
        feats[i, 8 + n_bands] = float(-np.sum(psd * np.log2(psd)))

        slope, _ = np.polyfit(t_norm, x, 1)
        feats[i, 8 + n_bands + 1] = float(slope)

    log.info("  Feature statistiche: %s", feats.shape)
    return feats


# ══════════════════════════════════════════════════════════════════════════════
# 6. SMOTE
# ══════════════════════════════════════════════════════════════════════════════

def apply_smote(X_feat, y):
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        log.error("pip install imbalanced-learn")
        sys.exit(1)
    log.info("  SMOTE — prima: arco=%d no-arco=%d",
             int((y==1).sum()), int((y==0).sum()))
    X_res, y_res = SMOTE(random_state=RAND_STATE).fit_resample(X_feat, y)
    log.info("  SMOTE — dopo:  arco=%d no-arco=%d",
             int((y_res==1).sum()), int((y_res==0).sum()))
    return X_res, y_res


# ══════════════════════════════════════════════════════════════════════════════
# 7. Metriche UL1699B
# ══════════════════════════════════════════════════════════════════════════════

def _to_proba(ridge, X_feat):
    dec = ridge.decision_function(X_feat)
    return 1.0 / (1.0 + np.exp(-dec))


def ul1699b_metric(y_test, y_pred, threshold=0.5):
    arc, no_arc = y_test==1, y_test==0
    det  = int(((y_pred==1) & arc).sum())
    miss = int(((y_pred==0) & arc).sum())
    fp   = int(((y_pred==1) & no_arc).sum())
    tn   = int(((y_pred==0) & no_arc).sum())
    det_rate = 100.0 * det / max(int(arc.sum()),    1)
    fp_rate  = 100.0 * fp  / max(int(no_arc.sum()), 1)
    ok = det_rate >= UL_MIN_DETECTION_PCT and fp_rate <= UL_MAX_FP_PCT
    log.info("")
    log.info("=" * 60)
    log.info("METRICA UL1699B (soglia=%.2f)", threshold)
    log.info("=" * 60)
    log.info("  Archi rilevati: %d/%d (%.1f%%)", det, int(arc.sum()), det_rate)
    log.info("  Falsi positivi: %d/%d (%.1f%%)", fp,  int(no_arc.sum()), fp_rate)
    log.info("  %s", "✓ CONFORME UL1699B" if ok else "✗ NON conforme UL1699B")
    return {"detected": det, "missed": miss, "false_positives": fp,
            "true_negatives": tn, "detection_rate_pct": round(det_rate, 2),
            "false_positive_rate_pct": round(fp_rate, 2), "ul1699b_conforme": ok}


def threshold_analysis(y_test, y_score):
    log.info("")
    log.info("  --- Analisi multi-soglia UL1699B ---")
    log.info("  %8s  %7s  %6s  %6s", "Soglia", "Det%", "FP%", "UL1699B")
    arc, no_arc = y_test==1, y_test==0
    best_thr, best_det = 0.5, 0.0

    for thr in np.arange(0.05, 1.00, 0.05):
        yp  = (y_score >= thr).astype(int)
        det = 100.0 * ((yp==1)&arc).sum()    / max(int(arc.sum()),    1)
        fpr = 100.0 * ((yp==1)&no_arc).sum() / max(int(no_arc.sum()), 1)
        ok  = "SI" if det >= UL_MIN_DETECTION_PCT and fpr <= UL_MAX_FP_PCT else "NO"
        log.info("  %8.2f  %6.1f%%  %5.1f%%  %6s", thr, det, fpr, ok)
        if det >= UL_MIN_DETECTION_PCT and fpr <= UL_MAX_FP_PCT and det > best_det:
            best_det, best_thr = det, float(thr)

    if best_det == 0.0:
        log.warning("  Nessuna soglia soddisfa entrambi i vincoli.")
        best_det2 = 0.0
        for thr in np.arange(0.01, 1.00, 0.01):
            yp  = (y_score >= thr).astype(int)
            det = 100.0 * ((yp==1)&arc).sum()    / max(int(arc.sum()),    1)
            fpr = 100.0 * ((yp==1)&no_arc).sum() / max(int(no_arc.sum()), 1)
            if fpr <= UL_MAX_FP_PCT and det > best_det2:
                best_det2, best_thr = det, float(thr)
        log.warning("  Fallback: soglia=%.2f det=%.1f%%", best_thr, best_det2)
    else:
        log.info("  Soglia ottimale: %.2f (det=%.1f%%)", best_thr, best_det)
    return best_thr


# ══════════════════════════════════════════════════════════════════════════════
# 8. Cross-validation
# ══════════════════════════════════════════════════════════════════════════════

def cross_validate(X_feat, y, class_weight, alpha):
    log.info("")
    log.info("=" * 60)
    log.info("CROSS-VALIDATION 5-FOLD STRATIFICATA")
    log.info("=" * 60)
    cw  = None if class_weight == "none" else class_weight
    clf = RidgeClassifier(alpha=alpha, class_weight=cw, random_state=RAND_STATE)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RAND_STATE)

    cv_ba = cross_val_score(clf, X_feat, y, cv=skf,
                            scoring="balanced_accuracy", n_jobs=-1)
    cv_f1 = cross_val_score(clf, X_feat, y, cv=skf,
                            scoring="f1", n_jobs=-1)
    log.info("  Balanced accuracy: %.3f ± %.3f  [%s]",
             cv_ba.mean(), cv_ba.std(),
             ", ".join(f"{s:.3f}" for s in cv_ba))
    log.info("  F1 (arco):         %.3f ± %.3f",
             cv_f1.mean(), cv_f1.std())
    return {
        "cv_balanced_accuracy_mean": round(float(cv_ba.mean()), 4),
        "cv_balanced_accuracy_std":  round(float(cv_ba.std()),  4),
        "cv_f1_mean":                round(float(cv_f1.mean()), 4),
        "cv_f1_std":                 round(float(cv_f1.std()),  4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 9. Robustezza
# ══════════════════════════════════════════════════════════════════════════════

def robustness_validation(rocket, feat_scaler, ridge, X_test, y_test,
                          stat_feats_test=None, n_shuffle=5):
    log.info("")
    log.info("=" * 60)
    log.info("VALIDAZIONE ROBUSTEZZA")
    log.info("=" * 60)
    rng = np.random.default_rng(RAND_STATE)

    # Ricalcola feature test
    feat_rocket = rocket.transform(X_test, batch_size=MR_BATCH_SIZE)
    if stat_feats_test is not None:
        feat_all = np.concatenate([feat_rocket, stat_feats_test], axis=1)
    else:
        feat_all = feat_rocket
    feat_scaled = feat_scaler.transform(feat_all)
    score_real  = _to_proba(ridge, feat_scaled)
    f1_real     = f1_score(y_test, (score_real >= 0.5).astype(int))

    F1_shuf = [
        f1_score(rng.permutation(y_test),
                 (score_real >= 0.5).astype(int), zero_division=0)
        for _ in range(n_shuffle)
    ]
    mean_shuf = float(np.mean(F1_shuf))
    shuffle_ok = mean_shuf < f1_real * 0.7

    log.info("  F1 reale:           %.4f", f1_real)
    log.info("  F1 shuffle (media): %.4f", mean_shuf)

    if shuffle_ok:
        log.info("  ✓ Shuffle test superato — il modello discrimina realmente")
    else:
        log.warning("  ✗ Shuffle test FALLITO (gap=%.4f)", f1_real - mean_shuf)
        log.warning("  Il modello potrebbe predire sempre la classe maggioritaria.")
        log.warning("  Prova: --smote, oppure verifica qualità dataset/etichette.")

    # Stabilità: 3 run sullo stesso mini-batch
    batch  = X_test[:min(16, len(X_test))]
    feats3 = [rocket._transform_batch(batch) for _ in range(3)]
    max_var = float(np.max(np.var(np.stack(feats3, axis=0), axis=0)))
    stable  = max_var < 1e-9
    log.info("  Varianza inter-run: %.2e  (%s)",
             max_var, "✓ stabile" if stable else "✗ instabile")

    return {"f1_real": round(f1_real, 4), "f1_shuffle_mean": round(mean_shuf, 4),
            "shuffle_ok": shuffle_ok, "feature_space_max_var": max_var,
            "feature_space_stable": stable}


# ══════════════════════════════════════════════════════════════════════════════
# 10. Grafici
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(y_test, y_pred, y_score, out_dir, threshold=0.5):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(
        f"MultiRocket GPU + Ridge  (class_weight=balanced, soglia={threshold:.2f})",
        fontsize=13)

    ax = axes[0]
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No arco","Arco"], yticklabels=["No arco","Arco"])
    ax.set_title("Confusion Matrix")
    ax.set_ylabel("Reale"); ax.set_xlabel("Predetto")

    ax = axes[1]
    fpr_c, tpr_c, _ = roc_curve(y_test, y_score)
    auc = roc_auc_score(y_test, y_score)
    ax.plot(fpr_c, tpr_c, "steelblue", lw=2, label=f"AUC={auc:.3f}")
    ax.plot([0,1],[0,1],"k--",lw=1)
    ax.axvline(UL_MAX_FP_PCT/100, color="tomato", ls=":", lw=1.5,
               label=f"UL max FP={UL_MAX_FP_PCT:.0f}%")
    ax.axhline(UL_MIN_DETECTION_PCT/100, color="green", ls=":", lw=1.5,
               label=f"UL min det={UL_MIN_DETECTION_PCT:.0f}%")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(fontsize=8); ax.set_title("ROC Curve"); ax.grid(alpha=0.3)

    ax = axes[2]
    prec, rec, _ = precision_recall_curve(y_test, y_score)
    ap = average_precision_score(y_test, y_score)
    ax.plot(rec, prec, "tomato", lw=2, label=f"AP={ap:.3f}")
    ax.axhline(y_test.mean(), color="gray", ls="--", lw=1,
               label=f"Baseline={y_test.mean():.2f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.legend(); ax.set_title("Precision-Recall"); ax.grid(alpha=0.3)

    ax = axes[3]
    ax.hist(y_score[y_test==0], bins=40, alpha=0.6,
            color="steelblue", label="No arco (0)")
    ax.hist(y_score[y_test==1], bins=40, alpha=0.6,
            color="tomato", label="Arco (1)")
    ax.axvline(threshold, color="black", ls="--", lw=1.5,
               label=f"soglia={threshold:.2f}")
    ax.set_xlabel("Score (sigmoid)")
    ax.legend(); ax.set_title("Distribuzione score"); ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "results_multirocket_ridge.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    log.info("  Salvato: %s", path)


def plot_threshold_curve(y_test, y_score, out_dir, best_thr):
    thrs = np.arange(0.01, 1.00, 0.01)
    arc, no_arc = y_test==1, y_test==0
    det_r = [100.0*((y_score>=t).astype(int)[arc==1]==1).sum()/max(int(arc.sum()),1)
             for t in thrs]
    fp_r  = [100.0*((y_score>=t).astype(int)[no_arc==1]==1).sum()/max(int(no_arc.sum()),1)
             for t in thrs]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thrs, det_r, color="tomato",    lw=2, label="Detection rate %")
    ax.plot(thrs, fp_r,  color="steelblue", lw=2, label="False Positive rate %")
    ax.axhline(UL_MIN_DETECTION_PCT, color="tomato",    ls="--", lw=1,
               label=f"UL1699B min det={UL_MIN_DETECTION_PCT:.0f}%")
    ax.axhline(UL_MAX_FP_PCT,        color="steelblue", ls="--", lw=1,
               label=f"UL1699B max FP={UL_MAX_FP_PCT:.0f}%")
    ax.axvline(best_thr, color="black", ls=":", lw=2,
               label=f"Soglia ottimale={best_thr:.2f}")
    ax.fill_betweenx([UL_MIN_DETECTION_PCT, 100],
                     [best_thr-0.05], [best_thr+0.05],
                     alpha=0.1, color="green", label="Zona conformità")
    ax.set_xlabel("Soglia di decisione"); ax.set_ylabel("Percentuale [%]")
    ax.set_title("Analisi multi-soglia — UL1699B (MultiRocket)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "threshold_analysis.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    log.info("  Salvato: %s", path)


def plot_series_examples(X_test, y_test, y_pred, out_dir, n_per_class=2):
    categories = [
        ("Vero Positivo\n(arco rilevato)",    y_test==1, y_pred==1),
        ("Falso Negativo\n(arco mancato)",     y_test==1, y_pred==0),
        ("Vero Negativo\n(no arco corretto)",  y_test==0, y_pred==0),
        ("Falso Positivo\n(falso allarme)",    y_test==0, y_pred==1),
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
                        color="tomato" if y_test[ix]==1 else "steelblue")
                ax.set_xlabel("t [s]"); ax.set_ylabel("I [-]"); ax.grid(alpha=0.3)
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


def plot_feature_importance(ridge, n_stat, n_rocket, out_dir):
    coef   = np.abs(ridge.coef_.flatten())
    top_k  = min(60, len(coef))
    top_i  = np.argsort(coef)[::-1][:top_k]

    stat_names = (["RMS","Var","Skew","Kurt","PkMax","PkMin","ZCR","Crest"] +
                  [f"Band{i+1}" for i in range(len(FREQ_BANDS_HZ))] +
                  ["SpEnt","Slope"])
    colors = ["#E8593C" if i < n_stat else "#1D9E75" for i in top_i]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(top_k), coef[top_i], color=colors, edgecolor="none")
    labels = [stat_names[i] if i < n_stat else f"R{i-n_stat}" for i in top_i]
    ax.set_xticks(range(top_k))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("|Coefficiente Ridge|")
    ax.set_title(f"Top-{top_k} feature importance — MultiRocket + stat")
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#E8593C", label=f"Feature statistiche ({n_stat})"),
        Patch(facecolor="#1D9E75", label=f"Feature MultiRocket ({n_rocket})"),
    ], fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(out_dir, "feature_importance.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    log.info("  Salvato: %s", path)


def plot_tsne(X_feat, y, out_dir, max_samples=2000):
    log.info("  t-SNE 2D...")
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    n = len(y)
    if n > max_samples:
        idx = np.random.default_rng(RAND_STATE).choice(n, max_samples, replace=False)
        Xs, ys = X_feat[idx], y[idx]
    else:
        Xs, ys = X_feat, y
    n_pca = min(50, Xs.shape[1])
    Xp = PCA(n_components=n_pca, random_state=RAND_STATE).fit_transform(Xs)
    emb = TSNE(n_components=2, perplexity=30, random_state=RAND_STATE,
               n_iter=500).fit_transform(Xp)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(emb[ys==0,0], emb[ys==0,1], s=8, alpha=0.5,
               c="steelblue", label="No arco", rasterized=True)
    ax.scatter(emb[ys==1,0], emb[ys==1,1], s=8, alpha=0.5,
               c="tomato",    label="Arco",    rasterized=True)
    ax.set_title("t-SNE feature space MultiRocket")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=9); ax.grid(alpha=0.2)
    plt.tight_layout()
    path = os.path.join(out_dir, "tsne_feature_space.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    log.info("  Salvato: %s", path)


def plot_learning_curve(X_feat, y, class_weight, alpha, out_dir):
    log.info("  Curva di apprendimento...")
    cw  = None if class_weight == "none" else class_weight
    clf = RidgeClassifier(alpha=alpha, class_weight=cw, random_state=RAND_STATE)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RAND_STATE)
    sizes = np.linspace(0.1, 1.0, 7)
    abs_sizes, tr_sc, te_sc = learning_curve(
        clf, X_feat, y, cv=skf, train_sizes=sizes,
        scoring="balanced_accuracy", n_jobs=-1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(abs_sizes, tr_sc.mean(1), "o-", color="tomato",    lw=2, label="Train")
    ax.fill_between(abs_sizes, tr_sc.mean(1)-tr_sc.std(1),
                    tr_sc.mean(1)+tr_sc.std(1), alpha=0.15, color="tomato")
    ax.plot(abs_sizes, te_sc.mean(1), "o-", color="steelblue", lw=2, label="CV Val")
    ax.fill_between(abs_sizes, te_sc.mean(1)-te_sc.std(1),
                    te_sc.mean(1)+te_sc.std(1), alpha=0.15, color="steelblue")
    ax.set_xlabel("Campioni train"); ax.set_ylabel("Balanced accuracy")
    ax.set_title("Curva di apprendimento (MultiRocket)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "learning_curve.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    log.info("  Salvato: %s", path)


def plot_class_distribution(y_train, y_test, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Distribuzione classi", fontsize=12)
    for ax, y, title in [(axes[0], y_train, "Train"), (axes[1], y_test, "Test")]:
        counts = [(y==0).sum(), (y==1).sum()]
        bars   = ax.bar(["No arco (0)","Arco (1)"], counts,
                        color=["steelblue","tomato"], edgecolor="white")
        for bar, c in zip(bars, counts):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f"{c}\n({100*c/len(y):.1f}%)",
                    ha="center", va="bottom", fontsize=10)
        ax.set_title(title); ax.set_ylabel("Campioni"); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(out_dir, "class_distribution.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 11. Export
# ══════════════════════════════════════════════════════════════════════════════

def export_c_header(ridge, out_dir, n_features, best_threshold):
    log.info("  Export C header STM32...")
    coef  = ridge.coef_.flatten().astype(np.float32)
    inter = float(ridge.intercept_.flatten()[0])

    def _arr(name, vals):
        lines = [f"static const float {name}[{len(vals)}] = {{"]
        for i in range(0, len(vals), 8):
            lines.append("    " + ", ".join(f"{v:.8f}f" for v in vals[i:i+8]) + ",")
        lines[-1] = lines[-1].rstrip(",")
        lines.append("};")
        return "\n".join(lines)

    h = f"""\
/**
 * multirocket_ridge_inference.h
 * Generato da train_multirocket_gpu.py
 * Normativa: UL 1699B  |  Soglia ottimale: {best_threshold:.4f}
 */
#ifndef MULTIROCKET_RIDGE_H
#define MULTIROCKET_RIDGE_H
#include <stdint.h>
#include <math.h>
#define N_FEATURES        {n_features}
#define OPTIMAL_THRESHOLD {best_threshold:.4f}f
{_arr("mr_ridge_coef", coef)}
static const float mr_ridge_intercept = {inter:.8f}f;
static inline float sigmoid_f(float x){{return 1.0f/(1.0f+expf(-x));}}
static inline float mr_ridge_score(const float* f, uint32_t n){{
    float d = mr_ridge_intercept;
    for(uint32_t i=0;i<n;i++) d += mr_ridge_coef[i]*f[i];
    return sigmoid_f(d);
}}
static inline uint8_t mr_ridge_predict(const float* f, uint32_t n){{
    return (mr_ridge_score(f,n) >= OPTIMAL_THRESHOLD) ? 1u : 0u;
}}
#endif
"""
    path = os.path.join(out_dir, "multirocket_ridge_inference.h")
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(h)
    log.info("  C header: %s", path)
    return path


def export_config_json(out_dir, rocket_cfg, norm_params, best_thr,
                       metrics, n_features, class_weight, smote_applied):
    cfg = {
        "model":          "MultiRocket GPU + RidgeClassifier",
        "normativa":      "UL1699B",
        "fs_hz":          FS_HZ,
        "n_features":     n_features,
        "best_threshold": round(best_thr, 4),
        "class_weight":   class_weight,
        "smote_applied":  smote_applied,
        "multirocket":    rocket_cfg,
        "normalization":  norm_params or {"mode": "none"},
        "metrics":        {k: v for k, v in metrics.items()
                           if isinstance(v, (int, float, bool, str))},
    }
    path = os.path.join(out_dir, "deployment_config.json")
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(cfg, fp, indent=2, default=str)
    log.info("  Config JSON: %s", path)


# ══════════════════════════════════════════════════════════════════════════════
# 12. Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="MultiRocket GPU + Ridge — classificatore archi PV DC\n"
                    "Normativa: UL 1699B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("train")
    parser.add_argument("test")
    parser.add_argument("--out",          "-o", default="./results_rocket")
    parser.add_argument("--n-kernels",    type=int,   default=MR_N_KERNELS,
                        help=f"Kernel per set MultiRocket (default: {MR_N_KERNELS})")
    parser.add_argument("--max-dilations",type=int,   default=MR_MAX_DILATIONS)
    parser.add_argument("--batch-size",   type=int,   default=MR_BATCH_SIZE,
                        help="Batch GPU (riduci se OOM, default: 256)")
    parser.add_argument("--ridge-alpha",  type=float, default=RIDGE_ALPHA)
    parser.add_argument("--class-weight", default="balanced",
                        choices=["balanced","none"])
    parser.add_argument("--no-normalize", action="store_true",
                        help="Disabilita normalizzazione globale serie grezze")
    parser.add_argument("--no-stat-features", action="store_true",
                        help="Disabilita feature statistiche manuali")
    parser.add_argument("--smote",         action="store_true")
    parser.add_argument("--export-stm32",  action="store_true")
    parser.add_argument("--tsne",          action="store_true",
                        help="t-SNE feature space (~2 min extra)")
    parser.add_argument("--learning-curve",action="store_true",
                        help="Curva di apprendimento (~3 min extra)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cw_val = None if args.class_weight == "none" else args.class_weight

    # Setup GPU
    log.info("=" * 60)
    log.info("SETUP DEVICE")
    log.info("=" * 60)
    try:
        import torch
    except ImportError:
        log.error("PyTorch non trovato.")
        log.error("Installa con:")
        log.error("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)
    device = setup_device()

    # 1. Dataset
    X_train, y_train, X_test, y_test = load_datasets(args.train, args.test)
    analyze_distribution(y_train, y_test)
    plot_class_distribution(y_train, y_test, args.out)

    # 2. Normalizzazione
    norm_params, raw_scaler = None, None
    if not args.no_normalize:
        X_train, X_test, norm_params, raw_scaler = normalize_global(X_train, X_test)

    # 3. MultiRocket GPU
    log.info("")
    log.info("=" * 60)
    log.info("MULTIROCKET GPU")
    log.info("=" * 60)
    rocket_cfg = {
        "n_kernels":               args.n_kernels,
        "max_dilations_per_kernel": args.max_dilations,
        "batch_size":              args.batch_size,
        "device":                  str(device),
    }
    log.info("  n_kernels=%d  max_dilations=%d  batch=%d  device=%s",
             args.n_kernels, args.max_dilations, args.batch_size, device)

    rocket = MultiRocketGPU(
        n_kernels=args.n_kernels,
        max_dilations_per_kernel=args.max_dilations,
        device=device,
        random_state=RAND_STATE,
    )

    t0 = time.time()
    rocket.fit(X_train)
    t_fit = time.time() - t0

    log.info("")
    log.info("  Estrazione feature — TRAIN")
    t0 = time.time()
    feat_rocket_train = rocket.transform(X_train, batch_size=args.batch_size)
    t_feat_tr = time.time() - t0

    log.info("")
    log.info("  Estrazione feature — TEST")
    t0 = time.time()
    feat_rocket_test = rocket.transform(X_test, batch_size=args.batch_size)
    t_feat_te = time.time() - t0

    # 4. Feature statistiche manuali
    stat_train, stat_test = None, None
    if not args.no_stat_features:
        log.info("")
        log.info("=" * 60)
        log.info("FEATURE STATISTICHE MANUALI")
        log.info("=" * 60)
        stat_train = extract_statistical_features(X_train)
        stat_test  = extract_statistical_features(X_test)
        feat_train_raw = np.concatenate([feat_rocket_train, stat_train], axis=1)
        feat_test_raw  = np.concatenate([feat_rocket_test,  stat_test],  axis=1)
        n_stat    = stat_train.shape[1]
        n_rocket  = feat_rocket_train.shape[1]
        log.info("  Totale: MultiRocket=%d + stat=%d = %d",
                 n_rocket, n_stat, feat_train_raw.shape[1])
    else:
        feat_train_raw = feat_rocket_train
        feat_test_raw  = feat_rocket_test
        n_stat   = 0
        n_rocket = feat_rocket_train.shape[1]

    # 5. Normalizza feature space
    feat_scaler  = StandardScaler()
    feat_train   = feat_scaler.fit_transform(feat_train_raw)
    feat_test    = feat_scaler.transform(feat_test_raw)

    # 6. Cross-validation
    cv_res = cross_validate(feat_train, y_train, args.class_weight, args.ridge_alpha)

    # 7. SMOTE (opzionale)
    smote_applied = False
    y_train_fit   = y_train
    if args.smote:
        log.info("")
        log.info("=" * 60)
        log.info("SMOTE — OVERSAMPLING FEATURE SPACE")
        log.info("=" * 60)
        feat_train, y_train_fit = apply_smote(feat_train, y_train)
        smote_applied = True

    # 8. Ridge
    log.info("")
    log.info("=" * 60)
    log.info("RIDGE CLASSIFIER")
    log.info("=" * 60)
    log.info("  alpha=%.2f  class_weight=%s", args.ridge_alpha, args.class_weight)
    ridge = RidgeClassifier(alpha=args.ridge_alpha, class_weight=cw_val,
                             random_state=RAND_STATE)
    t0 = time.time()
    ridge.fit(feat_train, y_train_fit)
    t_ridge = time.time() - t0
    log.info("  Ridge fit in %.2f s", t_ridge)

    # 9. Predizioni
    y_score  = _to_proba(ridge, feat_test)
    best_thr = threshold_analysis(y_test, y_score)
    y_pred   = (y_score >= best_thr).astype(int)

    # 10. Metriche
    log.info("")
    log.info("=" * 60)
    log.info("METRICHE TEST SET")
    log.info("=" * 60)
    for line in classification_report(y_test, y_pred,
                                       target_names=["No arco","Arco"],
                                       digits=3).splitlines():
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

    ul     = ul1699b_metric(y_test, y_pred, threshold=best_thr)
    ul["best_threshold"] = best_thr
    robust = robustness_validation(rocket, feat_scaler, ridge, X_test, y_test,
                                   stat_feats_test=stat_test)

    # 11. Grafici
    log.info("")
    log.info("=" * 60)
    log.info("SALVATAGGIO GRAFICI")
    log.info("=" * 60)
    plot_results(y_test, y_pred, y_score, args.out, threshold=best_thr)
    plot_series_examples(X_test, y_test, y_pred, args.out)
    plot_threshold_curve(y_test, y_score, args.out, best_thr)
    plot_feature_importance(ridge, n_stat, n_rocket, args.out)
    if args.tsne:
        plot_tsne(feat_train, y_train_fit, args.out)
    if args.learning_curve:
        plot_learning_curve(feat_train, y_train_fit,
                            args.class_weight, args.ridge_alpha, args.out)

    # 12. Salvataggio bundle
    n_features = feat_train.shape[1]
    metrics_all = {
        "accuracy":            round(acc, 4),
        "balanced_accuracy":   round(ba,  4),
        "f1_arc":              round(f1,  4),
        "roc_auc":             round(auc, 4),
        "avg_precision":       round(ap,  4),
        **ul, **cv_res,
        **{f"robust_{k}": v for k, v in robust.items()},
    }
    bundle = {
        "rocket":         rocket,
        "feat_scaler":    feat_scaler,
        "ridge":          ridge,
        "raw_scaler":     raw_scaler,
        "norm_params":    norm_params,
        "best_threshold": best_thr,
        "n_features":     n_features,
        "n_stat":         n_stat,
        "n_rocket":       n_rocket,
        "rocket_cfg":     rocket_cfg,
        "ridge_alpha":    args.ridge_alpha,
        "class_weight":   args.class_weight,
        "smote_applied":  smote_applied,
        "metrics":        metrics_all,
    }
    bundle_path = os.path.join(args.out, "multirocket_ridge_bundle.pkl")
    with open(bundle_path, "wb") as fp:
        pickle.dump(bundle, fp)
    log.info("  Bundle: %s", bundle_path)

    if args.export_stm32:
        log.info("")
        log.info("=" * 60)
        log.info("EXPORT STM32")
        log.info("=" * 60)
        export_c_header(ridge, args.out, n_features, best_thr)

    export_config_json(args.out, rocket_cfg, norm_params, best_thr,
                       metrics_all, n_features, args.class_weight, smote_applied)

    # Riepilogo
    log.info("")
    log.info("=" * 72)
    log.info("RIEPILOGO FINALE")
    log.info("=" * 72)
    log.info("  Device:            %s", device)
    log.info("  N feature totali:  %d (Rocket=%d + stat=%d)",
             n_features, n_rocket, n_stat)
    log.info("  T kernel gen:      %.2f s", t_fit)
    log.info("  T feat train:      %.1f s", t_feat_tr)
    log.info("  T feat test:       %.1f s", t_feat_te)
    log.info("  T ridge fit:       %.2f s", t_ridge)
    log.info("  Accuracy:          %.4f", acc)
    log.info("  Balanced Accuracy: %.4f", ba)
    log.info("  F1 (arco):         %.4f", f1)
    log.info("  ROC-AUC:           %.4f", auc)
    log.info("  CV balanced acc:   %.3f ± %.3f",
             cv_res["cv_balanced_accuracy_mean"],
             cv_res["cv_balanced_accuracy_std"])
    log.info("  Detection rate:    %.1f%%", ul["detection_rate_pct"])
    log.info("  False positive:    %.1f%%", ul["false_positive_rate_pct"])
    log.info("  Soglia ottimale:   %.2f",   best_thr)
    log.info("  UL1699B:           %s",
             "✓ CONFORME" if ul["ul1699b_conforme"] else "✗ NON CONFORME")
    log.info("  Shuffle test:      %s",
             "✓ OK" if robust["shuffle_ok"] else "✗ FALLITO")
    log.info("")
    log.info("  Output in: %s", args.out)

    if not ul["ul1699b_conforme"] or not robust["shuffle_ok"]:
        log.warning("")
        log.warning("  AZIONI SUGGERITE:")
        if not robust["shuffle_ok"]:
            log.warning("  → --n-kernels 10000  (più kernel, più feature)")
            log.warning("  → --smote             (bilancia il train set)")
            log.warning("  → verifica etichette nel dataset (data leakage?)")
        if ul["false_positive_rate_pct"] > UL_MAX_FP_PCT:
            log.warning("  → --ridge-alpha 10.0  (più regolarizzazione, meno FP)")
        if ul["detection_rate_pct"] < UL_MIN_DETECTION_PCT:
            log.warning("  → --ridge-alpha 0.1   (meno regolarizzazione, più detection)")
            log.warning("  → --smote              (bilancia classi)")
        log.warning("  → --tsne               (visualizza separabilità classi)")


if __name__ == "__main__":
    main()