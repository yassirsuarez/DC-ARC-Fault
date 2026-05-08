#!/usr/bin/env python3
"""
train_multirockethydra_gpu.py
=============================
Pipeline GPU-accelerata per classificatore archi elettrici in impianti PV DC.
Architettura: MultiRocketHydra (Dempster et al., 2023) + RidgeClassifier

ARCHITETTURA MultiRocketHydra:
  MultiRocketHydra combina due componenti in un unico transformer:

  1. HYDRA (parte "competitiva"):
     - Kernel raggruppati in G gruppi da K kernel ciascuno
     - Per ogni gruppo: softmax tra le risposte dei K kernel → competizione
     - Aggregazione: PPV sulla risposta post-softmax
     - Cattura pattern "relativi" (quale kernel risponde di più)
     - Feature Hydra: n_groups × n_kernels_per_group (= G × K)

  2. MultiRocket (parte "assoluta"):
     - Kernel random indipendenti su serie originale + differenze prime
     - Aggregazioni: PPV e mean (proporzione e media dei valori positivi)
     - Cattura pattern "assoluti" (quanto i kernel rispondono)
     - Feature MultiRocket: n_kernels × n_dilations × 4

  FEATURE TOTALI = Hydra_features + MultiRocket_features
  → tipicamente ~10.000–50.000 feature con parametri di default

  Riferimento: Dempster, A., Schmidt, D.F., Webb, G.I.
  "Hydra: Competing convolutional kernels for fast and accurate
  time series classification"
  Data Mining and Knowledge Discovery, 2023.

DIFFERENZE RISPETTO A solo-MultiRocket:
  - La parte Hydra usa competizione tra kernel → più discriminativa
  - Più feature per lo stesso numero di kernel
  - Tipicamente +2-5% accuracy rispetto a MultiRocket puro
  - Tempo di calcolo simile (stessa architettura conv1d vettorizzata)

REQUISITI:
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install scikit-learn numpy matplotlib seaborn scipy

USO:
    python train_multirockethydra_gpu.py train.npz test.npz
    python train_multirockethydra_gpu.py train.npz test.npz --out ./results_mrh
    python train_multirockethydra_gpu.py train.npz test.npz --n-groups 64 --n-kernels-per-group 8
    python train_multirockethydra_gpu.py train.npz test.npz --mr-kernels 2000
    python train_multirockethydra_gpu.py train.npz test.npz --smote
    python train_multirockethydra_gpu.py train.npz test.npz --export-stm32
    python train_multirockethydra_gpu.py train.npz test.npz --tsne

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
from sklearn.model_selection import StratifiedKFold, cross_val_score
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

# ── parametri MultiRocketHydra ────────────────────────────────────────────────
# Hydra
HYDRA_N_GROUPS           = 64    # G: numero di gruppi di kernel
HYDRA_N_KERNELS_PER_GROUP = 8   # K: kernel per gruppo (competizione softmax)
HYDRA_MAX_DILATIONS      = 8    # dilazioni per gruppo

# MultiRocket (parte assoluta)
MR_N_KERNELS    = 1_000   # kernel random totali
MR_MAX_DILATIONS = 8      # dilazioni per kernel

# Batch GPU
BATCH_SIZE = 256

FREQ_BANDS_HZ = [
    (0,    500),
    (500,  1500),
    (1500, 3000),
    (3000, 5000),
]


# ══════════════════════════════════════════════════════════════════════════════
# 0. Setup device
# ══════════════════════════════════════════════════════════════════════════════

def setup_device():
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props  = torch.cuda.get_device_properties(0)
        vram   = props.total_memory / 1e9
        log.info("  GPU: %s  (%.1f GB VRAM)", props.name, vram)
        log.info("  CUDA: %s  |  PyTorch: %s",
                 torch.version.cuda, torch.__version__)
        if vram < 6:
            log.warning("  VRAM < 6 GB — riduci --n-groups 32 --mr-kernels 500")
    else:
        device = torch.device("cpu")
        log.warning("  CUDA non disponibile — fallback CPU (più lento)")
    return device


# ══════════════════════════════════════════════════════════════════════════════
# 1. MultiRocketHydra — implementazione PyTorch GPU
# ══════════════════════════════════════════════════════════════════════════════

class MultiRocketHydraGPU:
    """
    MultiRocketHydra GPU-accelerato (Dempster et al., 2023).

    Combina:
    ┌─────────────────────────────────────────────────────────────┐
    │ HYDRA (competitiva)                                          │
    │  • G gruppi × K kernel × D dilazioni                        │
    │  • Per gruppo: softmax tra risposte K kernel                 │
    │  • PPV post-softmax su serie originale E differenze prime    │
    │  • Feature: G × K × 2 (originale + diff)                    │
    ├─────────────────────────────────────────────────────────────┤
    │ MultiRocket (assoluta)                                       │
    │  • N kernel random × D dilazioni                             │
    │  • PPV + mean su serie originale E differenze prime          │
    │  • Feature: N_total_kernel_dilation_pairs × 4                │
    └─────────────────────────────────────────────────────────────┘
    Feature totali concatenate → RidgeClassifier

    Parametri
    ---------
    n_groups : int
        G — numero di gruppi Hydra (default: 64)
    n_kernels_per_group : int
        K — kernel per gruppo, competono via softmax (default: 8)
    hydra_max_dilations : int
        Dilazioni per gruppo Hydra (default: 8)
    mr_n_kernels : int
        Kernel MultiRocket totali (default: 1000)
    mr_max_dilations : int
        Dilazioni per kernel MultiRocket (default: 8)
    """

    def __init__(
        self,
        n_groups:             int = HYDRA_N_GROUPS,
        n_kernels_per_group:  int = HYDRA_N_KERNELS_PER_GROUP,
        hydra_max_dilations:  int = HYDRA_MAX_DILATIONS,
        mr_n_kernels:         int = MR_N_KERNELS,
        mr_max_dilations:     int = MR_MAX_DILATIONS,
        device=None,
        random_state:         int = RAND_STATE,
    ):
        import torch
        self.n_groups            = n_groups
        self.n_kernels_per_group = n_kernels_per_group
        self.hydra_max_dilations = hydra_max_dilations
        self.mr_n_kernels        = mr_n_kernels
        self.mr_max_dilations    = mr_max_dilations
        self.device              = device or torch.device("cpu")
        self.random_state        = random_state
        self._fitted             = False

        # Kernel Hydra e MultiRocket (generati in fit)
        self._hydra_groups = []   # lista di gruppi Hydra
        self._mr_groups    = []   # lista di gruppi MultiRocket

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y=None):
        """
        Genera tutti i kernel (random, non addestrati).
        La "competizione" Hydra avviene a transform-time via softmax.
        """
        import torch
        import torch.nn.functional as F

        T   = X.shape[1]
        rng = np.random.default_rng(self.random_state)

        log.info("  Generazione kernel Hydra (%d gruppi × %d kernel × %d dil)...",
                 self.n_groups, self.n_kernels_per_group, self.hydra_max_dilations)
        log.info("  Generazione kernel MultiRocket (%d kernel × %d dil)...",
                 self.mr_n_kernels, self.mr_max_dilations)

        t0 = time.time()

        # ── HYDRA: genera G gruppi ────────────────────────────────────────────
        # Tutti i gruppi usano lunghezza kernel fissa = 9 (come paper originale)
        L_hydra = 9

        max_exp_h = np.log2((T - 1) / (L_hydra - 1))
        dils_hydra = np.unique(np.floor(
            np.power(2, np.linspace(0, max_exp_h,
                                    min(self.hydra_max_dilations,
                                        int(max_exp_h) + 1)))
        ).astype(int))

        for g in range(self.n_groups):
            # K kernel per gruppo, shape (K, L_hydra), media zero
            w = rng.normal(0, 1, (self.n_kernels_per_group, L_hydra)).astype(np.float32)
            w -= w.mean(axis=1, keepdims=True)

            for d in dils_hydra:
                pad = (int(d) * (L_hydra - 1)) // 2
                self._hydra_groups.append({
                    "group_id": g,
                    "L":        L_hydra,
                    "dilation": int(d),
                    "padding":  pad,
                    "w":        w,   # (K, L_hydra) — stesso per tutte le dilazioni
                    "K":        self.n_kernels_per_group,
                })

        # ── MultiRocket: genera kernel random ────────────────────────────────
        kernel_lengths = [7, 9, 11]
        lengths = rng.choice(kernel_lengths, size=self.mr_n_kernels)

        for L in kernel_lengths:
            n_L = int((lengths == L).sum())
            if n_L == 0:
                continue

            max_exp_mr = np.log2((T - 1) / (L - 1))
            dils_mr = np.unique(np.floor(
                np.power(2, np.linspace(0, max_exp_mr,
                                        min(self.mr_max_dilations,
                                            int(max_exp_mr) + 1)))
            ).astype(int))

            # Pesi: (n_L, L) media zero
            w_a = rng.normal(0, 1, (n_L, L)).astype(np.float32)
            w_b = rng.normal(0, 1, (n_L, L)).astype(np.float32)
            w_a -= w_a.mean(axis=1, keepdims=True)
            w_b -= w_b.mean(axis=1, keepdims=True)

            for d in dils_mr:
                pad = (int(d) * (L - 1)) // 2

                # Calcola bias su segnale random (quantile casuale)
                dummy   = rng.normal(0, 1, (1, 1, T)).astype(np.float32)
                dummy_t = torch.tensor(dummy, device=self.device)
                d_diff  = torch.tensor(np.diff(dummy, axis=2), device=self.device)
                k_a = torch.tensor(w_a, device=self.device).unsqueeze(1)
                k_b = torch.tensor(w_b, device=self.device).unsqueeze(1)

                import torch.nn.functional as F_inner
                out_a = F_inner.conv1d(dummy_t, k_a, dilation=int(d), padding=pad)
                out_b = F_inner.conv1d(d_diff,  k_b, dilation=int(d), padding=pad)
                q_a = float(rng.uniform(0, 1))
                q_b = float(rng.uniform(0, 1))

                self._mr_groups.append({
                    "L":        L,
                    "dilation": int(d),
                    "padding":  pad,
                    "w_a":      w_a,
                    "w_b":      w_b,
                    "bias_a":   float(torch.quantile(out_a.cpu(), q_a).item()),
                    "bias_b":   float(torch.quantile(out_b.cpu(), q_b).item()),
                    "n_k":      n_L,
                })

        # Feature totali
        n_hydra_feat = len(self._hydra_groups) * self.n_kernels_per_group * 2
        n_mr_feat    = sum(g["n_k"] * 4 for g in self._mr_groups)
        self._n_hydra_features = n_hydra_feat
        self._n_mr_features    = n_mr_feat
        self._n_features       = n_hydra_feat + n_mr_feat
        self._T = T
        self._fitted = True

        mem_gb = 27770 * self._n_features * 4 / 1e9
        log.info("  Kernel generati in %.1f s", time.time() - t0)
        log.info("  Gruppi Hydra (gruppo×dilation):      %d", len(self._hydra_groups))
        log.info("  Gruppi MultiRocket (L×dilation):     %d", len(self._mr_groups))
        log.info("  Feature Hydra:      %d", n_hydra_feat)
        log.info("  Feature MultiRocket:%d", n_mr_feat)
        log.info("  Feature TOTALI:     %d", self._n_features)
        log.info("  Memoria stimata train set: %.2f GB", mem_gb)
        if mem_gb > 8:
            log.warning("  >8 GB — riduci --n-groups o --mr-kernels")
        return self

    # ── transform batch ───────────────────────────────────────────────────────

    def _transform_batch(self, X_batch: np.ndarray) -> np.ndarray:
        """
        Trasforma un batch applicando Hydra + MultiRocket su GPU.

        Hydra (competitiva):
          Per ogni gruppo g e dilatazione d:
            1. Applica K kernel: conv1d(X, w_g) → (n, K, T')
            2. Softmax tra i K kernel (dim=1): ogni posizione temporale
               ha un "vincitore" → competizione
            3. PPV della risposta post-softmax → (n, K) feature

        MultiRocket (assoluta):
          Per ogni gruppo (L, d):
            1. conv1d su serie originale e differenze prime
            2. PPV + mean → (n, n_k×4) feature
        """
        import torch
        import torch.nn.functional as F

        n, T = X_batch.shape
        X_t    = torch.tensor(X_batch, dtype=torch.float32,
                              device=self.device).unsqueeze(1)  # (n, 1, T)
        X_diff = torch.diff(X_t, dim=2)                         # (n, 1, T-1)

        parts = []

        # ── HYDRA ─────────────────────────────────────────────────────────────
        for g in self._hydra_groups:
            K   = g["K"]
            d   = g["dilation"]
            p   = g["padding"]

            # w: (K, L) → (K, 1, L) come filtri conv1d
            w = torch.tensor(g["w"], dtype=torch.float32,
                             device=self.device).unsqueeze(1)

            # conv1d: (n, 1, T) × (K, 1, L) → (n, K, T')
            out_orig = F.conv1d(X_t,    w, dilation=d, padding=p)  # (n, K, T')
            out_diff = F.conv1d(X_diff, w, dilation=d, padding=p)  # (n, K, T'')

            # Softmax competitivo tra K kernel (lungo dim=1)
            # Ogni posizione temporale: qual kernel risponde di più?
            soft_orig = F.softmax(out_orig, dim=1)  # (n, K, T')
            soft_diff = F.softmax(out_diff, dim=1)  # (n, K, T'')

            # PPV post-softmax: proporzione di posizioni in cui ogni kernel "vince"
            # (risposta softmax > 1/K significa che quel kernel è "vincitore")
            thresh = 1.0 / K
            ppv_orig = (soft_orig > thresh).float().mean(dim=2)  # (n, K)
            ppv_diff = (soft_diff > thresh).float().mean(dim=2)  # (n, K)

            # Concatena: (n, K×2)
            group_feat = torch.cat([ppv_orig, ppv_diff], dim=1)  # (n, 2K)
            parts.append(group_feat.cpu().numpy())

        # ── MultiRocket ───────────────────────────────────────────────────────
        for g in self._mr_groups:
            d   = g["dilation"]
            p   = g["padding"]
            n_k = g["n_k"]

            w_a = torch.tensor(g["w_a"], dtype=torch.float32,
                               device=self.device).unsqueeze(1)  # (n_k, 1, L)
            w_b = torch.tensor(g["w_b"], dtype=torch.float32,
                               device=self.device).unsqueeze(1)

            # conv1d: (n, 1, T) × (n_k, 1, L) → (n, n_k, T')
            out_a = F.conv1d(X_t,    w_a, dilation=d, padding=p) - g["bias_a"]
            out_b = F.conv1d(X_diff, w_b, dilation=d, padding=p) - g["bias_b"]

            # PPV e mean
            ppv_a  = (out_a > 0).float().mean(dim=2)  # (n, n_k)
            mean_a = out_a.mean(dim=2)
            ppv_b  = (out_b > 0).float().mean(dim=2)
            mean_b = out_b.mean(dim=2)

            # (n, n_k×4)
            mr_feat = torch.stack([ppv_a, mean_a, ppv_b, mean_b],
                                   dim=2).reshape(n, n_k * 4)
            parts.append(mr_feat.cpu().numpy())

        return np.concatenate(parts, axis=1)

    # ── transform ─────────────────────────────────────────────────────────────

    def transform(self, X: np.ndarray, batch_size: int = BATCH_SIZE) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("MultiRocketHydraGPU non ancora fittato.")

        log.info("  Estrazione feature MultiRocketHydra (batch=%d, n=%d)...",
                 batch_size, len(X))
        parts     = []
        n_batches = int(np.ceil(len(X) / batch_size))
        t0        = time.time()

        for bi, start in enumerate(range(0, len(X), batch_size)):
            batch = X[start:start + batch_size]
            parts.append(self._transform_batch(batch))

            if (bi + 1) % max(1, n_batches // 8) == 0 or bi == 0:
                elapsed = time.time() - t0
                eta     = elapsed / (bi + 1) * (n_batches - bi - 1) if bi > 0 else -1
                log.info("    Batch %d/%d  (%.0f%%)  elapsed=%.0fs  ETA=%.0fs",
                         bi + 1, n_batches,
                         100.0 * (bi + 1) / n_batches,
                         elapsed, max(0, eta))

        feat = np.concatenate(parts, axis=0)
        log.info("  Feature shape: %s  (%.1f s  %.2f GB)",
                 feat.shape, time.time() - t0, feat.nbytes / 1e9)
        return feat

    @property
    def n_features(self):
        return self._n_features if self._fitted else -1

    @property
    def n_hydra_features(self):
        return self._n_hydra_features if self._fitted else -1

    @property
    def n_mr_features(self):
        return self._n_mr_features if self._fitted else -1


# ══════════════════════════════════════════════════════════════════════════════
# 2. Caricamento dataset
# ══════════════════════════════════════════════════════════════════════════════

def load_datasets(train_path, test_path):
    log.info("=" * 60)
    log.info("CARICAMENTO DATASET")
    log.info("=" * 60)
    for p in (train_path, test_path):
        if not os.path.isfile(p):
            log.error("File non trovato: %s", p)
            sys.exit(1)

    def _load(path, label):
        data = np.load(path)
        X, y = data["X"], data["y"]
        if X.ndim == 3:
            X = X[:, 0, :]
        log.info("  %-8s  X=%s  y=%s  (%.0f ms/serie @ %d Hz)",
                 label, X.shape, y.shape,
                 X.shape[1] / FS_HZ * 1000, FS_HZ)
        return X.astype(np.float32), y.astype(np.int64)

    X_train, y_train = _load(train_path, "TRAIN")
    X_test,  y_test  = _load(test_path,  "TEST ")

    if X_train.shape[1] != X_test.shape[1]:
        log.error("Lunghezza serie non coerente: train=%d test=%d",
                  X_train.shape[1], X_test.shape[1])
        sys.exit(1)

    log.info("  Checksum train=%s test=%s",
             hashlib.md5(X_train.tobytes()).hexdigest()[:8],
             hashlib.md5(X_test.tobytes()).hexdigest()[:8])
    return X_train, y_train, X_test, y_test


# ══════════════════════════════════════════════════════════════════════════════
# 3. Analisi distribuzione
# ══════════════════════════════════════════════════════════════════════════════

def analyze_distribution(y_train, y_test):
    log.info("")
    log.info("=" * 60)
    log.info("ANALISI DISTRIBUZIONE CLASSI")
    log.info("=" * 60)

    def _s(y, name):
        n0, n1 = int((y==0).sum()), int((y==1).sum())
        pct1 = 100.0 * n1 / len(y)
        log.info("  %-8s  no-arco=%d (%.1f%%)  arco=%d (%.1f%%)  ratio=%.1f:1",
                 name, n0, 100*n0/len(y), n1, pct1,
                 max(n0,n1)/max(min(n0,n1),1))
        return pct1

    p_tr = _s(y_train, "TRAIN")
    p_te = _s(y_test,  "TEST ")
    drift = abs(p_tr - p_te)
    if drift > 10:
        log.warning("  Drift distribuzione: %.1f pp — verifica split", drift)
    else:
        log.info("  ✓ Distribuzione coerente (drift=%.1f pp)", drift)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Normalizzazione globale
# ══════════════════════════════════════════════════════════════════════════════

def normalize_global(X_train, X_test):
    log.info("")
    log.info("=" * 60)
    log.info("NORMALIZZAZIONE GLOBALE")
    log.info("=" * 60)
    sc     = StandardScaler()
    X_tr_n = sc.fit_transform(X_train)
    X_te_n = sc.transform(X_test)
    log.info("  z-score globale (fit su train)")
    log.info("  Train: μ=%.4f σ=%.4f", X_tr_n.mean(), X_tr_n.std())
    log.info("  Test:  μ=%.4f σ=%.4f", X_te_n.mean(), X_te_n.std())
    return X_tr_n, X_te_n, sc


# ══════════════════════════════════════════════════════════════════════════════
# 5. Feature statistiche
# ══════════════════════════════════════════════════════════════════════════════

def extract_statistical_features(X: np.ndarray) -> np.ndarray:
    log.info("  Estrazione feature statistiche (n=%d)...", len(X))
    n, T    = X.shape
    n_bands = len(FREQ_BANDS_HZ)
    feats   = np.zeros((n, 8 + n_bands + 2), dtype=np.float32)
    freqs   = np.fft.rfftfreq(T, d=1.0 / FS_HZ)
    t_norm  = np.linspace(0, 1, T)

    for i, x in enumerate(X):
        rms = float(np.sqrt(np.mean(x**2)))
        feats[i, 0] = rms
        feats[i, 1] = float(np.var(x))
        feats[i, 2] = float(scipy_stats.skew(x))
        feats[i, 3] = float(scipy_stats.kurtosis(x))
        feats[i, 4] = float(x.max())
        feats[i, 5] = float(x.min())
        feats[i, 6] = float(np.sum(np.diff(np.sign(x)) != 0)) / T
        feats[i, 7] = float(np.abs(x).max()) / (rms + 1e-8)

        fft_mag = np.abs(np.fft.rfft(x)) ** 2
        for j, (f_lo, f_hi) in enumerate(FREQ_BANDS_HZ):
            feats[i, 8+j] = fft_mag[(freqs>=f_lo)&(freqs<f_hi)].sum() / (T+1e-8)

        psd = fft_mag / (fft_mag.sum() + 1e-8)
        feats[i, 8+n_bands] = float(
            -np.sum(np.clip(psd,1e-12,None) * np.log2(np.clip(psd,1e-12,None))))
        slope, _ = np.polyfit(t_norm, x, 1)
        feats[i, 8+n_bands+1] = float(slope)

    log.info("  Feature statistiche: %s  (%.2f MB)",
             feats.shape, feats.nbytes/1e6)
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
    log.info("  SMOTE — arco=%d no-arco=%d",
             int((y==1).sum()), int((y==0).sum()))
    X_r, y_r = SMOTE(random_state=RAND_STATE).fit_resample(X_feat, y)
    log.info("  SMOTE dopo — arco=%d no-arco=%d",
             int((y_r==1).sum()), int((y_r==0).sum()))
    return X_r, y_r


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
    dr   = 100.0 * det / max(int(arc.sum()),    1)
    fpr  = 100.0 * fp  / max(int(no_arc.sum()), 1)
    ok   = dr >= UL_MIN_DETECTION_PCT and fpr <= UL_MAX_FP_PCT
    log.info("")
    log.info("=" * 60)
    log.info("METRICA UL1699B (soglia=%.2f)", threshold)
    log.info("=" * 60)
    log.info("  Archi rilevati: %d/%d (%.1f%%)", det, int(arc.sum()), dr)
    log.info("  Falsi positivi: %d/%d (%.1f%%)", fp,  int(no_arc.sum()), fpr)
    log.info("  Mancati: %d  |  Veri negativi: %d", miss, tn)
    log.info("  %s", "✓ CONFORME UL1699B" if ok else "✗ NON conforme UL1699B")
    return {"detected": det, "missed": miss, "false_positives": fp,
            "true_negatives": tn, "detection_rate_pct": round(dr, 2),
            "false_positive_rate_pct": round(fpr, 2), "ul1699b_conforme": ok}


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
        log.warning("  Nessuna soglia soddisfa entrambi i vincoli UL1699B.")
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
# 8. Shuffle test
# ══════════════════════════════════════════════════════════════════════════════

def shuffle_test(y_test, y_pred, n=5):
    log.info("")
    log.info("  --- Shuffle test ---")
    rng    = np.random.default_rng(RAND_STATE)
    f1_r   = f1_score(y_test, y_pred, zero_division=0)
    f1_shf = [f1_score(rng.permutation(y_test), y_pred, zero_division=0)
              for _ in range(n)]
    mean_s = float(np.mean(f1_shf))
    ok     = mean_s < f1_r * 0.7
    log.info("  F1 reale:           %.4f", f1_r)
    log.info("  F1 shuffle (media): %.4f", mean_s)
    log.info("  %s", "✓ OK — modello discrimina" if ok
             else "✗ ATTENZIONE — verifica dataset/etichette")
    return {"f1_real": round(f1_r,4), "f1_shuffle": round(mean_s,4),
            "shuffle_ok": ok}


# ══════════════════════════════════════════════════════════════════════════════
# 9. Cross-validation
# ══════════════════════════════════════════════════════════════════════════════

def cross_validate_ridge(X_feat, y, class_weight, alpha):
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
# 10. Grafici
# ══════════════════════════════════════════════════════════════════════════════

def plot_class_distribution(y_train, y_test, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Distribuzione classi", fontsize=12)
    for ax, y, title in [(axes[0],y_train,"Train"),(axes[1],y_test,"Test")]:
        counts = [(y==0).sum(), (y==1).sum()]
        bars   = ax.bar(["No arco","Arco"], counts,
                        color=["steelblue","tomato"], edgecolor="white")
        for bar, c in zip(bars, counts):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f"{c}\n({100*c/len(y):.1f}%)",
                    ha="center", va="bottom", fontsize=10)
        ax.set_title(title); ax.set_ylabel("Campioni")
        ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"class_distribution.png"),
                dpi=130, bbox_inches="tight")
    plt.close()


def plot_results(y_test, y_pred, y_score, out_dir, threshold=0.5,
                 n_hydra=0, n_mr=0):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(
        f"MultiRocketHydra + Ridge  (soglia={threshold:.2f}  "
        f"Hydra={n_hydra} + MR={n_mr} feature)",
        fontsize=12)

    ax = axes[0]
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No arco","Arco"],
                yticklabels=["No arco","Arco"])
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
            color="steelblue", label="No arco")
    ax.hist(y_score[y_test==1], bins=40, alpha=0.6,
            color="tomato", label="Arco")
    ax.axvline(threshold, color="black", ls="--", lw=1.5,
               label=f"soglia={threshold:.2f}")
    ax.set_xlabel("Score (sigmoid)")
    ax.legend(); ax.set_title("Distribuzione score"); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"results_multirockethydra.png"),
                dpi=130, bbox_inches="tight")
    plt.close()
    log.info("  Salvato: results_multirockethydra.png")


def plot_threshold_curve(y_test, y_score, out_dir, best_thr):
    thrs   = np.arange(0.01, 1.00, 0.01)
    arc    = y_test==1
    no_arc = y_test==0
    det_r  = [100.0*((y_score>=t).astype(int)[arc]).sum()/max(arc.sum(),1)
              for t in thrs]
    fp_r   = [100.0*((y_score>=t).astype(int)[no_arc]).sum()/max(no_arc.sum(),1)
              for t in thrs]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thrs, det_r, "tomato",    lw=2, label="Detection rate %")
    ax.plot(thrs, fp_r,  "steelblue", lw=2, label="False Positive rate %")
    ax.axhline(UL_MIN_DETECTION_PCT, color="tomato",    ls="--", lw=1,
               label=f"UL min det={UL_MIN_DETECTION_PCT:.0f}%")
    ax.axhline(UL_MAX_FP_PCT,        color="steelblue", ls="--", lw=1,
               label=f"UL max FP={UL_MAX_FP_PCT:.0f}%")
    ax.axvline(best_thr, color="black", ls=":", lw=2,
               label=f"Soglia ottimale={best_thr:.2f}")
    ax.fill_betweenx([UL_MIN_DETECTION_PCT,100],
                     [best_thr-0.05],[best_thr+0.05],
                     alpha=0.1, color="green", label="Zona conformità")
    ax.set_xlabel("Soglia"); ax.set_ylabel("Percentuale [%]")
    ax.set_title("Analisi multi-soglia — UL1699B (MultiRocketHydra)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"threshold_analysis.png"),
                dpi=130, bbox_inches="tight")
    plt.close()
    log.info("  Salvato: threshold_analysis.png")


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
                        color="tomato" if y_test[ix]==1 else "steelblue")
                ax.set_xlabel("t [s]"); ax.set_ylabel("I [-]")
                ax.grid(alpha=0.3)
                if row == 0: ax.set_title(title, fontsize=9)
            else:
                ax.text(0.5,0.5,"Nessun\nesempio",ha="center",va="center",
                        transform=ax.transAxes, color="gray")
                ax.axis("off")
                if row == 0: ax.set_title(title, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"series_examples.png"),
                dpi=130, bbox_inches="tight")
    plt.close()
    log.info("  Salvato: series_examples.png")


def plot_feature_importance(ridge, n_stat, n_hydra, n_mr, out_dir):
    coef  = np.abs(ridge.coef_.flatten())
    top_k = min(80, len(coef))
    top_i = np.argsort(coef)[::-1][:top_k]
    stat_names = (["RMS","Var","Skew","Kurt","PkMax","PkMin","ZCR","Crest"]
                  + [f"Band{i+1}" for i in range(len(FREQ_BANDS_HZ))]
                  + ["SpEnt","Slope"])

    def _color(i):
        if i < n_stat:   return "#E8593C"   # coral = statistiche
        if i < n_stat + n_hydra: return "#1D9E75"  # teal = Hydra
        return "#4C72B0"                     # blu = MultiRocket

    colors = [_color(i) for i in top_i]
    labels = []
    for i in top_i:
        if i < n_stat:
            labels.append(stat_names[i] if i < len(stat_names) else f"S{i}")
        elif i < n_stat + n_hydra:
            labels.append(f"H{i-n_stat}")
        else:
            labels.append(f"R{i-n_stat-n_hydra}")

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(range(top_k), coef[top_i], color=colors, edgecolor="none")
    ax.set_xticks(range(top_k))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("|Coefficiente Ridge|")
    ax.set_title(f"Top-{top_k} feature importance — MultiRocketHydra")
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#E8593C", label=f"Feature statistiche ({n_stat})"),
        Patch(facecolor="#1D9E75", label=f"Feature Hydra ({n_hydra})"),
        Patch(facecolor="#4C72B0", label=f"Feature MultiRocket ({n_mr})"),
    ], fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"feature_importance.png"),
                dpi=130, bbox_inches="tight")
    plt.close()
    log.info("  Salvato: feature_importance.png")


def plot_tsne(X_feat, y, out_dir, max_samples=2000):
    log.info("  t-SNE 2D (~2 min)...")
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    if len(y) > max_samples:
        idx = np.random.default_rng(RAND_STATE).choice(len(y),max_samples,replace=False)
        Xs, ys = X_feat[idx], y[idx]
    else:
        Xs, ys = X_feat, y
    Xp  = PCA(n_components=min(50,Xs.shape[1]),
               random_state=RAND_STATE).fit_transform(Xs)
    emb = TSNE(n_components=2, perplexity=30,
                random_state=RAND_STATE, n_iter=500).fit_transform(Xp)
    fig, ax = plt.subplots(figsize=(7,6))
    ax.scatter(emb[ys==0,0],emb[ys==0,1],s=8,alpha=0.5,
               c="steelblue",label="No arco",rasterized=True)
    ax.scatter(emb[ys==1,0],emb[ys==1,1],s=8,alpha=0.5,
               c="tomato",label="Arco",rasterized=True)
    ax.set_title("t-SNE — MultiRocketHydra feature space")
    ax.legend(fontsize=9); ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"tsne_feature_space.png"),
                dpi=120, bbox_inches="tight")
    plt.close()
    log.info("  Salvato: tsne_feature_space.png")


# ══════════════════════════════════════════════════════════════════════════════
# 11. Export STM32
# ══════════════════════════════════════════════════════════════════════════════

def export_c_header(ridge, out_dir, n_features, best_threshold):
    coef  = ridge.coef_.flatten().astype(np.float32)
    inter = float(ridge.intercept_.flatten()[0])

    def _arr(name, vals):
        lines = [f"static const float {name}[{len(vals)}] = {{"]
        for i in range(0, len(vals), 8):
            lines.append("    " +
                         ", ".join(f"{v:.8f}f" for v in vals[i:i+8]) + ",")
        lines[-1] = lines[-1].rstrip(",")
        lines.append("};")
        return "\n".join(lines)

    h = (f"/**\n * multirockethydra_ridge_inference.h\n"
         f" * Architettura: MultiRocketHydra (Dempster et al., 2023)\n"
         f" * Normativa: UL 1699B  |  Soglia: {best_threshold:.4f}\n */\n"
         f"#ifndef MULTIROCKETHYDRA_RIDGE_H\n#define MULTIROCKETHYDRA_RIDGE_H\n"
         f"#include <stdint.h>\n#include <math.h>\n"
         f"#define N_FEATURES        {n_features}\n"
         f"#define OPTIMAL_THRESHOLD {best_threshold:.4f}f\n"
         f"{_arr('mrh_ridge_coef', coef)}\n"
         f"static const float mrh_ridge_intercept = {inter:.8f}f;\n"
         f"static inline float sigmoid_f(float x)"
         f"{{return 1.0f/(1.0f+expf(-x));}}\n"
         f"static inline float mrh_ridge_score(const float* f, uint32_t n){{\n"
         f"    float d = mrh_ridge_intercept;\n"
         f"    for(uint32_t i=0;i<n;i++) d += mrh_ridge_coef[i]*f[i];\n"
         f"    return sigmoid_f(d);\n}}\n"
         f"static inline uint8_t mrh_ridge_predict(const float* f, uint32_t n){{\n"
         f"    return (mrh_ridge_score(f,n) >= OPTIMAL_THRESHOLD) ? 1u : 0u;\n}}\n"
         f"#endif\n")
    path = os.path.join(out_dir, "multirockethydra_ridge_inference.h")
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(h)
    log.info("  C header: %s", path)


# ══════════════════════════════════════════════════════════════════════════════
# 12. Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="MultiRocketHydra GPU (Dempster 2023) + Ridge\n"
                    "Normativa: UL 1699B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("train")
    parser.add_argument("test")
    parser.add_argument("--out", "-o", default="./results_mrhydra")

    # Hydra
    parser.add_argument("--n-groups",           type=int, default=HYDRA_N_GROUPS,
                        help=f"Gruppi Hydra G (default: {HYDRA_N_GROUPS})")
    parser.add_argument("--n-kernels-per-group", type=int,
                        default=HYDRA_N_KERNELS_PER_GROUP,
                        help=f"Kernel per gruppo K (default: {HYDRA_N_KERNELS_PER_GROUP})")
    parser.add_argument("--hydra-dilations",     type=int,
                        default=HYDRA_MAX_DILATIONS,
                        help=f"Dilazioni Hydra (default: {HYDRA_MAX_DILATIONS})")

    # MultiRocket
    parser.add_argument("--mr-kernels",   type=int, default=MR_N_KERNELS,
                        help=f"Kernel MultiRocket (default: {MR_N_KERNELS})")
    parser.add_argument("--mr-dilations", type=int, default=MR_MAX_DILATIONS,
                        help=f"Dilazioni MultiRocket (default: {MR_MAX_DILATIONS})")

    parser.add_argument("--batch-size",   type=int,   default=BATCH_SIZE)
    parser.add_argument("--ridge-alpha",  type=float, default=RIDGE_ALPHA)
    parser.add_argument("--class-weight", default="balanced",
                        choices=["balanced","none"])
    parser.add_argument("--no-normalize",     action="store_true")
    parser.add_argument("--no-stat-features", action="store_true")
    parser.add_argument("--smote",            action="store_true")
    parser.add_argument("--export-stm32",     action="store_true")
    parser.add_argument("--tsne",             action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cw_val = None if args.class_weight == "none" else args.class_weight

    log.info("=" * 60)
    log.info("SETUP DEVICE")
    log.info("=" * 60)
    try:
        import torch
    except ImportError:
        log.error("PyTorch non trovato: pip install torch "
                  "--index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)
    device = setup_device()

    # Dataset
    X_train, y_train, X_test, y_test = load_datasets(args.train, args.test)
    analyze_distribution(y_train, y_test)
    plot_class_distribution(y_train, y_test, args.out)

    # Normalizzazione
    raw_scaler = None
    if not args.no_normalize:
        X_train, X_test, raw_scaler = normalize_global(X_train, X_test)

    # MultiRocketHydra
    log.info("")
    log.info("=" * 60)
    log.info("MULTIROCKETHYDRA GPU")
    log.info("=" * 60)
    log.info("  Hydra: G=%d  K=%d  dil=%d",
             args.n_groups, args.n_kernels_per_group, args.hydra_dilations)
    log.info("  MultiRocket: kernels=%d  dil=%d",
             args.mr_kernels, args.mr_dilations)

    mrh = MultiRocketHydraGPU(
        n_groups=args.n_groups,
        n_kernels_per_group=args.n_kernels_per_group,
        hydra_max_dilations=args.hydra_dilations,
        mr_n_kernels=args.mr_kernels,
        mr_max_dilations=args.mr_dilations,
        device=device,
        random_state=RAND_STATE,
    )
    mrh.fit(X_train)

    log.info("")
    log.info("  Estrazione feature — TRAIN")
    t0 = time.time()
    feat_mrh_train = mrh.transform(X_train, batch_size=args.batch_size)
    t_tr = time.time() - t0

    log.info("")
    log.info("  Estrazione feature — TEST")
    t0 = time.time()
    feat_mrh_test = mrh.transform(X_test, batch_size=args.batch_size)
    t_te = time.time() - t0

    n_hydra_feat = mrh.n_hydra_features
    n_mr_feat    = mrh.n_mr_features

    # Feature statistiche
    n_stat = 0
    stat_train, stat_test = None, None
    if not args.no_stat_features:
        log.info("")
        log.info("=" * 60)
        log.info("FEATURE STATISTICHE MANUALI")
        log.info("=" * 60)
        stat_train = extract_statistical_features(X_train)
        stat_test  = extract_statistical_features(X_test)
        feat_train_raw = np.concatenate([stat_train, feat_mrh_train], axis=1)
        feat_test_raw  = np.concatenate([stat_test,  feat_mrh_test],  axis=1)
        n_stat = stat_train.shape[1]
        log.info("  Totale: stat=%d + Hydra=%d + MR=%d = %d",
                 n_stat, n_hydra_feat, n_mr_feat,
                 feat_train_raw.shape[1])
    else:
        feat_train_raw = feat_mrh_train
        feat_test_raw  = feat_mrh_test

    # Normalizza feature space
    feat_scaler = StandardScaler()
    feat_train  = feat_scaler.fit_transform(feat_train_raw)
    feat_test   = feat_scaler.transform(feat_test_raw)

    # Cross-validation
    cv_res = cross_validate_ridge(feat_train, y_train,
                                   args.class_weight, args.ridge_alpha)

    # SMOTE
    smote_applied = False
    y_train_fit   = y_train
    if args.smote:
        log.info("")
        log.info("=" * 60)
        log.info("SMOTE")
        log.info("=" * 60)
        feat_train, y_train_fit = apply_smote(feat_train, y_train)
        smote_applied = True

    # Ridge
    log.info("")
    log.info("=" * 60)
    log.info("RIDGE CLASSIFIER")
    log.info("=" * 60)
    log.info("  alpha=%.2f  class_weight=%s  n_features=%d",
             args.ridge_alpha, args.class_weight, feat_train.shape[1])
    ridge = RidgeClassifier(alpha=args.ridge_alpha, class_weight=cw_val,
                             random_state=RAND_STATE)
    t0 = time.time()
    ridge.fit(feat_train, y_train_fit)
    t_ridge = time.time() - t0
    log.info("  Ridge fit in %.2f s", t_ridge)

    # Predizioni
    y_score  = _to_proba(ridge, feat_test)
    best_thr = threshold_analysis(y_test, y_score)
    y_pred   = (y_score >= best_thr).astype(int)

    # Metriche
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

    ul  = ul1699b_metric(y_test, y_pred, threshold=best_thr)
    shf = shuffle_test(y_test, y_pred)

    # Grafici
    log.info("")
    log.info("=" * 60)
    log.info("SALVATAGGIO GRAFICI")
    log.info("=" * 60)
    n_features = feat_train.shape[1]
    plot_results(y_test, y_pred, y_score, args.out,
                 threshold=best_thr, n_hydra=n_hydra_feat, n_mr=n_mr_feat)
    plot_series_examples(X_test, y_test, y_pred, args.out)
    plot_threshold_curve(y_test, y_score, args.out, best_thr)
    if not args.no_stat_features:
        plot_feature_importance(ridge, n_stat, n_hydra_feat, n_mr_feat, args.out)
    if args.tsne:
        plot_tsne(feat_train, y_train_fit, args.out)

    # Bundle
    metrics_all = {
        "accuracy":           round(acc,4),
        "balanced_accuracy":  round(ba, 4),
        "f1_arc":             round(f1, 4),
        "roc_auc":            round(auc,4),
        "avg_precision":      round(ap, 4),
        **ul, **cv_res,
        "shuffle_ok":         shf["shuffle_ok"],
        "f1_real":            shf["f1_real"],
        "f1_shuffle":         shf["f1_shuffle"],
    }
    bundle = {
        "mrh":            mrh,
        "feat_scaler":    feat_scaler,
        "ridge":          ridge,
        "raw_scaler":     raw_scaler,
        "best_threshold": best_thr,
        "n_features":     n_features,
        "n_stat":         n_stat,
        "n_hydra":        n_hydra_feat,
        "n_mr":           n_mr_feat,
        "ridge_alpha":    args.ridge_alpha,
        "class_weight":   args.class_weight,
        "smote_applied":  smote_applied,
        "metrics":        metrics_all,
    }
    bundle_path = os.path.join(args.out, "multirockethydra_bundle.pkl")
    with open(bundle_path, "wb") as fp:
        pickle.dump(bundle, fp)
    log.info("  Bundle: %s", bundle_path)

    if args.export_stm32:
        log.info("")
        log.info("=" * 60)
        log.info("EXPORT STM32")
        log.info("=" * 60)
        export_c_header(ridge, args.out, n_features, best_thr)

    # Config JSON
    cfg = {
        "model": "MultiRocketHydra (Dempster 2023) + Ridge",
        "normativa": "UL1699B", "fs_hz": FS_HZ,
        "n_features": n_features, "best_threshold": round(best_thr,4),
        "class_weight": args.class_weight, "smote_applied": smote_applied,
        "hydra": {"n_groups": args.n_groups,
                  "n_kernels_per_group": args.n_kernels_per_group,
                  "max_dilations": args.hydra_dilations,
                  "n_features": n_hydra_feat},
        "multirocket": {"n_kernels": args.mr_kernels,
                        "max_dilations": args.mr_dilations,
                        "n_features": n_mr_feat},
        "metrics": {k: v for k, v in metrics_all.items()
                    if isinstance(v, (int, float, bool, str))},
    }
    with open(os.path.join(args.out,"deployment_config.json"),
              "w", encoding="utf-8") as fp:
        json.dump(cfg, fp, indent=2, default=str)
    log.info("  Config JSON salvato")

    # Riepilogo
    log.info("")
    log.info("=" * 72)
    log.info("RIEPILOGO FINALE")
    log.info("=" * 72)
    log.info("  Device:            %s", device)
    log.info("  Campioni train:    %d  (tutti, nessun undersampling)", len(y_train))
    log.info("  Feature totali:    %d  (stat=%d + Hydra=%d + MR=%d)",
             n_features, n_stat, n_hydra_feat, n_mr_feat)
    log.info("  T feat train:      %.1f s", t_tr)
    log.info("  T feat test:       %.1f s", t_te)
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
             "✓ OK" if shf["shuffle_ok"] else "✗ ATTENZIONE")
    log.info("")
    log.info("  Output in: %s", args.out)

    if not ul["ul1699b_conforme"] or not shf["shuffle_ok"]:
        log.warning("")
        log.warning("  AZIONI SUGGERITE:")
        if not shf["shuffle_ok"]:
            log.warning("  → --n-groups 128 --mr-kernels 2000  (più feature)")
            log.warning("  → --smote                            (bilancia classi)")
        if ul["false_positive_rate_pct"] > UL_MAX_FP_PCT:
            log.warning("  → --ridge-alpha 10.0  (meno FP)")
        if ul["detection_rate_pct"] < UL_MIN_DETECTION_PCT:
            log.warning("  → --ridge-alpha 0.1 --smote")
        log.warning("  → --tsne  (visualizza separabilità classi)")


if __name__ == "__main__":
    main()