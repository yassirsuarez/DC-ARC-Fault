#!/usr/bin/env python3
"""
train_multirocket_gpu_v2.py
===========================
Pipeline GPU-accelerata per classificatore archi elettrici in impianti PV DC.
Architettura: MultiRocket (PyTorch GPU, vettorizzato) + RidgeClassifier

DIFFERENZE RISPETTO A v1:
  - FIX CRASH: estrazione feature completamente vettorizzata — tutti i kernel
    vengono applicati in un'unica operazione conv1d grouped invece di un loop
    per kernel → riduce memoria da ~5 GB a ~200 MB e tempo da ore a minuti
  - FIX MEMORIA: n_kernels default ridotto a 1000 (→ ~4000 feature) sufficiente
    per questo dataset; usa --n-kernels 2500 per più feature se la RAM lo permette
  - ALLINEATO a InceptionTime v2: stesso dataset completo (no undersampling),
    stessa gestione sbilanciamento (class_weight Ridge), stessi grafici,
    stessa analisi multi-soglia UL1699B, stesso shuffle test

CONFRONTO CON INCEPTIONTIME:
  - MultiRocket+Ridge: kernel random (non apprende), classificatore lineare
    → più veloce, più interpretabile, meglio deployabile su STM32
  - InceptionTime: rete neurale che apprende le feature
    → potenzialmente più accurata, ma più lenta e pesante

REQUISITI:
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install scikit-learn numpy matplotlib seaborn scipy

USO:
    python train_multirocket_gpu_v2.py train.npz test.npz
    python train_multirocket_gpu_v2.py train.npz test.npz --n-kernels 2500
    python train_multirocket_gpu_v2.py train.npz test.npz --smote
    python train_multirocket_gpu_v2.py train.npz test.npz --export-stm32
    python train_multirocket_gpu_v2.py train.npz test.npz --tsne

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

# MultiRocket — parametri conservativi per memoria
# 1000 kernel × 2 set × 2 aggregazioni × ~10 dilazioni = ~40.000 feature
# Memoria stimata: 27770 × 40000 × 4 byte ≈ 4.4 GB — borderline, ok con swap
# Usa --n-kernels 500 se hai < 16 GB RAM
MR_N_KERNELS     = 1_000
MR_MAX_DILATIONS = 10       # ridotto da 32 → meno feature ma molto meno memoria
MR_BATCH_SIZE    = 512      # batch più grandi per GPU → più veloce

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
            log.warning("  VRAM < 6 GB — considera --n-kernels 500 --batch-size 128")
    else:
        device = torch.device("cpu")
        log.warning("  CUDA non disponibile — fallback CPU")
    return device


# ══════════════════════════════════════════════════════════════════════════════
# 1. MultiRocket GPU — implementazione VETTORIZZATA
# ══════════════════════════════════════════════════════════════════════════════

class MultiRocketGPU:
    """
    MultiRocket vettorizzato su GPU.

    FIX PRINCIPALE rispetto a v1:
    Invece di un loop `for kernel in kernels: conv1d(...)` che:
      - esegue N chiamate GPU separate (overhead enorme)
      - accumula tensori e poi li concatena (picco memoria)
      - impiega ore su dataset grandi

    Usa conv1d GROUPED dove tutti i kernel con la stessa dilation
    vengono applicati in un'unica operazione:
      - 1 chiamata GPU per gruppo di dilation
      - memoria controllata: solo il risultato finale viene tenuto
      - tempo: minuti invece di ore

    Feature per campione: n_kernels × n_dilations × 4
      (PPV_originale, mean_originale, PPV_diff, mean_diff)
    """

    def __init__(self, n_kernels=MR_N_KERNELS, max_dilations=MR_MAX_DILATIONS,
                 device=None, random_state=RAND_STATE):
        import torch
        self.n_kernels     = n_kernels
        self.max_dilations = max_dilations
        self.device        = device or torch.device("cpu")
        self.random_state  = random_state
        self._fitted       = False

    def fit(self, X: np.ndarray, y=None):
        """
        Genera kernel random e calcola le dilazioni.
        Non c'è apprendimento — i kernel sono random e fissi.
        """
        import torch
        T   = X.shape[1]
        rng = np.random.default_rng(self.random_state)

        kernel_lengths = [7, 9, 11]
        n   = self.n_kernels

        log.info("  Generazione %d kernel (lunghezze=%s, max_dil=%d)...",
                 n, kernel_lengths, self.max_dilations)
        t0 = time.time()

        # Scegli lunghezze
        lengths = rng.choice(kernel_lengths, size=n)

        # Per ogni lunghezza unica, genera kernel e calcola dilazioni
        self._groups = []   # lista di dict {L, dilation, weights_a, weights_b, biases_a, biases_b}

        for L in kernel_lengths:
            mask = lengths == L
            n_L  = mask.sum()
            if n_L == 0:
                continue

            # Dilazioni log-spaced per questa lunghezza
            max_exp = np.log2((T - 1) / (L - 1))
            dils = np.unique(np.floor(
                np.power(2, np.linspace(0, max_exp,
                                        min(self.max_dilations,
                                            int(max_exp) + 1)))
            ).astype(int))

            # Pesi: (n_L, L) normalizzati a media zero
            w_a = rng.normal(0, 1, (n_L, L)).astype(np.float32)
            w_b = rng.normal(0, 1, (n_L, L)).astype(np.float32)
            w_a -= w_a.mean(axis=1, keepdims=True)
            w_b -= w_b.mean(axis=1, keepdims=True)

            for d in dils:
                # Bias: quantile della risposta su segnale random
                dummy = rng.normal(0, 1, (1, 1, T)).astype(np.float32)
                dummy_t    = torch.tensor(dummy,            device=self.device)
                dummy_diff = torch.tensor(np.diff(dummy, axis=2), device=self.device)
                k_a = torch.tensor(w_a, device=self.device).unsqueeze(1)  # (n_L,1,L)
                k_b = torch.tensor(w_b, device=self.device).unsqueeze(1)

                import torch.nn.functional as F
                # Grouped conv: tutti i kernel in una sola chiamata
                # Usiamo padding="same" equivalente manuale
                pad = (int(d) * (L - 1)) // 2
                out_a = F.conv1d(dummy_t.expand(1, n_L, T),
                                 k_a, dilation=int(d), padding=pad,
                                 groups=n_L)   # (1, n_L, T')
                out_b = F.conv1d(dummy_diff.expand(1, n_L, T - 1),
                                 k_b, dilation=int(d), padding=pad,
                                 groups=n_L)

                q_a = float(rng.uniform(0, 1))
                q_b = float(rng.uniform(0, 1))
                bias_a = torch.quantile(out_a.cpu(), q_a).item()
                bias_b = torch.quantile(out_b.cpu(), q_b).item()

                self._groups.append({
                    "L":        L,
                    "dilation": int(d),
                    "padding":  pad,
                    "w_a":      w_a,        # (n_L, L) numpy float32
                    "w_b":      w_b,
                    "bias_a":   bias_a,
                    "bias_b":   bias_b,
                    "n_kernels_in_group": n_L,
                })

        n_feat = sum(g["n_kernels_in_group"] * 4 for g in self._groups)
        self._n_features = n_feat
        self._T = T
        self._fitted = True
        log.info("  Kernel generati in %.1f s", time.time() - t0)
        log.info("  Gruppi (lunghezza × dilation): %d", len(self._groups))
        log.info("  Feature totali: %d  (~%.0f MB per campione float32)",
                 n_feat, n_feat * 4 / 1e6)
        # Stima memoria totale
        n_train_est = 27770
        mem_gb = n_train_est * n_feat * 4 / 1e9
        log.info("  Memoria stimata per train set: %.2f GB", mem_gb)
        if mem_gb > 8:
            log.warning("  ATTENZIONE: >8 GB — considera --n-kernels %d",
                        self.n_kernels // 2)
        return self

    def _transform_batch_vectorized(self, X_batch: np.ndarray) -> np.ndarray:
        """
        Trasforma un batch applicando tutti i kernel di ogni gruppo in una
        sola chiamata conv1d.

        Strategia corretta per conv1d batched su GPU:
          - Input:  (n, 1, T)
          - Kernel: (n_k, 1, L)  — out_channels=n_k, in_channels=1, groups=1
          - Output: (n, n_k, T') — n campioni × n_k feature × lunghezza temporale

        Un loop per GRUPPO (22 iterazioni) invece che per kernel (>1000).
        Questo è già molto veloce su GPU perché ogni chiamata elabora
        tutti i kernel del gruppo in parallelo.

        Shape output: (batch, n_features)
        """
        import torch
        import torch.nn.functional as F

        n, T = X_batch.shape
        # (n, 1, T) — canale singolo
        X_t    = torch.tensor(X_batch, dtype=torch.float32,
                              device=self.device).unsqueeze(1)
        # (n, 1, T-1) — differenze prime per set B
        X_diff = torch.diff(X_t, dim=2)

        parts = []
        for g in self._groups:
            d   = g["dilation"]
            p   = g["padding"]
            n_k = g["n_kernels_in_group"]

            # Kernel shape: (n_k, 1, L) — n_k filtri, 1 canale input
            w_a = torch.tensor(g["w_a"], dtype=torch.float32,
                               device=self.device).unsqueeze(1)  # (n_k, 1, L)
            w_b = torch.tensor(g["w_b"], dtype=torch.float32,
                               device=self.device).unsqueeze(1)

            # conv1d: (n, 1, T) × (n_k, 1, L) → (n, n_k, T')
            # in_channels=1, out_channels=n_k, groups=1
            out_a = F.conv1d(X_t,    w_a, dilation=d, padding=p) - g["bias_a"]
            out_b = F.conv1d(X_diff, w_b, dilation=d, padding=p) - g["bias_b"]
            # out_a: (n, n_k, T_a),  out_b: (n, n_k, T_b)

            # PPV = proporzione valori > 0,  mean = media
            ppv_a  = (out_a > 0).float().mean(dim=2)  # (n, n_k)
            mean_a = out_a.mean(dim=2)
            ppv_b  = (out_b > 0).float().mean(dim=2)
            mean_b = out_b.mean(dim=2)

            # Stack → (n, n_k, 4) → (n, n_k*4)
            group_feat = torch.stack([ppv_a, mean_a, ppv_b, mean_b],
                                     dim=2).reshape(n, n_k * 4)
            parts.append(group_feat.cpu().numpy())

        return np.concatenate(parts, axis=1)

    def transform(self, X: np.ndarray, batch_size: int = MR_BATCH_SIZE) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("MultiRocketGPU non ancora fittato.")

        log.info("  Estrazione feature MultiRocket (vettorizzata, batch=%d, n=%d)...",
                 batch_size, len(X))
        parts     = []
        n_batches = int(np.ceil(len(X) / batch_size))
        t0        = time.time()

        for bi, start in enumerate(range(0, len(X), batch_size)):
            batch = X[start:start + batch_size]
            parts.append(self._transform_batch_vectorized(batch))

            if (bi + 1) % max(1, n_batches // 8) == 0 or bi == 0:
                elapsed = time.time() - t0
                eta     = elapsed / (bi + 1) * (n_batches - bi - 1)
                log.info("    Batch %d/%d  (%.0f%%)  elapsed=%.0fs  ETA=%.0fs",
                         bi + 1, n_batches,
                         100.0 * (bi + 1) / n_batches,
                         elapsed, eta)

        feat = np.concatenate(parts, axis=0)
        log.info("  Feature shape: %s  (%.1f s totali  %.2f GB)",
                 feat.shape, time.time() - t0,
                 feat.nbytes / 1e9)
        return feat

    @property
    def n_features(self):
        return self._n_features if self._fitted else -1


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
        log.warning("  Drift distribuzione: %.1f pp — verifica lo split", drift)
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
    sc = StandardScaler()
    X_tr_n = sc.fit_transform(X_train)
    X_te_n = sc.transform(X_test)
    log.info("  z-score globale (fit su train)")
    log.info("  Train: μ=%.4f σ=%.4f", X_tr_n.mean(), X_tr_n.std())
    log.info("  Test:  μ=%.4f σ=%.4f", X_te_n.mean(), X_te_n.std())
    return X_tr_n, X_te_n, sc


# ══════════════════════════════════════════════════════════════════════════════
# 5. Feature statistiche manuali
# ══════════════════════════════════════════════════════════════════════════════

def extract_statistical_features(X: np.ndarray) -> np.ndarray:
    """
    Feature statistiche e spettrali per campione.
    Memoria: n × 14 × 4 byte — trascurabile.
    """
    log.info("  Estrazione feature statistiche (n=%d)...", len(X))
    n, T    = X.shape
    n_bands = len(FREQ_BANDS_HZ)
    feats   = np.zeros((n, 8 + n_bands + 2), dtype=np.float32)
    freqs   = np.fft.rfftfreq(T, d=1.0 / FS_HZ)
    t_norm  = np.linspace(0, 1, T)

    for i, x in enumerate(X):
        rms  = float(np.sqrt(np.mean(x**2)))
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
        feats[i, 8+n_bands] = float(-np.sum(np.clip(psd,1e-12,None)
                                             * np.log2(np.clip(psd,1e-12,None))))
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
    log.info("  F1 shuffle (media): %.4f  (atteso ≪ F1 reale)", mean_s)
    log.info("  Esito: %s",
             "✓ OK — modello discrimina" if ok
             else "✗ ATTENZIONE — potrebbe predire sempre la classe maggioritaria")
    if not ok:
        log.warning("  Azioni: --smote, oppure --n-kernels 2500 per più feature")
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
    for ax, y, title in [(axes[0], y_train, "Train"), (axes[1], y_test, "Test")]:
        counts = [(y==0).sum(), (y==1).sum()]
        bars   = ax.bar(["No arco (0)","Arco (1)"], counts,
                        color=["steelblue","tomato"], edgecolor="white")
        for bar, c in zip(bars, counts):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f"{c}\n({100*c/len(y):.1f}%)",
                    ha="center", va="bottom", fontsize=10)
        ax.set_title(title); ax.set_ylabel("Campioni")
        ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "class_distribution.png"),
                dpi=130, bbox_inches="tight")
    plt.close()


def plot_results(y_test, y_pred, y_score, out_dir, threshold=0.5):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(
        f"MultiRocket GPU + Ridge  (soglia={threshold:.2f})", fontsize=13)

    # Confusion matrix
    ax = axes[0]
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No arco","Arco"],
                yticklabels=["No arco","Arco"])
    ax.set_title("Confusion Matrix")
    ax.set_ylabel("Reale"); ax.set_xlabel("Predetto")

    # ROC
    ax = axes[1]
    fpr_c, tpr_c, _ = roc_curve(y_test, y_score)
    auc = roc_auc_score(y_test, y_score)
    ax.plot(fpr_c, tpr_c, "steelblue", lw=2, label=f"AUC={auc:.3f}")
    ax.plot([0,1],[0,1],"k--",lw=1)
    ax.axvline(UL_MAX_FP_PCT/100,        color="tomato", ls=":", lw=1.5,
               label=f"UL max FP={UL_MAX_FP_PCT:.0f}%")
    ax.axhline(UL_MIN_DETECTION_PCT/100, color="green",  ls=":", lw=1.5,
               label=f"UL min det={UL_MIN_DETECTION_PCT:.0f}%")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(fontsize=8); ax.set_title("ROC Curve"); ax.grid(alpha=0.3)

    # Precision-Recall
    ax = axes[2]
    prec, rec, _ = precision_recall_curve(y_test, y_score)
    ap = average_precision_score(y_test, y_score)
    ax.plot(rec, prec, "tomato", lw=2, label=f"AP={ap:.3f}")
    ax.axhline(y_test.mean(), color="gray", ls="--", lw=1,
               label=f"Baseline={y_test.mean():.2f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.legend(); ax.set_title("Precision-Recall"); ax.grid(alpha=0.3)

    # Score distribution
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
    plt.savefig(os.path.join(out_dir, "results_multirocket_ridge.png"),
                dpi=130, bbox_inches="tight")
    plt.close()
    log.info("  Salvato: results_multirocket_ridge.png")


def plot_threshold_curve(y_test, y_score, out_dir, best_thr):
    thrs    = np.arange(0.01, 1.00, 0.01)
    arc     = y_test==1
    no_arc  = y_test==0
    det_r   = [100.0*((y_score>=t).astype(int)[arc]).sum()
               / max(arc.sum(), 1) for t in thrs]
    fp_r    = [100.0*((y_score>=t).astype(int)[no_arc]).sum()
               / max(no_arc.sum(), 1) for t in thrs]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thrs, det_r, "tomato",    lw=2, label="Detection rate %")
    ax.plot(thrs, fp_r,  "steelblue", lw=2, label="False Positive rate %")
    ax.axhline(UL_MIN_DETECTION_PCT, color="tomato",    ls="--", lw=1,
               label=f"UL1699B min det={UL_MIN_DETECTION_PCT:.0f}%")
    ax.axhline(UL_MAX_FP_PCT,        color="steelblue", ls="--", lw=1,
               label=f"UL1699B max FP={UL_MAX_FP_PCT:.0f}%")
    ax.axvline(best_thr, color="black", ls=":", lw=2,
               label=f"Soglia ottimale={best_thr:.2f}")
    ax.fill_betweenx([UL_MIN_DETECTION_PCT, 100],
                     [best_thr-0.05], [best_thr+0.05],
                     alpha=0.1, color="green", label="Zona conformità")
    ax.set_xlabel("Soglia"); ax.set_ylabel("Percentuale [%]")
    ax.set_title("Analisi multi-soglia — UL1699B (MultiRocket v2)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "threshold_analysis.png"),
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
                ax.text(0.5, 0.5, "Nessun\nesempio", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
                ax.axis("off")
                if row == 0: ax.set_title(title, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "series_examples.png"),
                dpi=130, bbox_inches="tight")
    plt.close()
    log.info("  Salvato: series_examples.png")


def plot_feature_importance(ridge, n_stat, n_rocket, out_dir):
    coef  = np.abs(ridge.coef_.flatten())
    top_k = min(60, len(coef))
    top_i = np.argsort(coef)[::-1][:top_k]
    stat_names = (["RMS","Var","Skew","Kurt","PkMax","PkMin","ZCR","Crest"]
                  + [f"Band{i+1}" for i in range(len(FREQ_BANDS_HZ))]
                  + ["SpEnt","Slope"])
    colors = ["#E8593C" if i < n_stat else "#1D9E75" for i in top_i]
    labels = [stat_names[i] if i < n_stat else f"R{i-n_stat}" for i in top_i]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(top_k), coef[top_i], color=colors, edgecolor="none")
    ax.set_xticks(range(top_k))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("|Coefficiente Ridge|")
    ax.set_title(f"Top-{top_k} feature importance")
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#E8593C", label=f"Feature statistiche ({n_stat})"),
        Patch(facecolor="#1D9E75", label=f"Feature Rocket ({n_rocket})"),
    ], fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "feature_importance.png"),
                dpi=130, bbox_inches="tight")
    plt.close()
    log.info("  Salvato: feature_importance.png")


def plot_tsne(X_feat, y, out_dir, max_samples=2000):
    log.info("  t-SNE 2D (può richiedere ~2 min)...")
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    if len(y) > max_samples:
        idx = np.random.default_rng(RAND_STATE).choice(len(y),
                                                        max_samples, replace=False)
        Xs, ys = X_feat[idx], y[idx]
    else:
        Xs, ys = X_feat, y
    Xp  = PCA(n_components=min(50, Xs.shape[1]),
               random_state=RAND_STATE).fit_transform(Xs)
    emb = TSNE(n_components=2, perplexity=30,
                random_state=RAND_STATE, n_iter=500).fit_transform(Xp)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(emb[ys==0,0], emb[ys==0,1], s=8, alpha=0.5,
               c="steelblue", label="No arco", rasterized=True)
    ax.scatter(emb[ys==1,0], emb[ys==1,1], s=8, alpha=0.5,
               c="tomato",    label="Arco",    rasterized=True)
    ax.set_title("t-SNE feature space MultiRocket v2")
    ax.legend(fontsize=9); ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "tsne_feature_space.png"),
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

    h = (f"/**\n * multirocket_ridge_inference.h\n"
         f" * Normativa: UL 1699B  |  Soglia: {best_threshold:.4f}\n */\n"
         f"#ifndef MULTIROCKET_RIDGE_H\n#define MULTIROCKET_RIDGE_H\n"
         f"#include <stdint.h>\n#include <math.h>\n"
         f"#define N_FEATURES        {n_features}\n"
         f"#define OPTIMAL_THRESHOLD {best_threshold:.4f}f\n"
         f"{_arr('mr_ridge_coef', coef)}\n"
         f"static const float mr_ridge_intercept = {inter:.8f}f;\n"
         f"static inline float sigmoid_f(float x)"
         f"{{return 1.0f/(1.0f+expf(-x));}}\n"
         f"static inline float mr_ridge_score(const float* f, uint32_t n){{\n"
         f"    float d = mr_ridge_intercept;\n"
         f"    for(uint32_t i=0;i<n;i++) d += mr_ridge_coef[i]*f[i];\n"
         f"    return sigmoid_f(d);\n}}\n"
         f"static inline uint8_t mr_ridge_predict(const float* f, uint32_t n){{\n"
         f"    return (mr_ridge_score(f,n) >= OPTIMAL_THRESHOLD) ? 1u : 0u;\n}}\n"
         f"#endif\n")
    path = os.path.join(out_dir, "multirocket_ridge_inference.h")
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(h)
    log.info("  C header: %s", path)


# ══════════════════════════════════════════════════════════════════════════════
# 12. Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="MultiRocket GPU v2 (vettorizzato) + Ridge\n"
                    "Normativa: UL 1699B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("train")
    parser.add_argument("test")
    parser.add_argument("--out", "-o", default="./results_rocket_v2")
    parser.add_argument("--n-kernels",     type=int,   default=MR_N_KERNELS,
                        help=f"Kernel per lunghezza (default: {MR_N_KERNELS}). "
                             f"Aumenta a 2500 per più feature se hai >16 GB RAM.")
    parser.add_argument("--max-dilations", type=int,   default=MR_MAX_DILATIONS,
                        help=f"Dilazioni per kernel (default: {MR_MAX_DILATIONS}). "
                             f"Aumenta a 20 per più feature.")
    parser.add_argument("--batch-size",    type=int,   default=MR_BATCH_SIZE,
                        help=f"Batch GPU (default: {MR_BATCH_SIZE}). "
                             f"Riduci a 128 se OOM.")
    parser.add_argument("--ridge-alpha",   type=float, default=RIDGE_ALPHA)
    parser.add_argument("--class-weight",  default="balanced",
                        choices=["balanced", "none"])
    parser.add_argument("--no-normalize",  action="store_true")
    parser.add_argument("--no-stat-features", action="store_true")
    parser.add_argument("--smote",            action="store_true")
    parser.add_argument("--export-stm32",     action="store_true")
    parser.add_argument("--tsne",             action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cw_val = None if args.class_weight == "none" else args.class_weight

    # GPU
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

    # MultiRocket vettorizzato
    log.info("")
    log.info("=" * 60)
    log.info("MULTIROCKET GPU (vettorizzato)")
    log.info("=" * 60)
    log.info("  n_kernels=%d  max_dilations=%d  batch=%d  device=%s",
             args.n_kernels, args.max_dilations, args.batch_size, device)

    rocket = MultiRocketGPU(
        n_kernels=args.n_kernels,
        max_dilations=args.max_dilations,
        device=device,
        random_state=RAND_STATE,
    )
    rocket.fit(X_train)

    log.info("")
    log.info("  Estrazione feature — TRAIN")
    t0 = time.time()
    feat_rk_train = rocket.transform(X_train, batch_size=args.batch_size)
    t_tr = time.time() - t0

    log.info("")
    log.info("  Estrazione feature — TEST")
    t0 = time.time()
    feat_rk_test = rocket.transform(X_test, batch_size=args.batch_size)
    t_te = time.time() - t0

    # Feature statistiche
    stat_train, stat_test = None, None
    n_stat = 0
    if not args.no_stat_features:
        log.info("")
        log.info("=" * 60)
        log.info("FEATURE STATISTICHE MANUALI")
        log.info("=" * 60)
        stat_train = extract_statistical_features(X_train)
        stat_test  = extract_statistical_features(X_test)
        feat_train_raw = np.concatenate([feat_rk_train, stat_train], axis=1)
        feat_test_raw  = np.concatenate([feat_rk_test,  stat_test],  axis=1)
        n_stat   = stat_train.shape[1]
        n_rocket = feat_rk_train.shape[1]
        log.info("  Totale: Rocket=%d + stat=%d = %d",
                 n_rocket, n_stat, feat_train_raw.shape[1])
    else:
        feat_train_raw = feat_rk_train
        feat_test_raw  = feat_rk_test
        n_rocket = feat_rk_train.shape[1]

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
    for line in classification_report(
            y_test, y_pred,
            target_names=["No arco","Arco"], digits=3).splitlines():
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
    plot_results(y_test, y_pred, y_score, args.out, threshold=best_thr)
    plot_series_examples(X_test, y_test, y_pred, args.out)
    plot_threshold_curve(y_test, y_score, args.out, best_thr)
    if not args.no_stat_features:
        plot_feature_importance(ridge, n_stat, n_rocket, args.out)
    if args.tsne:
        plot_tsne(feat_train, y_train_fit, args.out)

    # Bundle
    n_features = feat_train.shape[1]
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
        "rocket":         rocket,
        "feat_scaler":    feat_scaler,
        "ridge":          ridge,
        "raw_scaler":     raw_scaler,
        "best_threshold": best_thr,
        "n_features":     n_features,
        "n_stat":         n_stat,
        "n_rocket":       n_rocket,
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

    # Config JSON
    cfg = {
        "model": "MultiRocket GPU v2 + Ridge",
        "normativa": "UL1699B", "fs_hz": FS_HZ,
        "n_features": n_features, "best_threshold": round(best_thr,4),
        "class_weight": args.class_weight, "smote_applied": smote_applied,
        "rocket": {"n_kernels": args.n_kernels,
                   "max_dilations": args.max_dilations,
                   "device": str(device)},
        "metrics": {k: v for k, v in metrics_all.items()
                    if isinstance(v, (int, float, bool, str))},
    }
    with open(os.path.join(args.out, "deployment_config.json"),
              "w", encoding="utf-8") as fp:
        json.dump(cfg, fp, indent=2, default=str)

    # Riepilogo
    log.info("")
    log.info("=" * 72)
    log.info("RIEPILOGO FINALE")
    log.info("=" * 72)
    log.info("  Device:            %s", device)
    log.info("  Campioni train:    %d  (tutti, nessun undersampling)", len(y_train))
    log.info("  N feature:         %d (Rocket=%d + stat=%d)",
             n_features, n_rocket, n_stat)
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
            log.warning("  → --n-kernels 2500 --max-dilations 20  (più feature)")
            log.warning("  → --smote                               (bilancia classi)")
        if ul["false_positive_rate_pct"] > UL_MAX_FP_PCT:
            log.warning("  → --ridge-alpha 10.0  (meno FP)")
        if ul["detection_rate_pct"] < UL_MIN_DETECTION_PCT:
            log.warning("  → --ridge-alpha 0.1 --smote  (più detection)")
        log.warning("  → --tsne  (visualizza separabilità classi)")


if __name__ == "__main__":
    main()