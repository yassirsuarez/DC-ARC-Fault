#!/usr/bin/env python3
"""
Confronto_modelli.py  (MS-RCFE)  -  v2, 2026-10-07
==================================================
Confronta il modello MS-RCFE in FP32 (ONNX originale) con la versione INT8
quantizzata da ST Edge AI, sul test set (arc_dataset_test.npz), con
onnxruntime su CPU.

Entrambe le versioni estraggono le 576 feature dalla finestra; la
classificazione e' affidata allo STESSO Ridge, quindi le differenze dipendono
solo dalla quantizzazione dell'estrattore. Il Ridge e' ricostruito da
ridge_coef.npy / ridge_intercept.npy: score = feature @ coef + intercept,
classe 1 (arco) se score > 0 (come RidgeClassifier.predict). Non serve torch.
(In alternativa --bundle legge il Ridge dal .pkl, che richiede torch.)

Output: accuratezza, Detection Rate (DR), False Positive Rate (FPR), FN, FP,
per FP32 e INT8, e fedelta' (percentuale di campioni su cui i due modelli
danno la stessa predizione).

I valori sono ottenuti su CPU host e NON sostituiscono la validazione sul
target (STM32H7S78), dove l'aritmetica INT8 puo' differire da onnxruntime.

Percorsi di default: relativi alla radice del repository
(questo file sta in script/modelli_quantizzati/msrcfe/).

USO:
    python Confronto_modelli.py                 # tutto il test set
    python Confronto_modelli.py --n 1000        # sottoinsieme casuale (--seed)
    python Confronto_modelli.py --original percorso/msrcfe.onnx \\
                                --quantized percorso/msrcfe_int8.onnx
Requisiti: numpy, onnxruntime (tqdm facoltativo).
"""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort

try:
    from tqdm import tqdm
except ImportError:                      # tqdm e' facoltativo
    def tqdm(it, **kw):
        return it

# --- PERCORSI DI DEFAULT (relativi al repository) ---
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # script/modelli_quantizzati/msrcfe -> radice del repo

PATH_ORIGINALE = ROOT / "script" / "training" / "msrcfe" / "export_msrcfe" / "msrcfe.onnx"
PATH_QUANTIZZATO = HERE / "msrcfe_PerChannel_quant_calibration_msrcfe_npz_1.onnx"  # da ST Edge AI
PATH_DATASET = ROOT / "dataset" / "dataset_new" / "arc_dataset_test.npz"
PATH_COEF = ROOT / "script" / "training" / "msrcfe" / "results" / "ridge_coef.npy"
PATH_INTERCEPT = ROOT / "script" / "training" / "msrcfe" / "results" / "ridge_intercept.npy"


def carica_ridge(args):
    if args.bundle is not None:                     # richiede torch
        import pickle
        with open(args.bundle, "rb") as f:
            ridge = pickle.load(f)["ridge"]
        return (np.asarray(ridge.coef_, dtype=np.float64).reshape(-1),
                float(np.asarray(ridge.intercept_).reshape(-1)[0]))
    coef = np.load(args.coef).astype(np.float64).reshape(-1)
    intercetta = float(np.load(args.intercept).reshape(-1)[0])
    return coef, intercetta


def predici(sess, input_name, coef, intercetta, sample_1d):
    """sample_1d: array (1000,) - singolo segnale normalizzato. Restituisce 0 o 1."""
    x = sample_1d.astype(np.float32).reshape(1, 1, -1)           # (1, 1, 1000)
    features = sess.run(None, {input_name: x})[0].reshape(-1)    # 576 feature
    score = float(features.astype(np.float64) @ coef + intercetta)
    return int(score > 0)


def metriche(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    return dict(acc=(tp + tn) / len(y), dr=tp / max(tp + fn, 1), fpr=fp / max(tn + fp, 1), fn=fn, fp=fp)


def validazione_msrcfe(args):
    data = np.load(args.dataset)
    X_test, y_test = data["X"], data["y"].astype(int)

    total = len(X_test)
    if args.n is None or args.n >= total:
        indices = np.arange(total)
    else:
        indices = np.sort(np.random.default_rng(args.seed).choice(total, args.n, replace=False))
    n_campioni = len(indices)

    sess_orig = ort.InferenceSession(str(args.original), providers=["CPUExecutionProvider"])
    sess_quant = ort.InferenceSession(str(args.quantized), providers=["CPUExecutionProvider"])
    name_orig = sess_orig.get_inputs()[0].name
    name_quant = sess_quant.get_inputs()[0].name
    coef, intercetta = carica_ridge(args)

    print(f"Analisi su {n_campioni} campioni...")
    pred_orig = np.empty(n_campioni, dtype=int)
    pred_quant = np.empty(n_campioni, dtype=int)
    for k, idx in enumerate(tqdm(indices)):
        pred_orig[k] = predici(sess_orig, name_orig, coef, intercetta, X_test[idx])
        pred_quant[k] = predici(sess_quant, name_quant, coef, intercetta, X_test[idx])
    y = y_test[indices]

    mo, mq = metriche(y, pred_orig), metriche(y, pred_quant)
    fidelity = (pred_orig == pred_quant).mean() * 100

    print("\n" + "=" * 52)
    print("      VALIDAZIONE MS-RCFE FP32 vs INT8 (CPU host)")
    print("=" * 52)
    print(f"  Campioni analizzati:  {n_campioni}")
    print(f"  {'':<10}{'Acc.':>9}{'DR':>9}{'FPR':>9}{'FN':>6}{'FP':>6}")
    for nome, m in (("FP32", mo), ("INT8", mq)):
        print(f"  {nome:<10}{100*m['acc']:>8.2f}%{100*m['dr']:>8.2f}%{100*m['fpr']:>8.2f}%{m['fn']:>6}{m['fp']:>6}")
    print(f"  Delta accuratezza (FP32 - INT8): {100*(mo['acc']-mq['acc']):+.2f} punti")
    print("-" * 52)
    print(f"  FEDELTA' (INT8 vs FP32): {fidelity:.2f}%")
    print("=" * 52)
    print("  Soglia: score del Ridge > 0. Valori da onnxruntime su CPU, non dal target.")


def main():
    p = argparse.ArgumentParser(description="Confronto FP32 vs INT8 per MS-RCFE + Ridge.")
    p.add_argument("--original", type=Path, default=PATH_ORIGINALE, help="msrcfe.onnx (FP32)")
    p.add_argument("--quantized", type=Path, default=PATH_QUANTIZZATO, help="modello INT8 da ST Edge AI")
    p.add_argument("--dataset", type=Path, default=PATH_DATASET, help="arc_dataset_test.npz")
    p.add_argument("--coef", type=Path, default=PATH_COEF, help="ridge_coef.npy")
    p.add_argument("--intercept", type=Path, default=PATH_INTERCEPT, help="ridge_intercept.npy")
    p.add_argument("--bundle", type=Path, default=None,
                   help="msrcfe_bundle.pkl (alternativa ai .npy; richiede torch)")
    p.add_argument("--n", type=int, default=None, help="numero di campioni (default: tutto il test set)")
    p.add_argument("--seed", type=int, default=42, help="seed per il sottoinsieme casuale")
    validazione_msrcfe(p.parse_args())


if __name__ == "__main__":
    main()
