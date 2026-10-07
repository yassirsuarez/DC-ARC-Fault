#!/usr/bin/env python3
"""
Confronto_modelli.py  (InceptionTime)  -  v2, 2026-10-07
========================================================
Confronta InceptionTime in FP32 (ONNX originale) con la versione INT8
quantizzata da ST Edge AI, sul test set (arc_dataset_test.npz), con
onnxruntime su CPU.

I modelli restituiscono due logit (classe 0 = pre-arco, classe 1 = arco); la
predizione e' l'argmax (soglia 0.5 sulla probabilita' di arco; a parita' vince
la classe 0).

Output: accuratezza, Detection Rate (DR), False Positive Rate (FPR), FN, FP,
per FP32 e INT8, e fedelta' (percentuale di campioni su cui i due modelli
danno la stessa predizione).

I valori sono ottenuti su CPU host e NON sostituiscono la validazione sul
target (STM32N6), dove l'esecuzione INT8 (NPU) puo' differire da onnxruntime.

Percorsi di default: relativi alla radice del repository
(questo file sta in script/modelli_quantizzati/Inception/).

USO:
    python Confronto_modelli.py                 # tutto il test set (~2.5 min per modello su 1 CPU)
    python Confronto_modelli.py --n 1000        # sottoinsieme casuale (--seed)
    python Confronto_modelli.py --batch 64
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
ROOT = HERE.parents[2]  # script/modelli_quantizzati/Inception -> radice del repo

PATH_ORIGINALE = ROOT / "script" / "training" / "inception_time" / "results" / "inceptiontime.onnx"
PATH_QUANTIZZATO = HERE / "inceptiontime_PerChannel_quant_calibration_data_npz_1.onnx"  # da ST Edge AI
PATH_DATASET = ROOT / "dataset" / "dataset_new" / "arc_dataset_test.npz"


def predici(sess, X, batch):
    """X: (N, 1000). Restituisce le classi predette (argmax dei logit), shape (N,)."""
    nome = sess.get_inputs()[0].name
    out = np.empty(len(X), dtype=int)
    for i in tqdm(range(0, len(X), batch), total=(len(X) + batch - 1) // batch):
        xb = X[i:i + batch].astype(np.float32).reshape(-1, 1, X.shape[1])
        out[i:i + batch] = np.argmax(sess.run(None, {nome: xb})[0], axis=1)
    return out


def metriche(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    return dict(acc=(tp + tn) / len(y), dr=tp / max(tp + fn, 1), fpr=fp / max(tn + fp, 1), fn=fn, fp=fp)


def validazione_generale(args):
    data = np.load(args.dataset)
    X_test, y_test = data["X"], data["y"].astype(int)

    total = len(X_test)
    if args.n is None or args.n >= total:
        indices = np.arange(total)
    else:
        indices = np.sort(np.random.default_rng(args.seed).choice(total, args.n, replace=False))
    n_campioni = len(indices)
    X, y = X_test[indices], y_test[indices]

    sess_orig = ort.InferenceSession(str(args.original), providers=["CPUExecutionProvider"])
    sess_quant = ort.InferenceSession(str(args.quantized), providers=["CPUExecutionProvider"])

    print(f"Analisi su {n_campioni} campioni (FP32)...")
    pred_orig = predici(sess_orig, X, args.batch)
    print(f"Analisi su {n_campioni} campioni (INT8)...")
    pred_quant = predici(sess_quant, X, args.batch)

    mo, mq = metriche(y, pred_orig), metriche(y, pred_quant)
    fidelity = (pred_orig == pred_quant).mean() * 100

    print("\n" + "=" * 52)
    print("   VALIDAZIONE InceptionTime FP32 vs INT8 (CPU host)")
    print("=" * 52)
    print(f"  Campioni analizzati:  {n_campioni}")
    print(f"  {'':<10}{'Acc.':>9}{'DR':>9}{'FPR':>9}{'FN':>6}{'FP':>6}")
    for nome, m in (("FP32", mo), ("INT8", mq)):
        print(f"  {nome:<10}{100*m['acc']:>8.2f}%{100*m['dr']:>8.2f}%{100*m['fpr']:>8.2f}%{m['fn']:>6}{m['fp']:>6}")
    print(f"  Delta accuratezza (FP32 - INT8): {100*(mo['acc']-mq['acc']):+.2f} punti")
    print("-" * 52)
    print(f"  FEDELTA' (INT8 vs FP32): {fidelity:.2f}%")
    print("=" * 52)
    print("  Soglia: argmax dei logit (0.5). Valori da onnxruntime su CPU, non dal target.")


def main():
    p = argparse.ArgumentParser(description="Confronto FP32 vs INT8 per InceptionTime.")
    p.add_argument("--original", type=Path, default=PATH_ORIGINALE, help="inceptiontime.onnx (FP32)")
    p.add_argument("--quantized", type=Path, default=PATH_QUANTIZZATO, help="modello INT8 da ST Edge AI")
    p.add_argument("--dataset", type=Path, default=PATH_DATASET, help="arc_dataset_test.npz")
    p.add_argument("--n", type=int, default=None, help="numero di campioni (default: tutto il test set)")
    p.add_argument("--seed", type=int, default=42, help="seed per il sottoinsieme casuale")
    p.add_argument("--batch", type=int, default=64, help="dimensione del batch di inferenza")
    validazione_generale(p.parse_args())


if __name__ == "__main__":
    main()
