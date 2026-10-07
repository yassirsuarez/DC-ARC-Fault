"""
Confronto_modelli.py  (MS-RCFE)
===============================
Confronta il modello MS-RCFE in FP32 (ONNX originale) con la versione INT8
quantizzata da ST Edge AI, sul test set indipendente.

Entrambe le versioni estraggono le 576 feature dalla finestra; la
classificazione e' affidata allo stesso Ridge (bundle .pkl), quindi le
differenze dipendono solo dalla quantizzazione dell'estrattore.

Output:
    accuratezza FP32, accuratezza INT8, delta, e fedelta' (percentuale di
    campioni su cui i due modelli danno la stessa predizione).

Percorsi di default: relativi alla radice del repository
(questo file sta in script/modelli_quantizzati/msrcfe/).

Requisiti: onnxruntime, numpy, tqdm, scikit-learn e torch (il bundle .pkl
contiene i pesi dell'estrattore come tensori PyTorch, quindi serve torch
anche solo per leggere il Ridge).

USO:
    python Confronto_modelli.py                 # tutto il test set
    python Confronto_modelli.py --n 1000        # sottoinsieme casuale
    python Confronto_modelli.py --original percorso/msrcfe.onnx \
                                --quantized percorso/msrcfe_int8.onnx
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tqdm import tqdm

# --- PERCORSI DI DEFAULT (relativi al repository) ---
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # script/modelli_quantizzati/msrcfe -> radice del repo

PATH_ORIGINALE = ROOT / "script" / "training" / "msrcfe" / "export_msrcfe" / "msrcfe.onnx"
# modello INT8 scaricato da ST Edge AI
PATH_QUANTIZZATO = HERE / "msrcfe_PerChannel_quant_calibration_msrcfe_npz_1.onnx"
PATH_DATASET = ROOT / "dataset" / "dataset_new" / "arc_dataset_test.npz"
PATH_BUNDLE = ROOT / "script" / "training" / "msrcfe" / "results" / "msrcfe_bundle.pkl"


def predici(sess, input_name, ridge, sample_1d):
    """
    sample_1d: array (1000,) - singolo segnale normalizzato.
    Restituisce 0 o 1.
    """
    # Shape (1, 1, 1000), attesa dal modello ONNX
    x = sample_1d.astype(np.float32).reshape(1, 1, -1)

    # Estrazione delle 576 feature con l'estrattore convoluzionale multi-scala
    features = sess.run(None, {input_name: x})[0]  # (1, 576)

    # Classificazione con Ridge
    return int(ridge.predict(features)[0])


def validazione_msrcfe(args):
    # 1. Caricamento dataset
    data = np.load(args.dataset)
    X_test, y_test = data["X"], data["y"]

    total = len(X_test)
    if args.n is None or args.n >= total:
        indices = np.arange(total)
    else:
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(total, args.n, replace=False)
    n_campioni = len(indices)

    # 2. Caricamento modelli
    sess_orig = ort.InferenceSession(str(args.original), providers=["CPUExecutionProvider"])
    sess_quant = ort.InferenceSession(str(args.quantized), providers=["CPUExecutionProvider"])
    name_orig = sess_orig.get_inputs()[0].name
    name_quant = sess_quant.get_inputs()[0].name

    with open(args.bundle, "rb") as f:
        ridge = pickle.load(f)["ridge"]

    # 3. Inferenza
    corrette_orig = 0
    corrette_quant = 0
    match_modelli = 0

    print(f"Analisi su {n_campioni} campioni...")

    for idx in tqdm(indices):
        sample = X_test[idx]  # (1000,)
        label_reale = y_test[idx]

        pred_orig = predici(sess_orig, name_orig, ridge, sample)
        pred_quant = predici(sess_quant, name_quant, ridge, sample)

        if pred_orig == label_reale:
            corrette_orig += 1
        if pred_quant == label_reale:
            corrette_quant += 1
        if pred_orig == pred_quant:
            match_modelli += 1

    # 4. Risultati
    acc_orig = corrette_orig / n_campioni * 100
    acc_quant = corrette_quant / n_campioni * 100
    fidelity = match_modelli / n_campioni * 100
    delta = acc_orig - acc_quant

    print("\n" + "=" * 45)
    print("      VALIDAZIONE MS-RCFE FP32 vs INT8")
    print("=" * 45)
    print(f"  Campioni analizzati:     {n_campioni}")
    print(f"  Accuratezza FP32:        {acc_orig:.2f}%")
    print(f"  Accuratezza INT8:        {acc_quant:.2f}%")
    print(f"  Delta accuratezza:       {delta:+.2f}%")
    print("-" * 45)
    print(f"  FEDELTÀ (INT8 vs FP32):  {fidelity:.2f}%")
    print("=" * 45)

    if abs(delta) < 1.0:
        print("  ✔ Quantizzazione eccellente — perdita < 1%")
    elif abs(delta) < 3.0:
        print("  ⚠ Quantizzazione accettabile — perdita < 3%")
    else:
        print("  ✘ Perdita significativa — valuta più campioni di calibrazione")


def main():
    p = argparse.ArgumentParser(description="Confronto FP32 vs INT8 per MS-RCFE + Ridge.")
    p.add_argument("--original", type=Path, default=PATH_ORIGINALE, help="msrcfe.onnx (FP32)")
    p.add_argument("--quantized", type=Path, default=PATH_QUANTIZZATO, help="modello INT8 da ST Edge AI")
    p.add_argument("--dataset", type=Path, default=PATH_DATASET, help="arc_dataset_test.npz")
    p.add_argument("--bundle", type=Path, default=PATH_BUNDLE, help="msrcfe_bundle.pkl (contiene il Ridge)")
    p.add_argument("--n", type=int, default=None, help="numero di campioni (default: tutto il test set)")
    p.add_argument("--seed", type=int, default=42, help="seed per il sottoinsieme casuale")
    validazione_msrcfe(p.parse_args())


if __name__ == "__main__":
    main()
