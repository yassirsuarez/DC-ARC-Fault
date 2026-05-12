import numpy as np
import onnxruntime as ort
import pickle
from sklearn.linear_model import RidgeClassifier
from tqdm import tqdm

# --- CONFIGURAZIONE PERCORSI ---
PATH_ORIGINALE   = r"C:\Users\Asus\Desktop\progetto_manutenzione\script\training\msrcfe\export_msrcfe\msrcfe.onnx"
PATH_QUANTIZZATO = r"C:\Users\Asus\Desktop\progetto_manutenzione\script\modelli_quantizzati\msrcfe\msrcfe_PerChannel_quant_calibration_msrcfe_npz_1.onnx"   # ← scaricato da ST Edge AI
PATH_DATASET     = r"C:\Users\Asus\Desktop\progetto_manutenzione\dataset\dataset_new\arc_dataset_test.npz"
PATH_BUNDLE      = r"C:\Users\Asus\Desktop\progetto_manutenzione\script\training\msrcfe\results\msrcfe_bundle.pkl"


def predici(sess, ridge, sample_1d):
    """
    sample_1d: numpy array (1000,) — singolo segnale normalizzato
    Restituisce: 0 o 1
    """
    # Shape (1, 1, 1000) — attesa dal modello ONNX
    x = sample_1d.astype(np.float32).reshape(1, 1, 1000)
    
    # Estrazione features con Hydra
    input_name = sess.get_inputs()[0].name
    features = sess.run(None, {input_name: x})[0]   # (1, 576)
    
    # Classificazione con Ridge
    pred = ridge.predict(features)[0]
    return int(pred)


def validazione_hydra(n_campioni=1000):
    # 1. Caricamento dataset
    data   = np.load(PATH_DATASET)
    X_test = data["X"]
    y_test = data["y"]

    indices = range(len(X_test))
    n_campioni = len(X_test)

    # 2. Caricamento modelli
    sess_orig  = ort.InferenceSession(PATH_ORIGINALE,   providers=["CPUExecutionProvider"])
    sess_quant = ort.InferenceSession(PATH_QUANTIZZATO, providers=["CPUExecutionProvider"])

    with open(PATH_BUNDLE, "rb") as f:
        bundle = pickle.load(f)
    ridge = bundle["ridge"]

    # 3. Inferenza
    corrette_orig  = 0
    corrette_quant = 0
    match_modelli  = 0

    print(f"Analisi su {n_campioni} campioni...")

    for idx in tqdm(indices):
        sample       = X_test[idx]       # (1000,)
        label_reale  = y_test[idx]

        pred_orig  = predici(sess_orig,  ridge, sample)
        pred_quant = predici(sess_quant, ridge, sample)

        if pred_orig  == label_reale: corrette_orig  += 1
        if pred_quant == label_reale: corrette_quant += 1
        if pred_orig  == pred_quant:  match_modelli  += 1

    # 4. Risultati
    acc_orig  = corrette_orig  / n_campioni * 100
    acc_quant = corrette_quant / n_campioni * 100
    fidelity  = match_modelli  / n_campioni * 100
    delta     = acc_orig - acc_quant

    print("\n" + "="*45)
    print("        VALIDAZIONE HYDRA FP32 vs INT8")
    print("="*45)
    print(f"  Campioni analizzati:     {n_campioni}")
    print(f"  Accuratezza FP32:        {acc_orig:.2f}%")
    print(f"  Accuratezza INT8:        {acc_quant:.2f}%")
    print(f"  Delta accuratezza:       {delta:+.2f}%")
    print("-"*45)
    print(f"  FEDELTÀ (INT8 vs FP32):  {fidelity:.2f}%")
    print("="*45)

    if abs(delta) < 1.0:
        print("  ✔ Quantizzazione eccellente — perdita < 1%")
    elif abs(delta) < 3.0:
        print("  ⚠ Quantizzazione accettabile — perdita < 3%")
    else:
        print("  ✘ Perdita significativa — valuta più campioni di calibrazione")


if __name__ == "__main__":
    validazione_hydra(n_campioni=10000)