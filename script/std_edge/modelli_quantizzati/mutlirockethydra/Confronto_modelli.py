import numpy as np
import onnxruntime as ort
from tqdm import tqdm

# --- CONFIGURAZIONE PERCORSI ---
PATH_ORIGINALE = r"C:\Users\Asus\Desktop\progetto_manutenzione\script\onnx\export_mrh\hydra.onnx"
PATH_QUANTIZZATO = r"C:\Users\Asus\Desktop\progetto_manutenzione\script\std_edge\modelli_quantizzati\mutlirockethydra\hydra_PerChannel_quant_calibration_data_hydra_npz_1.onnx"
PATH_DATASET = r"C:\Users\Asus\Desktop\progetto_manutenzione\dataset\dataset_new\arc_dataset_new.npz"

def validazione_generale(n_campioni=200):
    # 1. Caricamento Dataset
    data = np.load(PATH_DATASET)
    X_test, y_test = data['X'], data['y']
    
    indices = np.random.choice(len(X_test), n_campioni, replace=False)
    
    # 2. Sessioni ONNX
    sess_orig = ort.InferenceSession(PATH_ORIGINALE, providers=['CPUExecutionProvider'])
    sess_quant = ort.InferenceSession(PATH_QUANTIZZATO, providers=['CPUExecutionProvider'])
    
    # DEBUG input modello
    print("Input richiesti dal modello FP32:")
    for inp in sess_orig.get_inputs():
        print(" -", inp.name, inp.shape)

    print("\nInput richiesti dal modello INT8:")
    for inp in sess_quant.get_inputs():
        print(" -", inp.name, inp.shape)

    corrette_orig = 0
    corrette_quant = 0
    match_modelli = 0
    
    print(f"\nAnalisi in corso su {n_campioni} campioni...")

    for idx in tqdm(indices):
        # ✔ input raw corretto
        sample = X_test[idx].astype(np.float32).reshape(1, 1, 1000)

        # ✔ diff SENZA prepend (produce 999 come richiesto dal modello)
        sample_diff = np.diff(sample, axis=2).astype(np.float32)

        label_reale = y_test[idx]

        # ✔ input richiesti da Hydra
        input_dict = {
            "input": sample,
            "input_diff": sample_diff
        }

        # Inferenza
        pred_orig = np.argmax(sess_orig.run(None, input_dict)[0])
        pred_quant = np.argmax(sess_quant.run(None, input_dict)[0])

        # metriche
        if pred_orig == label_reale:
            corrette_orig += 1
        if pred_quant == label_reale:
            corrette_quant += 1
        if pred_orig == pred_quant:
            match_modelli += 1

    acc_orig = (corrette_orig / n_campioni) * 100
    acc_quant = (corrette_quant / n_campioni) * 100
    fidelity = (match_modelli / n_campioni) * 100

    print("\n" + "="*40)
    print("   SINTESI GENERALE VALIDAZIONE HYDRA")
    print("="*40)
    print(f"Campioni analizzati:     {n_campioni}")
    print(f"Accuratezza FP32:        {acc_orig:.2f}%")
    print(f"Accuratezza INT8:        {acc_quant:.2f}%")
    print("-"*40)
    print(f"FEDELTÀ (INT8 vs FP32):  {fidelity:.2f}%")
    print("="*40)


if __name__ == "__main__":
    validazione_generale(n_campioni=10000)