import numpy as np
import onnxruntime as ort
from tqdm import tqdm

# --- CONFIGURAZIONE PERCORSI ---
PATH_ORIGINALE = r"C:\Users\Asus\Desktop\progetto_manutenzione\script\training\inception_time\results\inceptiontime.onnx"
PATH_QUANTIZZATO = r"C:\Users\Asus\Desktop\progetto_manutenzione\script\std_edge\modelli_quantizzati\Inception\inceptiontime_PerChannel_quant_calibration_data_npz_1.onnx"
PATH_DATASET = r"C:\Users\Asus\Desktop\progetto_manutenzione\dataset\dataset_new\arc_dataset_test.npz"


def validazione_generale(n_campioni=None):
    # 1. Caricamento Dataset
    data = np.load(PATH_DATASET)
    X_test, y_test = data['X'], data['y']

    total_samples = len(X_test)

    # Se non specifichi n_campioni → usa tutto
    if n_campioni is None or n_campioni > total_samples:
        indices = np.arange(total_samples)
        n_campioni = total_samples
    else:
        indices = np.random.choice(total_samples, n_campioni, replace=False)

    # 2. Inizializzazione Sessioni
    sess_orig = ort.InferenceSession(PATH_ORIGINALE, providers=['CPUExecutionProvider'])
    sess_quant = ort.InferenceSession(PATH_QUANTIZZATO, providers=['CPUExecutionProvider'])

    input_name = sess_orig.get_inputs()[0].name

    # Contatori
    corrette_orig = 0
    corrette_quant = 0
    match_modelli = 0

    print(f"Analisi su {n_campioni} campioni...")

    for idx in tqdm(indices):
        sample = X_test[idx].astype(np.float32).reshape(1, 1, -1)
        label_reale = y_test[idx]

        # Inferenza
        pred_orig = np.argmax(sess_orig.run(None, {input_name: sample})[0])
        pred_quant = np.argmax(sess_quant.run(None, {input_name: sample})[0])

        # Metriche
        if pred_orig == label_reale:
            corrette_orig += 1
        if pred_quant == label_reale:
            corrette_quant += 1
        if pred_orig == pred_quant:
            match_modelli += 1

    # 3. Risultati
    acc_orig = (corrette_orig / n_campioni) * 100
    acc_quant = (corrette_quant / n_campioni) * 100
    fidelity = (match_modelli / n_campioni) * 100

    print("\n" + "="*40)
    print("       SINTESI GENERALE VALIDAZIONE")
    print("="*40)
    print(f"Campioni analizzati:     {n_campioni}")
    print(f"Accuratezza FP32:        {acc_orig:.2f}%")
    print(f"Accuratezza INT8:        {acc_quant:.2f}%")
    print("-"*40)
    print(f"FEDELTÀ (INT8 vs FP32):  {fidelity:.2f}%")


if __name__ == "__main__":
    validazione_generale()  # ← usa tutto il dataset