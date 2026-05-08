# DC-ARC-Fault ⚡

Pipeline completa per la rilevazione di **DC Arc Fault** in impianti fotovoltaici tramite tecniche di **Deep Learning** e **Time Series Classification**, con supporto al deployment su dispositivi embedded mediante **ST Edge AI**.

Il progetto integra:

- preprocessing avanzato di serie temporali
- controllo del data leakage
- training di modelli Deep Learning e feature-based
- esportazione ONNX
- quantizzazione INT8
- benchmark edge-oriented su STM32

---

# 🧠 Obiettivo del Progetto

L'obiettivo del progetto è sviluppare un sistema di classificazione in grado di identificare fault da arco elettrico DC in impianti fotovoltaici, ottimizzando contemporaneamente:

- accuratezza del modello
- robustezza contro il data leakage
- compatibilità con sistemi embedded
- inferenza real-time su hardware edge

L'intera pipeline è progettata per essere:

- riproducibile
- scalabile
- edge-ready

---

# 📁 Struttura del Progetto

```text
PROGETTO_MANUTENZIONE/
│
├── dataset/                                      # DATA MANAGEMENT
│   ├── dataset/                                  # Dataset Scaricato da kaggle
│   ├── dataset_new/                              # Dataset preprocessato finale
│   │
│   ├── build_dataset_new.py                      # Windowing e creazione dataset
│   ├── split_dataset.py                          # Split fisico train/test
│   ├── check.py                                  # Verifica leakage e integrità
│   └── Check2.py                                 # Controlli aggiuntivi dataset
│
├── script/
│   │
│   ├── training/                                 # FASE 1 — TRAINING
│   │   │
│   │   ├── inception_time/
│   │   │   ├── risultati_finali/
│   │   │   ├── *.keras                           # Checkpoint modelli
│   │   │
│   │   ├── mcnn/
│   │   │   └── ...
│   │   │
│   │   └── multirockethydra/
│   │       └── ...
│   │
│   ├── onnx/                                     # FASE 2 — EXPORT & QUANTIZATION
│   │   │
│   │   ├── dataset_ottimizzazione/
│   │   │   └── calibration_inceptiontime/
│   │   │
│   │   ├── export_mcnn/
│   │   │
│   │   ├── dataset_calibrazione_inception.py
│   │   ├── dataset_calibrazione_mrh.py
│   │   ├── export_mcnn.py
│   │   └── export_mrh.py
│   │
│   └── std_edge/modelli_quantizzati/             # FASE 3 — EDGE AI DEPLOYMENT
│       │
│       ├── Inception/
│       │   ├── Confronto_modelli.py
│       │   └── inceptiontime_PerChannel_quant_*.onnx
│       │
│       ├── mcnn/
│       │   ├── Confronto_modelli.py
│       │   └── hydra_PerChannel_quant_*.onnx
│       │
│       └── multirockethydra/
│           └── Confronto_modelli.py
│
├── pipeline.png                                  # Schema della pipeline
├── requirements.txt                              # Dipendenze Python
└── README.md
```

---

# ⚙️ Installazione

## 1️⃣ Clonare il repository

```bash
git clone <repository-url>
cd PROGETTO_MANUTENZIONE
```

---

## 2️⃣ Creare un ambiente virtuale

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## 3️⃣ Installare le dipendenze

```bash
pip install -r requirements.txt
```

---

# 📚 Dataset

Il progetto utilizza il dataset pubblico:

## Photovoltaic (PV) DC Arc-Fault Library

https://ieee-dataport.org/open-access/photovoltaic-pv-dc-arc-library

Dal dataset originale vengono estratti i segnali di:

- corrente
- tensione
- potenza

Dataset preprocessato disponibile anche su Kaggle:

https://www.kaggle.com/datasets/yassirsuarez/dc-arc-fault

I segnali vengono successivamente trasformati in finestre temporali supervisionate utilizzabili dai modelli di classificazione.

---

# 🔄 Pipeline del Progetto

![Pipeline](pipeline.png)

La pipeline è suddivisa in sei fasi principali.

---

# 1️⃣ Data Management

Il dataset grezzo viene:

- validato
- organizzato
- convertito in una struttura coerente per il training

Questa fase comprende:

- caricamento dei segnali originali
- verifica integrità dati
- preparazione delle serie temporali
- controllo preliminare leakage

Script principali:

```bash
python check.py
python Check2.py
```

---

# 2️⃣ Preprocessing & Physical Split

I segnali vengono preprocessati tramite:

- finestra scorrevole
- normalizzazione
- costruzione sample supervisionati

Script principale:

```bash
python build_dataset_new.py
```

Successivamente viene effettuato uno split fisico train/test:

```bash
python split_dataset.py
```

## Dataset generati

- Train Set → 80%
- Test Set → 20%

Lo split fisico viene eseguito prima del training per evitare:

- overlap tra finestre
- contaminazione train/test
- leakage temporale

I dataset finali vengono salvati nei formati:

- `.npz`
- `.csv`

---

# 3️⃣ Training & Feature Extraction

La cartella:

```text
script/training/
```

contiene differenti approcci di classificazione.

| Modello | Tipologia |
|---|---|
| InceptionTime | Deep Learning |
| MCNN | CNN Multi-scala |
| MultiRocket + Hydra | Feature-based |

Durante il training vengono eseguiti:

- controllo anti leakage
- validazione
- normalizzazione globale
- feature extraction
- salvataggio checkpoint

---

# 4️⃣ Export ONNX & Quantizzazione

I modelli addestrati vengono esportati in formato:

```text
ONNX (FP32)
```

Script disponibili:

```bash
python export_mcnn.py
python export_mrh.py
```

Dataset di calibrazione:

```bash
python dataset_calibrazione_inception.py
python dataset_calibrazione_mrh.py
```

La quantizzazione viene effettuata in:

- INT8 Per-Channel
- configurazioni ottimizzate per STM32

---

# 5️⃣ Deployment su ST Edge AI

I modelli quantizzati vengono validati tramite:

- ST Edge AI Core
- ST Edge AI Developer Cloud

Configurazioni benchmark:

| Configurazione | Obiettivo |
|---|---|
| FP32 | Accuratezza massima |
| INT8 | Ottimizzazione embedded |

Metriche monitorate:

- RAM usage
- Flash usage
- inferenza
- accuratezza post-quantizzazione

Script benchmark:

```bash
Confronto_modelli.py
```

---

# 6️⃣ Valutazione Finale

Il test set rimane completamente indipendente dall'intero processo di training e calibrazione.

Vengono confrontati:

- modello FP32
- modello INT8

Metriche finali:

- Accuracy
- Precision
- Recall
- F1-score
- impatto quantizzazione

---

# 📊 Risultati

Di seguito sono riportate le accuratezze ottenute dai principali modelli utilizzati nel progetto.

| Modello | FP32 Accuracy | INT8 Accuracy |
|---|---|---|
| InceptionTime | -- | -- |
| MCNN | -- | -- |
| MultiRocket + Hydra | -- | -- |


---

# 🛠️ Tecnologie Utilizzate

## Machine Learning / Deep Learning

- PyTorch
- NumPy
- Scikit-learn
- ONNX
- MultiRocket
- Hydra
- InceptionTime

## Edge AI

- ST Edge AI
- STM32
- Quantizzazione INT8
- ONNX Runtime

---

# 🚀 Esempi di Utilizzo

## Costruzione Dataset

```bash
python build_dataset_new.py
```

## Split Train/Test

```bash
python split_dataset.py
```

## Export ONNX

```bash
python export_mcnn.py
```

## Training

```bash
python train_multirocket_fixed.py \
    dataset_train.npz \
    dataset_test.npz \
    --out ./results
```

---

# 🎯 Obiettivi Tecnici

- rilevazione real-time archi DC
- robustezza contro leakage
- deployment embedded STM32
- ottimizzazione memoria/inferenza
- pipeline riproducibile
- inferenza edge AI

---

# 👥 Team

- Lorenzo Meloccaro — MSc, Università Politecnica delle Marche (UNIVPM)
- Yassir Flavio Suarez Sanchez — MSc, Università Politecnica delle Marche (UNIVPM)

---

# 📄 Licenza

Il dataset originale appartiene ai rispettivi autori del progetto pubblicato su IEEE DataPort.

Il codice del repository è distribuito secondo la licenza specificata nel progetto.