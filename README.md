# DC-ARC-Fault ⚡

Pipeline completa per la rilevazione di **DC Arc Fault** in impianti fotovoltaici tramite tecniche di **Deep Learning** e **Time Series Classification**, con supporto al deployment su dispositivi embedded mediante **ST Edge AI**.

Il progetto integra:

- preprocessing avanzato di serie temporali
- controllo del **data leakage**
- training di modelli **Deep Learning** e **feature-based**
- esportazione in formato **ONNX**
- quantizzazione **INT8**
- benchmark edge-oriented su **STM32**

---

# 🧠 Obiettivo del Progetto

L’obiettivo del progetto è sviluppare un sistema di classificazione capace di identificare fault da arco elettrico DC in impianti fotovoltaici, ottimizzando contemporaneamente:

- accuratezza del modello
- robustezza rispetto al **data leakage**
- compatibilità con sistemi embedded
- inferenza **real-time** su hardware edge

L’intera pipeline è progettata per essere:

- riproducibile
- scalabile
- edge-ready

---

# 📁 Struttura del Progetto

```text
DC-ARC-Fault/
│
├── dataset/                                  # Dataset e preprocessing
│   ├── dataset/                              # Dataset originale scaricato da Kaggle
│   ├── dataset_new/                          # Dataset preprocessato finale
│   │
│   ├── build_dataset_new.py                  # Sliding window + preprocessing
│   ├── split_dataset.py                      # Split fisico train/test
│   └── dataset_leakage_check.py              # Verifica data leakage
│
├── script/
│   │
│   ├── training/                             # FASE 1 — Training modelli
│   │   │
│   │   ├── inception_time/
│   │   │   ├── results/
│   │   │   ├── train_inceptiontime.py
│   │   │   └── *.keras
│   │   │
│   │   ├── mcnn/
│   │   │   ├── results/
│   │   │   └── train_mcnn.py
│   │   │
│   │   └── multirockethydra/
│   │
│   ├── onnx/                                 # FASE 2 — Export ONNX
│   │   │
│   │   ├── export_mcnn/
│   │   ├── export_mcnn.py
│   │   └── export_mr.py
│   │
│   └── st_edge/                              # FASE 3 — Edge AI deployment
│       │
│       ├── dataset_calibrazione/             # Dataset per quantizzazione
│       │   ├── calibration_inceptiontime/
│       │   ├── dataset_calibrazione_inception.py
│       │   └── dataset_calibrazione_mrh.py
│       │
│       └── modelli_quantizzati/
│           │
│           ├── inception/
│           │   ├── Confronto_modelli.py
│           │   └── inceptiontime_PerChannel_quant_*.onnx
│           │
│           ├── mcnn/
│           │   ├── Confronto_modelli.py
│           │   └── hydra_PerChannel_quant_*.onnx
│           │
│           └── multirocket/
│               ├── Confronto_modelli.py
│               └── *.onnx
│
├── pipeline.png                              # Schema pipeline
├── requirements.txt                          # Dipendenze
└── README.md
```
---

# ⚙️ Installazione

## 1️⃣ Clonare il repository

```bash
git clone <repository-url>
cd DC-ARC-Fault
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

[IEEE DataPort – Photovoltaic (PV) DC Arc-Fault Library](https://ieee-dataport.org/open-access/photovoltaic-pv-dc-arc-library)

Dal dataset originale vengono estratti i seguenti segnali elettrici:

- **Corrente (Current)**
- **Tensione (Voltage)**
- **Potenza (Power)**

È disponibile anche una versione preprocessata su Kaggle:

[Kaggle – DC Arc Fault Dataset](https://www.kaggle.com/datasets/yassirsuarez/dc-arc-fault)

## Download del dataset

Scaricare il dataset da Kaggle e inserirlo nella cartella:

```text
dataset/dataset/
```

Struttura attesa:

```text
dataset/
└── dataset/
    ├── E001_0050V_00.50A_G001
    ├── E001_0050V_00.50A_G002
    └── ...
```

---

# 🔄 Pipeline del Progetto

![Pipeline](pipeline.png)

La pipeline è suddivisa in **sei fasi principali**, dalla preparazione dei dati fino al deployment edge.

---

# 1️⃣ Preprocessing & Physical Split

In questa fase i segnali grezzi vengono preprocessati tramite:

- **Sliding Window** per la segmentazione temporale
- **Normalizzazione** dei segnali
- **Costruzione di campioni supervisionati**

## Script principali

```bash
build_dataset_new.py
split_dataset.py
```

## Output generato

- **Training Set** → 80%
- **Test Set** → 20%

Lo split fisico viene eseguito **prima del training** per evitare:

- overlap tra finestre temporali
- contaminazione tra train e test
- **temporal data leakage**

Formati di output:

- `.npz` → training
- `.csv` → analisi/debug

---

# 2️⃣ Data Leakage Verification

In questa fase viene verificata l’assenza di **data leakage** tra training e test set.

Controlli effettuati:

- ricerca di duplicati
- verifica overlap temporale
- validazione dello split fisico

## Script principale

```bash
dataset_leakage_check.py
```

---

# 3️⃣ Model Training

Addestramento dei modelli di classificazione:

- **InceptionTime**
- **MCNN**
- **MultiRocket**

Directory:

```text
script/training/
```

Ogni modello salva:

- checkpoint (`.keras`)
- metriche
- risultati finali

---

# 4️⃣ ONNX Export

Conversione dei modelli in formato **ONNX** per deployment multipiattaforma.

Script disponibili in:

```text
script/onnx/
```

Output:

- modelli `.onnx`

---

# 5️⃣ INT8 Quantization

Quantizzazione dei modelli tramite dataset di calibrazione per ridurre:

- memoria
- latenza
- consumo energetico

Formato finale:

- `*_quant.onnx`

---

# 6️⃣ Edge Deployment (STM32)

Benchmark dei modelli quantizzati tramite **ST Edge AI** su hardware embedded STM32.

Metriche valutate:

- tempo di inferenza
- utilizzo RAM
- utilizzo Flash
- accuratezza post-quantizzazione

Directory:

```text
script/st_edge/
```

---

# 📊 Risultati

Il progetto confronta diversi modelli considerando:

- accuracy
- latency
- memory footprint
- deployability su edge

---

# 👨‍💻 Tecnologie Utilizzate

- Python
- TensorFlow / Keras
- ONNX
- NumPy / Pandas
- Scikit-learn
- ST Edge AI
- STM32

---
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
| InceptionTime | 98.62%| 98.97% |
| MCNN | 99.60% | 99.09% |
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