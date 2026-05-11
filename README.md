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

L'obiettivo è sviluppare un sistema di classificazione capace di identificare fault da arco elettrico DC in impianti fotovoltaici, ottimizzando contemporaneamente:

- accuratezza del modello
- robustezza rispetto al **data leakage**
- compatibilità con sistemi embedded
- inferenza **real-time** su hardware edge

L'intera pipeline è progettata per essere riproducibile, scalabile ed edge-ready.

---

# 📁 Struttura del Progetto

```text
DC-ARC-Fault/
│
├── dataset/
│   ├── dataset/                          # Dataset originale da Kaggle
│   ├── dataset_new/                      # Dataset preprocessato finale
│   ├── build_dataset_new.py              # Sliding window + preprocessing
│   ├── split_dataset.py                  # Split fisico train/test
│   └── dataset_leakage_check.py          # Verifica data leakage
│
├── scripts/
│   ├── training/
│   │   ├── inception_time/
│   │   │   ├── results/
│   │   │   └── train_inceptiontime.py    # Training + export ONNX + calibrazione
│   │   │
│   │   ├── mcnn/
│   │   │   ├── results/
│   │   │   ├── export_mcnn/
│   │   │   ├── train_mcnn.py             # Training MCNN
│   │   │   └── export_mcnn.py            # Export ONNX + calibrazione
│   │   │
│   │   └── multirockethydra/
│   │       └── train_arc_compare.py      # Training comparativo Ridge/Hydra/ArcNet
│   │
│   └── modelli_quantizzati/
│       ├── inception_time/
│       │   ├── Confronto_modelli.py
│       │   └── inceptiontime_PerChannel_quant_calibration_data_npz_1.onnx
│       │
│       └── mcnn/
│           ├── stm32_float/
│           ├── stm32_int/
│           ├── Confronto_modelli.py
│           ├── mcnn_PerChannel_quant_calibration_mcnn_npz_1.onnx
│           └── misura_risorse.py
│
├── pipeline.png
├── requirements.txt
└── README.md
```

---

# ⚙️ Installazione

## 1️⃣ Clonare il repository

```bash
git clone <repository-url>
cd DC-ARC-Fault
```

## 2️⃣ Creare un ambiente virtuale

```bash
# Linux / macOS
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

## 3️⃣ Installare le dipendenze

```bash
pip install -r requirements.txt
```

---

# 📚 Dataset

Il progetto utilizza il dataset pubblico:

**Photovoltaic (PV) DC Arc-Fault Library**
[IEEE DataPort](https://ieee-dataport.org/open-access/photovoltaic-pv-dc-arc-library)

Dal dataset originale vengono estratti: **Corrente**, **Tensione**, **Potenza**.

È disponibile anche una versione preprocessata su Kaggle:
[Kaggle – DC Arc Fault Dataset](https://www.kaggle.com/datasets/yassirsuarez/dc-arc-fault)

Scaricare il dataset e inserirlo in `dataset/dataset/`. Struttura attesa:

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

---

# 1️⃣ Preprocessing & Physical Split

I segnali grezzi vengono preprocessati tramite:

- **Sliding Window** per la segmentazione temporale
- **Normalizzazione** dei segnali
- **Costruzione di campioni supervisionati**

```bash
python build_dataset_new.py
python split_dataset.py
```

Lo split fisico (80% train / 20% test) viene eseguito **prima del training** per evitare overlap tra finestre temporali e temporal data leakage. Output: `.npz` per il training, `.csv` per analisi.

---

# 2️⃣ Data Leakage Verification

Verifica dell'assenza di data leakage tra training e test set:

- ricerca di duplicati
- verifica overlap temporale
- validazione dello split fisico

```bash
python dataset_leakage_check.py
```

---

# 3️⃣ Training & Feature Extraction

La cartella `scripts/training/` contiene le pipeline di training per tre architetture distinte.

## Modelli utilizzati

| Modello | Tipologia | Deploy STM32 |
|---|---|---|
| InceptionTime | Deep Learning (CNN temporale) | ✅ ONNX → ST Edge AI |
| MCNN | CNN multi-scala | ✅ ONNX → ST Edge AI |
| MultiRocket + varianti | Feature-based (ML classico) | ⚠️ Solo sperimentale |

---

## 🧠 InceptionTime

`train_inceptiontime.py` gestisce in un unico script:

- training con tsai + PyTorch su GPU
- metriche UL1699B (detection rate, false positive rate)
- analisi multi-soglia
- export `inceptiontime.onnx` (shape dinamica)
- generazione dataset di calibrazione per quantizzazione INT8

**Output prodotti:**

```text
risultati_inception/
├── inceptiontime.onnx            ← per ST Edge AI
├── inceptiontime_training.png    ← curve loss/accuracy
├── results_inceptiontime.png     ← confusion matrix, ROC, PR, score dist.
├── inceptiontime_report.txt      ← metriche complete
├── calibration_data.npz          ← calibrazione ST Edge AI (chiave: 'input')
├── calibration_data.npy          ← alternativo
├── calibration_data_flat.npy     ← fallback 2D
├── calibration_labels.npy        ← label per verifica
└── calibration_info.txt          ← istruzioni per ST Edge AI
```

```bash
python train_inceptiontime.py --epochs 50 --out risultati_inception
python train_inceptiontime.py --epochs 50 --n-cal 200 --out risultati_inception
```

---

## 🧠 MCNN

Il training produce un bundle `mcnn_bundle.pkl`. L'export viene eseguito separatamente:

```bash
python export_mcnn.py mcnn_bundle.pkl arc_dataset_train.npz --out export_mcnn
```

**Output prodotti:**

```text
export_mcnn/
├── mcnn.onnx                 ← feature extractor CNN per ST Edge AI
├── ridge_weights.h           ← classificatore Ridge in C
└── calibration_mcnn.npz      ← dataset calibrazione INT8
```

> **Nota:** viene esportata solo la CNN (feature extraction). Il classificatore Ridge rimane esterno e viene implementato separatamente in C tramite `ridge_weights.h`.

---

## 🚀 MultiRocket + varianti (analisi comparativa)

`train_arc_compare.py` addestra e confronta tre pipeline in un unico run:

| Pipeline | Descrizione |
|---|---|
| MultiRocket + Ridge | Feature extraction + classificatore lineare |
| MultiRocketHydra + Ridge | Kernel Hydra + Ridge interno |
| MultiRocket + PCA + ArcNet | Feature extraction + PCA + rete neurale |

```bash
python train_arc_compare.py train.npz test.npz
python train_arc_compare.py train.npz test.npz --models ridge arcnet
python train_arc_compare.py train.npz test.npz --pca-components 128 --epochs 40
```

**Output prodotti:**

```text
results/
├── ridge/
│   ├── bundle.pkl               modello + transformer + scaler
│   └── config.json              metriche + risorse stimate
├── hydra/
│   ├── bundle.pkl               modello completo
│   └── config.json              metriche
├── arcnet/
│   ├── bundle.pkl               transformer + scaler + pca + model state_dict
│   └── config.json              metriche + risorse stimate
├── comparison.json
├── comparison_metrics.csv
├── comparison_confusion.csv
├── comparison_resources.csv     risorse per componente
├── comparison_report.txt
└── comparison_plots.png         ROC, PR, confusion matrix, metriche
```

### ⚠️ Nota sul deploy STM32 — MultiRocket

Tutte e tre le pipeline MultiRocket usano MultiRocket come preprocessing. MultiRocket **non ha un export C/ONNX automatico**: nessuna delle tre pipeline è deployabile su STM32 senza reimplementare manualmente i kernel in C. Sono state usate esclusivamente per analisi comparativa offline.

La stima delle risorse prodotta dallo script separa:
- **costo preprocessing** (MultiRocket, comune a Ridge e ArcNet — da reimplementare)
- **costo classificatore finale** (confrontabile tra pipeline, ArcNet verificabile via ONNX)

---

# 4️⃣ Export ONNX & Quantizzazione

Dopo il training, i modelli vengono esportati e quantizzati per il deployment embedded.

| Modello | Export ONNX | Quantizzazione INT8 |
|---|---|---|
| InceptionTime | ✅ automatico nel training | ✅ via ST Edge AI |
| MCNN | ✅ `export_mcnn.py` | ✅ via ST Edge AI |
| MultiRocket | ❌ non disponibile | ❌ non applicabile |

---

# 5️⃣ Deployment su ST Edge AI

I modelli ONNX vengono validati tramite ST Edge AI Core / Developer Cloud.

Per InceptionTime e MCNN il flusso è:

1. Importa il modello `.onnx` in ST Edge AI
2. Carica il dataset di calibrazione `.npz` (chiave: `input`)
3. Seleziona quantizzazione INT8 Per-Channel
4. Avvia la quantizzazione → genera `model_int8.onnx`
5. Carica il modello quantizzato nel progetto
6. Esegui inferenza sul test set e confronta con FP32

---

# 📊 Risultati

## Accuratezza FP32 vs INT8

| Modello | FP32 Accuracy | INT8 Accuracy |
|---|---|---|
| InceptionTime | 98.62% | 98.97% |
| MCNN | 99.60% | 99.09% |

## Confronto MultiRocket (solo offline)

| Pipeline | Accuracy | Balanced Acc | F1 | ROC-AUC |
|---|---|---|---|---|
| MultiRocket + Ridge | 99.54% | 99.52% | 99.63% | 99.93% |
| MultiRocketHydra + Ridge | **99.65%** | **99.66%** | **99.72%** | 99.66% |
| MultiRocket + PCA + ArcNet | 99.55% | 99.52% | 99.65% | **99.99%** |

---

# 📈 Metriche monitorate

Durante deployment e benchmark:

- RAM usage
- Flash usage (interna ed esterna QSPI)
- latenza di inferenza
- accuracy post-quantizzazione

---

# 👥 Team

- **Lorenzo Meloccaro** — MSc, Università Politecnica delle Marche (UNIVPM)
- **Yassir Flavio Suarez Sanchez** — MSc, Università Politecnica delle Marche (UNIVPM)

---

# 📄 Licenza

Il dataset originale appartiene ai rispettivi autori del progetto pubblicato su IEEE DataPort.
Il codice del repository è distribuito secondo la licenza specificata nel progetto.