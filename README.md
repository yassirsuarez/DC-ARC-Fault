# DC-ARC-Fault ⚡

I DC arc fault rappresentano un rischio critico negli impianti fotovoltaici e sono tra le principali cause di incendio. La rilevazione affidabile è complessa perché i segnali sono rumorosi e non stazionari.

Questo lavoro affronta il problema in ottica **UL1699B**, sviluppando un sistema di rilevazione basato su deep learning progettato non solo per alta accuratezza, ma anche per essere deployabile su dispositivi embedded **STM32** in tempo reale.

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
│   │   ├── msrcfe/
│   │   │   ├── results/
│   │   │   ├── export_msrcfe/
│   │   │   ├── train_msrcfe.py           # Training MS-RCFE + Ridge
│   │   │   └── export_msrcfe.py          # Export ONNX + calibrazione
│   │   │
│   │   └── multirockethydra/
│   │       ├── train_arc_compare.py      # Training comparativo Ridge/Hydra/ArcNet
│   │       ├── arcnet/
│   │       ├── hydra/
│   │       ├── ridge/
│   │       └── altri file risultati e confronti
│   │
│   └── modelli_quantizzati/
│       ├── inception_time/
│       │   ├── confronto_modelli.py
│       │   └── inceptiontime_PerChannel_quant_calibration_data_npz_1.onnx
│       │
│       └── msrcfe/
│           ├── stm32_float/
│           ├── stm32_int/
│           ├── confronto_modelli.py
│           ├── msrcfe_PerChannel_quant_calibration_msrcfe_npz_1.onnx
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

| Modello | Sigla letteratura | Tipologia | Deploy STM32 |
|---|---|---|---|
| InceptionTime | InceptionTime | Deep Learning end-to-end (CNN temporale addestrata) | ✅ ONNX → ST Edge AI |
| MS-RCFE + Ridge | MS-RCFE | Fixed-kernel ROCKET multi-scala + classificatore lineare | ✅ ONNX → ST Edge AI |
| MultiRocket + varianti | ROCKET / MultiRocket | Feature-based (kernel casuali + ML classico) | ⚠️ Solo sperimentale |

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

## 🧠 MS-RCFE + Ridge

### Cos'è MS-RCFE

**MS-RCFE** è un estrattore di feature ultra-leggero per serie temporali. Utilizza kernel convoluzionali casuali **fissi** (non addestrati) per proiettare il segnale in uno spazio ad alta dimensione, facilitando la classificazione tramite un modello lineare. L'approccio è ispirato al paradigma **ROCKET** (Dempster et al., 2020).

#### Architettura

Il segnale viene elaborato attraverso **9 rami convoluzionali 1D paralleli** basati su diverse scale temporali:

```
Segnale x ∈ ℝ^T
      ↓
9 Conv1D parallele (kernel fissi, non addestrati)
  ├── kernel=3, dilation=1  →  max, mean  →  64 feat
  ├── kernel=3, dilation=2  →  max, mean  →  64 feat
  ├── kernel=3, dilation=4  →  max, mean  →  64 feat
  ├── kernel=5, dilation=1  →  max, mean  →  64 feat
  ├── kernel=5, dilation=2  →  max, mean  →  64 feat
  ├── kernel=5, dilation=4  →  max, mean  →  64 feat
  ├── kernel=9, dilation=1  →  max, mean  →  64 feat
  ├── kernel=9, dilation=2  →  max, mean  →  64 feat
  └── kernel=9, dilation=4  →  max, mean  →  64 feat
      ↓
Concatenazione  →  z ∈ ℝ^576
      ↓
Ridge Classifier (lineare)  →  ŷ ∈ {0, 1}
```

Per ogni ramo vengono estratte due statistiche globali (**Max** e **Mean**), producendo un vettore finale di **576 feature**. La classificazione viene affidata a un **Ridge Classifier** lineare, che rappresenta l'unica componente ottimizzata durante l'addestramento.

#### Vantaggi principali

- **Addestramento ultra-rapido:** non richiede backpropagation per la parte convoluzionale, in modo analogo a un Extreme Learning Machine (Huang et al., 2006).
- **Analisi multi-scala:** identifica pattern a diverse frequenze e risoluzioni temporali, in linea con l'approccio di MultiRocket (Tan et al., 2022).
- **Design per embedded:** l'uso di pesi fissi riduce drasticamente l'occupazione di memoria, rendendo il modello compatibile con tutta la gamma STM32.

### Training ed export

```bash
python train_msrcfe.py --train train.npz --test test.npz --out results_msrcfe
python export_msrcfe.py msrcfe_bundle.pkl arc_dataset_train.npz --out export_msrcfe
```

**Output prodotti:**

```text
export_msrcfe/
├── msrcfe.onnx               ← feature extractor (kernel fissi) per ST Edge AI
├── ridge_weights.h           ← classificatore Ridge in C
└── calibration_msrcfe.npz   ← dataset calibrazione INT8
```

> **Nota:** viene esportato solo il feature extractor (kernel fissi). Il classificatore Ridge viene implementato in C tramite `ridge_weights.h`. L'inferenza embedded consiste in tre passi: (1) applicazione dei 9 kernel fissi al segnale grezzo, (2) calcolo di max e mean per ogni kernel, (3) prodotto scalare con i pesi Ridge.

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

> **Nota deploy STM32:** tutte e tre le pipeline usano MultiRocket come preprocessing. MultiRocket non ha un export C/ONNX automatico, quindi nessuna delle tre è deployabile su STM32 senza reimplementare manualmente i kernel in C. Sono state usate esclusivamente per analisi comparativa offline.

---

# 4️⃣ Export ONNX & Quantizzazione

Dopo il training, i modelli vengono esportati e quantizzati per il deployment embedded.

| Modello | Export ONNX | Quantizzazione INT8 |
|---|---|---|
| InceptionTime | ✅ automatico nel training | ✅ via ST Edge AI |
| MS-RCFE + Ridge | ✅ `export_msrcfe.py` (solo feature extractor) | ✅ via ST Edge AI |
| MultiRocket + varianti | ❌ non disponibile | ❌ non applicabile |

---

# 5️⃣ Deployment su ST Edge AI

I modelli ONNX vengono validati tramite ST Edge AI Core / Developer Cloud.

Per InceptionTime e MS-RCFE il flusso è:

1. Importa il modello `.onnx` in ST Edge AI
2. Carica il dataset di calibrazione `.npz` (chiave: `input`)
3. Seleziona quantizzazione INT8 Per-Channel
4. Avvia la quantizzazione → genera `model_int8.onnx`
5. Carica il modello quantizzato nel progetto STM32CubeIDE
6. Esegui inferenza sul test set e confronta con FP32

---

# 📊 Risultati

## Accuratezza FP32 vs INT8

| Modello | FP32 Accuracy | INT8 Accuracy |
|---|---|---|
| InceptionTime | 99.60% | 98.97% |
| MS-RCFE + Ridge | 99.64% | 99.43% |

Entrambi i modelli mantengono un'accuratezza superiore al 98.9% dopo quantizzazione INT8, confermando la stabilità delle architetture scelte rispetto alla riduzione di precisione numerica.

## Confronto MultiRocket (solo offline)

| Pipeline | Accuracy | Balanced Acc | F1 | ROC-AUC |
|---|---|---|---|---|
| MultiRocket + Ridge | 99.54% | 99.52% | 99.63% | 99.93% |
| MultiRocketHydra + Ridge | **99.65%** | **99.66%** | **99.72%** | 99.66% |
| MultiRocket + PCA + ArcNet | 99.55% | 99.52% | 99.65% | **99.99%** |

---

## ⚡ Benchmark STM32 — InceptionTime

Testato su **STM32N6570-DK** con **Neural-ART NPU**, confrontando FP32 (CPU) vs INT8 (NPU).

| Metrica | FP32 (CPU) | INT8 (NPU) | Δ |
|---|---|---|---|
| Inference time | 6250.68 ms | 21.17 ms | ↓ ~295× |
| Throughput | 0.16 inf/s | 47.24 inf/s | ↑ ~295× |
| RAM totale | 3.175 MB | 1.762 MB | ↓ ~44% |
| Flash (pesi) | 1.562 MB | 399 KB | ↓ ~74% |

La latenza FP32 elevata (6.25 s) è attesa su questa board: InceptionTime è una rete profonda (~1.5 MB di pesi) e il Cortex-M55 senza NPU non dispone di accelerazione hardware per operazioni floating point su reti di questa dimensione. La quantizzazione INT8 abilita il pieno utilizzo della **Neural-ART NPU**, portando l'inferenza a 21 ms con un guadagno di ~295×. La precisione è preservata: cosine similarity INT8 vs FP32 ≈ **0.9999**.

---

## ⚡ Benchmark STM32 — MS-RCFE + Ridge

Testato su **STM32H7S78-DK** (Cortex-M7), confrontando la pipeline completa (feature extractor + Ridge) in FP32 vs INT8.

| Metrica | FP32 | INT8 | Δ |
|---|---|---|---|
| Inference time | 84.75 ms | 70.04 ms | ↓ ~1.2× |
| Throughput | 11.80 inf/s | 14.28 inf/s | ↑ ~1.2× |
| RAM totale | 135.61 KB | 141.11 KB | +4% |
| Flash totale | 23.56 KB | 30.81 KB | +30% |

**Dettaglio per componente:**

| Componente | FP32 | INT8 |
|---|---|---|
| Feature extractor (ONNX) | 0.333 ms | 0.684 ms |
| Ridge Classifier (C) | 0.006 ms | 0.006 ms |

La latenza è dominata dal feature extractor; il Ridge è computazionalmente trascurabile. Entrambe le configurazioni sono ampiamente sopra i requisiti real-time a 50 Hz. La quantizzazione INT8 su modelli piccoli non garantisce sempre un miglioramento della latenza — su questo modello introduce un lieve overhead sulla CNN ma riduce il consumo globale di memoria.

> **Nota sui valori di memoria:** i valori di Flash e RAM in tabella sono misurati da ST Edge AI su hardware reale e includono il runtime della libreria (~9 KB di overhead). La stima analitica dei soli pesi del modello (feature extractor + Ridge) è **14.7 KB Flash · 10.7 KB RAM** — il delta rispetto ai valori misurati è interamente dovuto a questo overhead.

---

## 📊 Stima risorse embedded — MultiRocket varianti

Le pipeline MultiRocket non sono state deployate su STM32 (preprocessing non esportabile in C). La tabella riporta una stima analitica basata sulle dimensioni dei modelli addestrati. Il costo dei kernel MultiRocket (~24.4 KB Flash) è comune a Ridge e ArcNet ed è incluso nelle stime.

| Pipeline | Flash interna | Flash esterna | RAM |
|---|---|---|---|
| MultiRocket + Ridge | ~607 KB | — | ~194 KB |
| MultiRocket + PCA + ArcNet | ~680 KB | ~48.6 MB (QSPI) | ~195 KB |

> MultiRocketHydra non è stimabile: i kernel Hydra interni non sono accessibili come array.

---

# 🏁 Conclusioni e Raccomandazioni

Il progetto ha valutato tre famiglie di modelli per la rilevazione di DC arc fault, con requisiti congiunti di alta accuratezza e compatibilità embedded.

**MS-RCFE + Ridge** è il modello raccomandato per il deployment su STM32 di fascia media (es. STM32H7). Combina un'accuratezza superiore al 99.4% anche dopo quantizzazione INT8, un'impronta hardware minima (23.6 KB Flash, 135.6 KB RAM misurati su hardware) e latenza di 70 ms in INT8 — ben entro i requisiti real-time a 50 Hz. La pipeline è completamente esportabile tramite ST Edge AI senza modifiche al firmware.

**InceptionTime** è la scelta corretta quando è disponibile una board con NPU dedicata (es. STM32N6570-DK con Neural-ART). In quel caso il guadagno di latenza è di ~295× rispetto a FP32, rendendo praticabile anche l'inferenza continua. Su board Cortex-M senza NPU la latenza FP32 è proibitiva per uso real-time.

**MultiRocket e varianti** offrono le metriche di classificazione più alte (F1 fino a 99.72% con Hydra) ma non sono deployabili su STM32 senza reimplementare manualmente il preprocessing in C. Sono stati usati esclusivamente come baseline comparativa offline per contestualizzare le scelte architetturali.

---

# 👥 Team

- **Lorenzo Meloccaro** — MSc, Università Politecnica delle Marche (UNIVPM)
- **Yassir Flavio Suarez Sanchez** — MSc, Università Politecnica delle Marche (UNIVPM)

---

# 📄 Licenza

Il dataset originale appartiene ai rispettivi autori del progetto pubblicato su IEEE DataPort.
Il codice del repository è distribuito secondo la licenza specificata nel progetto.