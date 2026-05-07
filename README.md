# DC-ARC-Fault

Pipeline completa per la rilevazione di archi elettrici DC in impianti fotovoltaici tramite tecniche di Deep Learning e Time Series Classification, con supporto al deployment su dispositivi embedded attraverso ST Edge AI.

Il progetto include:

- costruzione del dataset
- preprocessing e controllo leakage
- training di modelli deep learning e feature-based
- esportazione ONNX
- quantizzazione INT8
- validazione su hardware edge

---

# 📁 Struttura del Progetto

```text
DC-ARC-Fault/
│
├── dataset/
│   ├── dataset/                         # Dataset grezzo originale
│   ├── dataset_new/                     # Dataset preprocessato finale
│   │
│   ├── build_dataset_new.py             # Creazione finestre temporali
│   ├── split_dataset.py                 # Split fisico train/test
│   ├── check.py                         # Controlli anti data leakage
│   └── Check2.py                        # Controlli aggiuntivi
│
├── script/
│   ├── onnx/
│   │   ├── dataset_otimizzazione/
│   │   │   ├── calibration_inceptiontime/
│   │   │   └── dataset_calibrazione_inception.py
│   │   │
│   │   ├── export_mcnn/
│   │   └── export_mcnn.py               # Export modelli in ONNX
│   │
│   ├── std_edge/
│   │   └── modelli_quantizzati/
│   │       ├── Inception/
│   │       └── mcnn/
│   │
│   └── training/
│       ├── inception_time/              # Training InceptionTime
│       ├── mcnn/                        # Training MCNN
│       └── multirockethydra/            # Training MultiRocket + Hydra
│
└── Altri file
```

---

# 🧠 Obiettivo del Progetto

L'obiettivo è sviluppare un sistema di classificazione in grado di identificare fault DC arc in segnali provenienti da impianti fotovoltaici, ottimizzando contemporaneamente:

- accuratezza del modello
- robustezza contro il data leakage
- compatibilità con sistemi embedded
- inferenza real-time su hardware edge

---

# 📚 Dataset

Il progetto utilizza il dataset pubblico:

**Photovoltaic (PV) DC Arc-Fault Library**

https://ieee-dataport.org/open-access/photovoltaic-pv-dc-arc-library

Dal dataset originale vengono estratti:

- corrente
- tensione
- potenza

I segnali vengono successivamente trasformati in finestre temporali utilizzabili dai modelli di classificazione.

---

# 🔄 Pipeline del Progetto

![Pipeline](pipeline.png)

La pipeline completa è suddivisa in cinque fasi principali.

---

# 1. 📂 Data Management

Il dataset grezzo viene:

- validato
- organizzato
- convertito in una struttura coerente per il training

Questa fase comprende:

- caricamento dei segnali originali
- verifica integrità dei dati
- preparazione delle serie temporali

---

# 2. ⚙️ Preprocessing & Physical Split

I segnali vengono preprocessati tramite:

- segmentazione a finestra scorrevole
- normalizzazione
- costruzione dei sample supervisionati

Script principale:

```bash
build_dataset_new.py
```

Successivamente viene eseguito uno split fisico tramite:

```bash
split_dataset.py
```

## Dataset generati

- Train Set → 80%
- Test Set → 20%

Lo split viene effettuato prima del training per evitare:

- overlap tra finestre
- contaminazione tra train e test
- data leakage temporale

I dataset finali vengono salvati in:

- formato `.npz`
- formato `.csv`

---

# 3. 🧠 Training & Feature Extraction

La cartella:

```text
script/training/
```

contiene differenti approcci di classificazione:

| Modello | Descrizione |
|---|---|
| InceptionTime | Deep Learning per Time Series Classification |
| MCNN | Multi-scale Convolutional Neural Network |
| MultiRocket + Hydra | Approccio feature-based ad alte prestazioni |

Durante il training vengono eseguiti:

- controllo anti data leakage
- verifica distribuzione classi
- normalizzazione globale
- feature extraction
- training e validazione

---

# 4. 📦 Export ONNX & Quantizzazione

I modelli addestrati vengono esportati in:

```text
ONNX (FP32)
```

Script principali:

```bash
export_mcnn.py
```

Un sottoinsieme del train set viene utilizzato come:

- dataset di calibrazione
- supporto alla quantizzazione INT8

---

# 5. ⚡ Deployment su ST Edge AI

I modelli vengono verificati tramite:

- ST Edge AI Core
- ST Edge AI Developer Cloud

Obiettivi:

- riduzione memoria
- inferenza embedded
- compatibilità hardware STM32

Vengono analizzate:

| Configurazione | Obiettivo |
|---|---|
| FP32 | Accuratezza massima |
| INT8 | Ottimizzazione Edge AI |

Metriche monitorate:

- RAM usage
- Flash usage
- tempo di inferenza
- accuratezza post-quantizzazione

---

# 6. 📊 Valutazione Finale

Il test set rimane completamente indipendente dall'intero processo di training e calibrazione.

Vengono confrontati:

- modello FP32
- modello INT8

L'obiettivo finale è misurare:

- accuracy
- precision
- recall
- F1-score
- impatto della quantizzazione

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

---

# 🚀 Esempio di Training

```bash
python train_multirocket_fixed.py \
    dataset_train.npz \
    dataset_test.npz \
    --out ./results
```

---

# 🎯 Obiettivi Tecnici

- rilevazione real-time di archi DC
- riduzione del data leakage
- deployment embedded
- ottimizzazione memoria/inferenza
- pipeline riproducibile e scalabile

---

# 📄 Licenza

Il dataset originale appartiene ai rispettivi autori del progetto IEEE DataPort.
