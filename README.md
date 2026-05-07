# DC-ARC-Fault

![Pipeline](pipeline.png)

## 🚀 Pipeline di Training e Deploy per Edge AI

Questa pipeline descrive il flusso completo di sviluppo di un modello di machine learning ottimizzato per il deployment su hardware embedded tramite **ST Edge AI**.

Il processo si articola in cinque fasi principali:

---

### 1. 📂 Gestione Dati
A partire dal dataset grezzo integrale, viene eseguita:
- l’estrazione dei dati rilevanti  
- l’organizzazione in una cartella dedicata  

---

### 2. ⚙️ Pre-processing & Physical Split
I dati vengono:
- processati tramite **tecnica a finestra scorrevole**
- suddivisi fisicamente in:
  - **Train set (80%)**
  - **Test set (20%)**

👉 Lo split avviene **prima del training**, riducendo il rischio di data leakage.

Entrambi i dataset vengono salvati in:
- formato **NPZ**
- formato **CSV**

---

### 3. 🧠 Training & Calibration
Sul **train set** vengono eseguite le seguenti operazioni:

- controllo esplicito di **data leakage**
  - verifica di overlap tra finestre  
  - verifica distribuzioni  

- training dei modelli  

- esportazione in formato:
  - **ONNX (F32)**  

Inoltre:
- un sottoinsieme del train set viene utilizzato come **dataset di calibrazione** per la quantizzazione

---

### 4. ⚡ ST Edge AI Hardware Check
Il modello ONNX viene analizzato tramite:

- **ST Edge AI Tool**

per verificare la compatibilità con i vincoli hardware:

- **RAM**
- **Flash**

Vengono valutate due configurazioni:
- modello in **F32**
- modello quantizzato in **INT8**

👉 Il dataset di calibrazione viene utilizzato per il processo di quantizzazione.

---

### 5. 📊 Verifica Finale
Il **test set**, rimasto completamente indipendente, viene utilizzato esclusivamente per la valutazione finale.

Viene eseguito:
- test del modello **F32**
- test del modello **INT8**

👉 I risultati vengono confrontati per:
- valutare l’accuratezza
- quantificare l’impatto della quantizzazione sulle prestazioni

---

