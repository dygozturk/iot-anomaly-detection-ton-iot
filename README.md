# SSB – TON_IoT Anomaly Detection (Fridge) 🛰️📊

This repository contains multiple **machine learning / deep learning** approaches for **anomaly detection** on the **TON_IoT – IoT_Fridge** dataset.

> Notes  
> - These scripts were originally exported from Jupyter and may include absolute Windows paths.  
> - The dataset file is **not** included in the repo. Place it locally under `data/` as shown below.

---

## Project Goals
- Prepare and preprocess IoT tabular data
- Train/evaluate multiple models for anomaly detection / classification
- Compare classical ML vs deep learning vs graph-based approaches

---

## Models Included
### Classical / Unsupervised
- **Isolation Forest**
- **K-Means** (with evaluation against labels)

### Deep Learning (TensorFlow / Keras)
- **DNN (MLP)**
- **1D CNN**
- **LSTM**
- **Autoencoder** (reconstruction error based anomaly detection)

### Graph-based (PyTorch Geometric)
- **GCN**
- **GNN (graph construction from KNN similarity)**

---

## Dataset
Expected file:
```
data/IoT_Fridge.csv
```

The scripts assume columns like:
- `date`, `time`, `type` (dropped)
- `temp_condition` (one-hot encoded)
- `label` (target)

---

## Repository Structure (Recommended)
```
ssb-ton-iot-anomaly-detection/
├─ data/
│  └─ IoT_Fridge.csv              
├─ models/
│  ├─ SSB_DNN_Modeli.py
│  ├─ SSB_CNN_Modeli.py
│  ├─ SSB_LSTM_Modeli.py
│  ├─ SSB_Autoencoder_Modeli.py
│  ├─ SSB_KMeans_Modeli.py
│  ├─ SSB_IsolationForest_Modeli.py
│  ├─ SSB_GCN_Modeli.py
│  └─ SSB_GNN_Modeli.py
├─ requirements.txt
└─ README.md
```

---

## Quick Start

### 1) Create environment
**Option A – Conda**
```bash
conda create -n ssb-ton-iot python=3.10 -y
conda activate ssb-ton-iot
pip install -r requirements.txt
```

**Option B – venv**
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Put the dataset
Copy your CSV to:
```text
data/IoT_Fridge.csv
```

### 3) Run a model
Example:
```bash
python models/SSB_DNN_Modeli.py
```

## Results & Reporting
Most scripts print:
- Confusion Matrix
- Classification Report (precision/recall/F1)

If you want, we can also add:
- ROC-AUC (for classification models)
- Unified evaluation table across models
- Saved plots to `outputs/`

---

## License
For academic / portfolio use. If you include TON_IoT references, keep dataset/source attribution in your report/README.
