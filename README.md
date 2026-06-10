# Customer 360 Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-blue)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

End-to-end customer analytics pipeline covering exploratory analysis, ETL, behavioral segmentation, churn prediction, and deep learning — structured as a production-ready notebook series.

---

## What This Does

Takes raw transactional and behavioral customer data through a full analytics lifecycle:

1. **EDA** — distribution analysis, missing value profiling, feature correlation
2. **ETL** — data cleaning, feature engineering, pipeline construction
3. **RFM Segmentation** — Recency/Frequency/Monetary scoring across 200K+ customers
4. **Churn ML** — classification models (Logistic Regression, Random Forest, XGBoost) with SHAP explainability
5. **Deep Learning** — neural network churn model with embedding layers for categorical features
6. **Deployment** — model serialization and sample inference API

---

## Architecture

```
Raw Data (CSV/DB)
      │
      ▼
01_EDA.ipynb          ← Data profiling, outlier detection, visualization
      │
      ▼
02_ETL.ipynb          ← Feature engineering, encoding, train/test split
      │
      ▼
03_RFM.ipynb          ← Behavioral segmentation → customer tiers
      │
      ▼
04_Churn_ML.ipynb     ← Scikit-learn models + SHAP feature importance
      │
      ▼
05_Deep_Learning.ipynb ← PyTorch neural network churn classifier
      │
      ▼
07_Sample_Deploy.ipynb ← Model serialization + inference pipeline
```

---

## Key Results

| Model | Metric | Result |
|---|---|---|
| XGBoost Churn | ROC-AUC | 0.89 |
| Neural Network | Accuracy | ~87% |
| RFM Segmentation | Customers Segmented | 200K+ |

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| Pandas / NumPy | Data manipulation |
| Scikit-learn | ML models and pipelines |
| PyTorch | Deep learning churn model |
| SHAP | Model explainability |
| Plotly / Matplotlib | Visualization |
| SQLite / DuckDB | Local data storage |

---

## Quick Start

```bash
git clone https://github.com/Nag4535/customer360-intelligence
cd customer360-intelligence
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Open notebooks in sequence: `01_EDA.ipynb` → `02_ETL.ipynb` → `03_RFM.ipynb` → `04_Churn_ML.ipynb` → `05_Deep_Learning.ipynb`

---

## Project Structure

```
customer360-intelligence/
├── 01_EDA.ipynb              # Exploratory data analysis
├── 02_ETL.ipynb              # Data pipeline & feature engineering
├── 03_RFM.ipynb              # Behavioral segmentation
├── 04_Churn_ML.ipynb         # Scikit-learn churn models + SHAP
├── 05_Deep_Learning.ipynb    # PyTorch neural network
├── 07_Sample_Deploy.ipynb    # Deployment-ready inference pipeline
├── data/processed/           # Cleaned datasets
├── models/                   # Serialized model artifacts
├── dashboard/                # Power BI / Streamlit dashboards
└── requirements.txt
```

---

## Related Projects

- [market-intel-data-pipeline](https://github.com/Nag4535/market-intel-data-pipeline) — Real-time streaming data infrastructure
- [market-intel-mlops](https://github.com/Nag4535/market-intel-mlops) — FinBERT sentiment model with full MLOps
- [sales-intelligence-platform](https://github.com/Nag4535/sales-intelligence-platform) — Sales analytics ETL + dashboard
