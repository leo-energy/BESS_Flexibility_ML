# BESS_Flexibility_ML
### Machine Learning-Driven BI for BESS Market Flexibility & Price Dynamics

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Market Focus](https://img.shields.io/badge/Market-Netherlands_%26_Iberia-green.svg)](#)

---

## 📖 Abstract
This research develops an interpretable Business Intelligence (BI) framework to optimize Battery Energy Storage System (BESS) arbitrage strategies amidst the increasing frequency of negative price regimes in European markets. Using 70,104 hourly records from the Netherlands and Iberian markets (2018–2025), a Random Forest (RF) regressor achieves **R² = 0.9655** and **MAE = 8.38 EUR/MWh**, effectively capturing extreme volatility including a **-500.0 EUR/MWh** price floor. SHAP (SHapley Additive exPlanations) transforms black-box outputs into transparent strategic triggers, identifying price inertia and temporal seasonality as primary drivers. 

---

## 🌟 Key Highlights
* **High Predictive Fidelity**: R² = 0.9655, MAE = 8.38 EUR/MWh across 70,104 hourly records.
* **Explainable AI (XAI)**: SHAP decomposes Random Forest outputs into actionable managerial triggers.
* **Negative Price Capture**: Detected -500.0 EUR/MWh extreme price floor.
* **Cross-Market Analytics**: Leveraged NL-ES correlation (r = 0.702) for FaaS arbitrage insights.

---

## 📊 Performance Metrics

| Metric | Random Forest | Operational Context |
| :--- | :--- | :--- |
| **R² Score** | `0.9655` | Explains >96% of multi-regional price variance |
| **MAE** | `8.38 EUR/MWh` | ~5.5% relative error vs. >150 EUR/MWh arbitrage spreads |
| **Price Floor** | `-500.00 EUR/MWh`| Successfully models extreme negative price events |
| **Dataset** | `70,104` records | Synchronized UTC time-series (NL, ES, PT) |

---

## 📂 Repository Structure
```text
BESS_Flexibility_ML/
├── data/
│ └── sample_market_data.csv       # Sample preprocessed hourly data
├── notebooks/
│ ├── 01_data_alignment.ipynb      # UTC synchronization & anomaly filtering
│ ├── 02_rf_model_training.ipynb   # Random Forest regression & tuning
│ └── 03_shap_interpretation.ipynb # SHAP feature attribution & strategic triggers
├── src/
│ ├── preprocessing.py             # Data governance funnel (ETL)
│ └── modeling.py                  # ML pipeline configurations
├── requirements.txt               # Python dependencies
└── README.md
