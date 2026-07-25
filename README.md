# BESS_Flexibility_ML

# Machine Learning-Driven BI for BESS Market Flexibility & Price Dynamics

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Market Focus](https://img.shields.io/badge/Market-Netherlands_%26_Iberia-green.svg)](#)

An interpretable Business Intelligence (BI) and Machine Learning framework designed to predict day-ahead electricity prices, manage extreme negative pricing regimes (e.g., -500.0 EUR/MWh), and optimize Battery Energy Storage System (BESS) flexibility bidding strategies under the European Electricity Market Design (EMD) framework.

---

## 🌟 Key Highlights & Engineering Features

- **High Predictive Fidelity**: Achieved **$R^2: 0.9655$** and **$\text{MAE}: 8.38\text{ EUR/MWh}$** across 70,104 hourly market records (2018–2025), effectively capturing extreme tail-risk volatility.
- **Explainable AI (XAI)**: Integrated `SHAP (TreeExplainer)` to decompose black-box Random Forest outputs into transparent, actionable managerial triggers, bridging the AI trust gap for institutional energy investors.
- **Cross-Market Flexibility Analytics**: Leveraged non-synchronized price dynamics between the North Sea (Netherlands) and Iberian (MIBEL) markets ($r = 0.702$) to provide empirical foundations for Flexibility-as-a-Service (FaaS) operations.

---

## 📊 Performance Metrics

| Metric | Random Forest Model Value | Operational Context |
| :--- | :--- | :--- |
| **$R^2$ Score** | **`0.9655`** | Captures >96% of multi-regional price variance |
| **MAE** | **`8.38 EUR/MWh`** | ~5.5% relative error against >150 EUR/MWh arbitrage spreads |
| **Price Floor Captured**| **`-500.00 EUR/MWh`** | Successfully models extreme negative price extremities |
| **Data Granularity** | `70,104 records` | Hourly synchronized UTC time-series (NL, ES, PT) |

---

## 🛠 Repository Structure

```text
├── data/
│   ├── sample_market_data.csv    # Sample preprocessed hourly market data
├── notebooks/
│   ├── 01_data_alignment.ipynb    # UTC synchronization & anomaly filtering
│   ├── 02_rf_model_training.ipynb  # Random Forest regression & hyperparameter tuning
│   └── 03_shap_interpretation.ipynb# SHAP feature attribution & strategic triggers
├── src/
│   ├── preprocessing.py           # Data governance funnel & pipeline
│   └── modeling.py                # ML pipeline configurations
├── requirements.txt               # Dependencies
└── README.md
