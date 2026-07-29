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

```text

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- pip

### Installation
```bash
git clone https://github.com/leo-energy/BESS_Flexibility_ML.git
cd BESS_Flexibility_ML
pip install -r requirements.txt

---

## Data
Raw data is sourced from the ENTSO‑E Transparency Platform. Sample data is included for demonstration.

---

## Run the Pipeline
# 1. Data alignment
jupyter notebook notebooks/01_data_alignment.ipynb

# 2. Model training
jupyter notebook notebooks/02_rf_model_training.ipynb

# 3. SHAP interpretation
jupyter notebook notebooks/03_shap_interpretation.ipynb

---

## 📈 Key Findings
Price Inertia Dominance: Lag1_NL contributes ~95% of predictive weight.

Negative Price Triggers: Nocturnal windows (00:00–05:00) + renewable oversupply.

Cross‑Market Decoupling: 30% asynchronous volatility enables FaaS cross‑border arbitrage.

---

##🔮 Future Work (Proposal Extension)
Incorporate Gaussian Process Regression (GPR) for uncertainty quantification.

Apply Deep Reinforcement Learning (DDPG/PPO) for real‑time bidding.

Integrate battery degradation models (He et al., 2016) for life‑cycle cost optimization.

Deploy SHAP‑based SOPs for trading desk decision support.

---

## 📚 Related Publications
Paper 1: SLR on FaaS Platforms (in progress)

Paper 2: SBM‑DEA Eco‑Efficiency Benchmarking (in progress)

Paper 3: This ML pipeline + extensions (targeting IEEE Trans on Power Systems)

📄 License
This project is licensed under the MIT License – see the LICENSE file for details.

🤝 Contributing
Contributions are welcome! Please open an issue or pull request for any improvements.
