# 📈 Market Trend Prediction Platform

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-4169E1?style=flat&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-blueviolet?style=flat)](https://shap.readthedocs.io/)

An end-to-end, production-grade **Market Trend Prediction & Quantitative Analytics Engine**. The platform automates 1-hour candle data ingestion, computes rolling technical indicators, trains calibrated multi-class machine learning models, serves live predictions with **SHAP TreeExplainer** attribution, and tracks forward real-world accuracy through an automated outcome resolution pipeline.

---

## 🌟 Key Features

- **Ternary Trend Classification (`UP`, `NEUTRAL`, `DOWN`):** Evaluates multi-period price movement over a 5-hour horizon with configurable threshold boundaries ($\pm 0.5\%$).
- **Strict Model Selection & Verification:** Calibrates models with 5-fold cross-validation and moving-block bootstrap confidence intervals ($95\%$ CI), selecting serving models based strictly on out-of-sample MCC (Matthews Correlation Coefficient).
- **Explainable AI (SHAP):** On-demand `/api/explain/{symbol}` endpoint calculating tree-based Shapley values for the predicted class, with directional impact breakdowns.
- **Forward Outcome Tracking:** Automated background job resolves predictions once the 5-bar horizon elapses, recording actual returns, labels, and comparing live accuracy against an "Always NEUTRAL" baseline.
- **Interactive Multi-Asset Dashboard:** Streamlit frontend organized into 3 dedicated workspaces:
  - 🔮 **Live Forecast & Drivers:** Key metrics, timestamp metadata, and dark-themed SHAP contribution charts.
  - 🎯 **Live Outcome Tracker:** Progress tracking toward statistical significance ($30$ predictions), accuracy cards, and forward testing table.
  - 📊 **Model Evidence & Diagnostics:** Honest statistical verdicts, model comparison tables, $3 \times 3$ confusion matrices, and Move vs. No-Move accuracy metrics.
- **Fully Containerized:** Multi-container deployment using Docker Compose (PostgreSQL, FastAPI Backend, and Streamlit Dashboard).

---

## 🏗️ Architecture

```mermaid
flowchart TD
    subgraph Data Layer
        YF[Yahoo Finance API] --> Ingest[Candle Ingestion Service]
        Ingest --> DB[(PostgreSQL Database)]
    end

    subgraph Machine Learning Pipeline
        DB --> FE[Feature Engineering Engine]
        FE --> Calib[Isotonic Calibration CV=5]
        Calib --> Model[Saved Models & Metrics JSON]
    end

    subgraph Service Layer
        DB --> API[FastAPI REST Server]
        Model --> API
        API --> Pred[/api/predict/{symbol}/]
        API --> Exp[/api/explain/{symbol}/]
        API --> Perf[/api/performance/{symbol}/]
        Track[Outcome Tracker Scheduler] --> DB
    end

    subgraph Frontend
        API --> UI[Streamlit Interactive Dashboard]
    end
```

---

## 📁 Project Structure

```text
.
├── app/                        # FastAPI Backend & Core Services
│   ├── main.py                 # FastAPI application factory
│   ├── database.py             # PostgreSQL connection pooling
│   ├── scheduler.py            # Automated jobs (ingestion & outcome resolution)
│   ├── routes/                 # REST API endpoints (predict, explain, performance)
│   └── services/               # Predictor, explainer, ingester, and outcome tracker
├── ml/                         # Machine Learning Pipeline
│   ├── feature_engineering.py  # Canonical indicators & target generation
│   ├── train_offline.py        # 5-fold CV, bootstrap CIs & model training
│   ├── market_data_fetcher.py  # Closed candle fetching & normalization
│   └── models/                 # Calibrated .pkl models, metadata & metrics.json
├── dashboard/                  # Streamlit Visualization App
│   └── app.py                  # Responsive tabbed dashboard
├── scripts/                    # Database migrations & verification utilities
├── Dockerfile                  # Production container definition
├── docker-compose.yml          # Multi-container orchestration
├── requirements.txt            # Pinned dependencies
├── schema.sql                  # PostgreSQL table schemas & indexes
└── .env.example                # Environment configuration template
```

---

## 🚀 Quickstart with Docker (Recommended)

### 1. Clone the repository
```bash
git clone https://github.com/<YOUR_USERNAME>/market-trend-predictions.git
cd market-trend-predictions
```

### 2. Configure environment variables
```bash
cp .env.example .env
```

### 3. Launch the platform
```bash
docker compose up --build -d
```

### 4. Access the services
- **Streamlit Dashboard:** [http://localhost:8501](http://localhost:8501)
- **FastAPI Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Database Port:** `localhost:5432`

---

## 💻 Manual Local Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run offline training (if updating models)
python -m ml.train_offline

# Start FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# In a separate terminal, launch Streamlit
streamlit run dashboard/app.py
```

---

## 📡 REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Health check and database connectivity status |
| `GET` | `/api/predict/{symbol}` | Returns latest prediction, direction, confidence, and class probabilities |
| `GET` | `/api/explain/{symbol}` | Returns top 5 SHAP feature drivers for the predicted class |
| `GET` | `/api/performance/{symbol}` | Returns live resolved prediction accuracy vs Always NEUTRAL baseline |

### Example: Explainability Response (`GET /api/explain/BITCOIN`)
```json
{
  "symbol": "BITCOIN",
  "model_type": "RandomForestClassifier",
  "predicted_class": "NEUTRAL",
  "confidence": 84.13,
  "candle_timestamp": "2026-10-03T17:00:00+00:00",
  "top_drivers": [
    {
      "feature": "vol_10",
      "feature_value": 0.001,
      "shap_value": 0.1795,
      "direction": "pushes_toward"
    },
    {
      "feature": "ret_3",
      "feature_value": 0.0012,
      "shap_value": 0.0228,
      "direction": "pushes_toward"
    }
  ]
}
```

---

## 📊 Backtest & Evidence Summary

Model selection is strictly based on out-of-sample MCC across 5-fold cross-validation with 95% moving-block bootstrap confidence intervals:

- **BITCOIN:** `RandomForest` (MCC: `0.1067`, 95% CI: `[0.0801, 0.1317]`) — 🟢 **Statistically significant edge** over best baseline.
- **ETHEREUM:** `RandomForest` (MCC: `0.0732`, 95% CI: `[0.0454, 0.1001]`) — 🟡 **Inconclusive** (point estimate beats baseline, but 95% CI overlaps).
- **IBM:** `RandomForest` (MCC: `0.0419`, 95% CI: `[0.0094, 0.0743]`) — 🔴 **No proven edge over baseline** (baseline point estimate outperforms ML).

---

## ⚠️ Disclaimer
*This platform and all associated backtest metrics are developed for quantitative research and educational demonstration only. Nothing contained herein constitutes financial advice.*