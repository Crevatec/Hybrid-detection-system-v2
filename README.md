# Hybrid Behavioral Detection System (HBDS)

## Real-Time Mitigation of Business Logic & Credential Stuffing Attacks in APIs

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

The **Hybrid Behavioural Detection System (HBDS)** is an intelligent API security middleware that intercepts incoming HTTP traffic, analyses the behavioural signatures of requests in real time, and executes automated mitigation before malicious requests reach the backend server.

The system combines four machine learning models — **Random Forest**, **Isolation Forest**, **ANN**, and **LSTM** — into individual and hybrid ensemble configurations to detect three classes of traffic:

- **Normal** (Class 0) — Legitimate authorised user interactions
- **Credential Stuffing** (Class 1) — High-velocity automated login attempts using compromised credentials
- **Business Logic Attacks** (Class 2) — Sequential workflow exploitation and non-volumetric API abuse

---

## System Architecture

The HBDS operates as a modular pipeline across three phases:

### Monitoring Phase (Data Acquisition)
Continuously ingests real-time API traffic and log data. Captures HTTP metadata and payload characteristics, aggregates server-side events, and tracks session-based interactions to establish a normal activity baseline.

### Detection Phase (Hybrid Analysis Engine)
The core inference layer. Applies the hybrid ML ensemble to compute a probabilistic threat score (0–1) for each request. Combines structured anomaly detection with deep sequential analysis to cover both volumetric and logic-based attacks.

### Mitigation Phase (Graduated Response)
Executes tiered automated responses based on the computed threat score:

| Score Range | Classification | Action | Description |
|---|---|---|---|
| 0.00 – 0.29 | Safe | **ALLOW** | Request forwarded to backend |
| 0.30 – 0.69 | Suspicious | **THROTTLE** | Rate-limited or CAPTCHA challenge triggered |
| 0.70 – 1.00 | Malicious | **BLOCK** | Session terminated with HTTP 403 Forbidden |

A **Feedback Loop** feeds mitigation outcomes back into the monitoring stage, allowing models to adapt to evolving attack patterns and reduce future false positives.

---

## Dataset

- **Name:** `api_security_dataset_2026`
- **Size:** 100,000 API behavioural records
- **Fields:** `timestamp`, `user_id`, `ip_address`, `endpoint`, `method`, `status_code`, `latency_ms`, `label`

### Class Distribution

| Traffic Class | Records | Percentage | Description |
|---|---|---|---|
| Normal | 69,913 | 69.9% | Legitimate user API interactions |
| Credential Stuffing | 14,892 | 14.9% | Automated credential abuse attempts |
| Business Logic Attack | 15,195 | 15.2% | Sequential workflow exploitation |
| **Total** | **100,000** | **100%** | |

### Train/Test Split

| Split | Records | Percentage |
|---|---|---|
| Training Set | 80,000 | 80% |
| Test Set | 20,000 | 20% |

Stratified 80/20 split applied to preserve class proportions in both subsets.

### Label Mapping

| Raw Label | Mapped Class |
|---|---|
| Normal | Normal (0) |
| Web Attack – Brute Force, SSH-Patator | Credential Stuffing (1) |
| Web Attack – XSS, SQL Injection, Bot, DDoS, DoS | Business Logic Attack (2) |
| All error labels (DB failure, latency, etc.) | Normal (0) |

---

## Feature Engineering

Raw API request fields are transformed into a **20-dimensional numerical feature vector**:

| Feature | Source Field | Type | Description |
|---|---|---|---|
| `hour_sin`, `hour_cos` | timestamp | Cyclical | Hour of day (circular encoding) |
| `dow_sin`, `dow_cos` | timestamp | Cyclical | Day of week (circular encoding) |
| `day_of_month`, `month` | timestamp | Numeric | Calendar components |
| `endpoint_hash` | endpoint | Numeric | Hash bucket of API route (0–9999) |
| `method_*` | method | Binary (OHE) | One-hot HTTP method flags (GET, POST, DELETE, PATCH, PUT) |
| `status_code` | status_code | Numeric | Raw HTTP status code |
| `status_family` | status_code | Numeric | Status class (2xx, 4xx, 5xx) |
| `is_error`, `is_server_error` | status_code | Binary | Error type indicators |
| `is_auth_error` | status_code | Binary | 401/403 authentication failure flag |
| `latency_log` | latency_ms | Numeric | Log-transformed response time (log1p) |
| `is_high_latency` | latency_ms | Binary | Latency > 5000ms indicator |
| `user_id` | user_id | Numeric | Extracted user ID integer |
| `ip_address` | ip_address | Numeric | 32-bit IP integer |

All features standardised using **StandardScaler** (zero mean, unit variance).

---

## Models Implemented

### Individual Models (Phase 2)

| Model | Key Parameters | Strategy | Class Handling |
|---|---|---|---|
| Random Forest | 200 trees, max_depth=None | Supervised, parallel (n_jobs=-1) | Balanced class weights |
| Isolation Forest | 200 estimators, contamination=0.30 | Unsupervised, no labels used | Not applicable |
| ANN | Layers: 256→128→64→32, Dropout: 0.3/0.2 | Adam lr=0.001, EarlyStopping patience=5 | Balanced class weights |
| LSTM | LSTM(128)+LSTM(64), seq_len=5, 4 features/step | Adam lr=0.001, EarlyStopping patience=5 | Balanced class weights |

**ANN Architecture:** Batch Normalisation → FC(256, ReLU) → Dropout(0.3) → FC(128, ReLU) → Dropout(0.2) → FC(64, ReLU) → FC(32, ReLU) → Softmax(3)

**LSTM Architecture:** Input reshaped to (5 time-steps × 4 features) → LSTM(128, return_sequences=True) → Dropout(0.3) → LSTM(64) → Dropout(0.2) → Dense(32) → Softmax(3)

### Hybrid Ensemble Models (Phase 3)

| Hybrid Model | Strategy | Component Weights |
|---|---|---|
| RF + ANN | Stacking: RF probabilities concatenated with original features as ANN input | Learned by meta-ANN |
| IF + ANN | Augmentation: Normalised IF anomaly score appended as additional ANN input feature | Learned by ANN |
| RF + LSTM | Parallel weighted voting of independent probability outputs | RF=45%, LSTM=55% |
| ANN + LSTM + RF (Master) | Triple weighted soft voting across all three supervised models | RF=30%, ANN=35%, LSTM=35% |

---

## Results

### Individual Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Isolation Forest | 0.6061 | 0.5929 | 0.6061 | 0.5970 |
| ANN | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| LSTM | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### Hybrid Ensemble Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| RF + ANN | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| IF + ANN | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| RF + LSTM | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **ANN + LSTM + RF (Master)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

> All supervised models achieved perfect classification on `api_security_dataset_2026`. The Master Hybrid confusion matrix recorded zero misclassifications across all 20,000 test samples (13,983 Normal, 2,978 Credential Stuffing, 3,039 Business Logic Attack). The Isolation Forest F1 of 0.5970 reflects expected unsupervised behaviour — it functions as a complementary zero-day detection layer, not a primary classifier.

---

## Project Structure

```
hybrid_detection_system/
│
├── README.md                          # This file
├── requirements.txt                   # All Python dependencies
│
├── data/
│   └── (place api_security_dataset_2026.csv here)
│
├── utils/
│   ├── __init__.py
│   ├── data_loader.py                 # Data loading & preprocessing
│   ├── feature_engineering.py        # Feature extraction & encoding
│   └── metrics.py                    # Evaluation utilities
│
├── models/
│   ├── __init__.py
│   ├── random_forest_model.py        # Phase 2: RF standalone
│   ├── isolation_forest_model.py     # Phase 2: IF standalone
│   ├── ann_model.py                  # Phase 2: ANN standalone
│   ├── lstm_model.py                 # Phase 2: LSTM standalone
│   ├── hybrid_rf_ann.py              # Phase 3: RF + ANN stacking
│   ├── hybrid_if_ann.py              # Phase 3: IF + ANN augmentation
│   ├── hybrid_rf_lstm.py             # Phase 3: RF + LSTM parallel voting
│   └── hybrid_master.py             # Phase 3: ANN + LSTM + RF master hybrid
│
├── outputs/
│   ├── metrics/                      # Saved metrics JSON files
│   ├── plots/                        # Saved comparison charts
│   └── saved_models/                 # Persisted trained model artifacts
│
├── dashboard/
│   ├── __init__.py
│   ├── realtime_simulator.py         # Real-time API event stream simulation
│   └── dashboard_app.py              # Streamlit monitoring dashboard
│
├── train_individual.py               # Run Phase 2: train all individual models
├── train_hybrid.py                   # Run Phase 3: train all hybrid models
├── evaluate_all.py                   # Run Phase 4: metrics + charts
└── run_dashboard.py                  # Run Phase 5: launch dashboard
```

---

## Installation & Usage

### 1. Create Virtual Environment (Python 3.11 required)

```bash
py -3.11 -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Dataset

Place `api_security_dataset_2026.csv` into the `data/` folder.

### 4. Run the System (in order)

```bash
# Phase 2: Train all individual models (~20 mins)
python train_individual.py

# Phase 3: Train all hybrid/ensemble models (~35 mins)
python train_hybrid.py

# Phase 4: Generate metrics and comparison charts
python evaluate_all.py

# Phase 5: Launch real-time monitoring dashboard
streamlit run run_dashboard.py
```

---

## Dashboard Features

The real-time monitoring dashboard provides full operational visibility into the classification decisions of the HBDS:

- **KPI Summary Cards** — Total events processed, attack rate %, live accuracy, cumulative Credential Stuffing and Business Logic detection counts
- **Live Event Feed** — Scrollable table of the most recent 50 classified API requests with timestamp, endpoint, method, status, latency, class, and confidence score
- **Rolling Threat Distribution** — Pie chart showing proportional breakdown across Normal, Credential Stuffing, and Business Logic classifications (last 200 events)
- **Attack Timeline** — Stacked bar chart visualising temporal distribution of threat classifications
- **Attack Rate Gauge** — Real-time gauge displaying rolling attack percentage with colour-coded risk thresholds
- **Master Hybrid Panel** — Dedicated ANN+LSTM+RF panel showing component weights, most recent prediction confidence breakdown, and per-class probability bar chart
- **Model Metrics Comparison** — Interactive grouped bar chart comparing Accuracy, Precision, Recall, and F1-Score across all 8 model configurations

---

## Tech Stack

| Library | Version |
|---|---|
| Python | 3.11.9 |
| TensorFlow / Keras | 2.15.0 |
| Scikit-learn | 1.4.0 |
| Pandas | 2.1.4 |
| NumPy | 1.26.4 |
| Streamlit | 1.31.0 |
| Plotly | 5.18 |
| Matplotlib | 3.8 |

**Development Environment:** Windows 11 (64-bit), Intel Core i7, 16GB RAM, 512GB SSD

---

## Key Contributions

- **Hybrid Detection Architecture** — Novel four-model framework (IF + RF + ANN + LSTM) combining unsupervised anomaly detection with supervised multi-class classification and temporal sequence analysis at the API layer
- **Behavioural Feature Engineering** — Comprehensive pipeline converting raw HTTP metadata into rich 20-dimensional behavioural vectors including cyclical temporal encoding, hash-based endpoint representation, and log-transformed latency features
- **Behavioural-to-Sequential Transformation** — Methodology for reshaping stateless HTTP request vectors into temporal sequences for LSTM-based deep learning without requiring session-level tracking infrastructure
- **Graduated Mitigation Strategy** — Tiered Allow/Throttle/Block response framework minimising false positive impact on legitimate users while disrupting automated attack tools
- **Real-Time Operational Dashboard** — Live Streamlit dashboard integrating event classification, threat distribution, attack rate tracking, and multi-model performance comparison in a single interface

---

## Future Work

- Adversarial robustness testing against evasion-aware attack sequences
- Edge computing deployment (Cloudflare Workers, AWS Lambda@Edge)
- Zero-Trust Architecture integration for dynamic permission adjustment
- Federated learning for privacy-preserving collaborative model training
- Extension to OWASP API Security Top 10 threats (BOLA, mass assignment, function-level authorisation bypass)
- Transformer-based sequence models as LSTM alternatives for endpoint pattern analysis

---

## License

MIT License — free to use, modify, and distribute.