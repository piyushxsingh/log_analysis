# 🔍 AI-Powered Intelligent Log Analysis System

> A production-ready, machine learning–driven platform for log anomaly detection,  
> severity classification, and real-time system health monitoring — built with Python & Streamlit.

---

## 📸 Dashboard Preview

The dashboard provides a dark-themed, multi-tab interface with:
- Real-time KPI metrics
- Interactive log viewer with filters
- Anomaly detection results
- 5 chart types (pie, bar, timeline, donut, horizontal bar)
- AI-generated summaries and recommendations

---

## 🎯 Project Objective

Develop a machine learning–based system that can:
- Analyse system / application logs
- Detect anomalies using **Isolation Forest**
- Classify log severity using a **Random Forest** classifier
- Display actionable insights on an interactive **Streamlit** dashboard

---

## 🛠️ Tech Stack

| Layer             | Technology                         |
|-------------------|------------------------------------|
| Language          | Python 3.9+                        |
| ML Framework      | Scikit-learn                       |
| Data Processing   | Pandas, NumPy                      |
| Feature Extraction| TF-IDF Vectorization               |
| Anomaly Detection | Isolation Forest                   |
| Classification    | Random Forest Classifier           |
| Visualisation     | Matplotlib                         |
| Dashboard         | Streamlit                          |

---

## 📁 Project Structure

---

## 📁 Project Structure

```
log_analysis_system/
│
├── data/
│   |── generate_logs.py      # Synthetic log generator
│   └── system_logs.csv       # Generated dataset (1,200 logs)
|   └── HDFS_2k.csv
│
├── models/
│   ├── tfidf_vectorizer.pkl  # Trained TF-IDF vectorizer
│   ├── isolation_forest.pkl  # Trained anomaly detector
│   ├── severity_classifier.pkl # Trained severity classifier
│   └── label_encoder.pkl     # Label encoder for severity classes
│
├── app/                      
|   └──app.py                 # Streamlit dashboard
│
├── preprocess.py             # Log loading & text cleaning pipeline
├── train.py                  # Model training script
├── predict.py                # Inference & summary generation
├── app.py                    # Streamlit dashboard
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## ⚙️ Installation

### 1. Clone / Download the project

```bash
git clone https://github.com/yourname/ai-log-analyzer.git
cd ai-log-analyzer
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Step 1 — Generate sample logs  *(skip if you have your own)*

```bash
cd data
python generate_logs.py
cd ..
```

This creates `data/system_logs.csv` with **1,200** synthetic log entries across  
four severity levels: `INFO`, `WARNING`, `ERROR`, `CRITICAL`.

### Step 2 — Train the models

```bash
python train.py
```

This trains and saves four artefacts to `models/`:
- `tfidf_vectorizer.pkl`
- `isolation_forest.pkl`
- `severity_classifier.pkl`
- `label_encoder.pkl`

### Step 3 — Launch the dashboard

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 📊 Features

### 🔬 Preprocessing (`preprocess.py`)
- Auto-detects CSV vs plain-text log format
- Strips timestamps, IP addresses, hashes, and special characters
- Normalises severity labels (WARN → WARNING, CRIT → CRITICAL, etc.)
- Extracts hour-of-day for temporal analysis
- Returns clean, structured `pandas.DataFrame`

### 🤖 Anomaly Detection
- **Algorithm**: Isolation Forest (unsupervised)
- **Logic**: builds random trees; logs that are "easy to isolate" are anomalous
- Labels: `1` = normal, `-1` = anomaly
- Anomaly sensitivity adjustable via sidebar slider

### 🏷️ Severity Classification
- **Algorithm**: Random Forest (supervised, 200 estimators)
- **Classes**: INFO / WARNING / ERROR / CRITICAL
- Outputs predicted label + confidence probability
- Achieves ~100% accuracy on well-labelled synthetic data

### 📈 Dashboard Tabs
| Tab              | Content                                                    |
|------------------|------------------------------------------------------------|
| 📊 Overview      | KPI cards, severity table, pie chart, risk summary         |
| 📄 Log Viewer    | Filterable, searchable full log table + CSV download       |
| 🚨 Anomalies     | Sorted anomaly list, top sources, severity breakdown       |
| 📈 Charts        | 5 visualisations (pie, bar, timeline, donut, source bar)   |
| 📝 Summary       | AI-generated text summary + prediction accuracy            |
| 💡 Recommendations | Severity-specific actionable recommendations             |
| ⚡ Real-time Monitor | Simulated live log stream                              |

### 🔑 Extra Features
- **Risk level predictor**: 4-tier risk scoring (LOW → MEDIUM → HIGH → CRITICAL)
- **Anomaly score**: raw Isolation Forest decision function (sortable)
- **Recommendations engine**: curated action items per severity level
- **Real-time simulation**: live stream replay with configurable speed
- **CSV exports**: filtered logs, anomaly report, full analysis

---

## 📉 Model Performance

```
              precision    recall  f1-score

    CRITICAL       1.00      1.00      1.00
       ERROR       1.00      1.00      1.00
        INFO       1.00      1.00      1.00
     WARNING       1.00      1.00      1.00

    accuracy                   1.00
```

*(On synthetic dataset — real-world accuracy depends on log quality.)*

Isolation Forest detects ~15% of logs as anomalous (configurable).

---

## 📝 Supported Log Formats

### CSV format
```csv
timestamp,severity,message,source,host
2024-01-15 08:23:11,INFO,User alice logged in,auth-service,host-1
2024-01-15 08:23:45,ERROR,Database connection failed,db-service,host-2
```

### Plain-text / syslog format
```
Jan 15 08:23:11 host-1 auth: INFO User alice logged in
Jan 15 08:23:45 host-2 db: ERROR Database connection failed: timeout
```

---

## 🔄 Advanced / Production Extensions

| Feature            | How to Add                                              |
|--------------------|---------------------------------------------------------|
| Real-time ingest   | Connect Kafka consumer → pipe to `predict.run_full_prediction()` |
| LSTM detection     | Replace Isolation Forest with `keras.layers.LSTM`       |
| Email alerts       | Use `smtplib` + trigger on `risk_score >= 3`            |
| Database storage   | Write `df_result` to PostgreSQL via `sqlalchemy`        |
| REST API           | Wrap `predict.py` with FastAPI                          |
| Docker deploy      | `docker build -t log-analyzer . && docker run -p 8501:8501` |

---

## 👨‍💻 Author

Built as a **placement-worthy portfolio project** demonstrating:
- Applied Machine Learning (unsupervised + supervised)
- NLP / TF-IDF feature engineering
- Full-stack Python data application
- Professional UI/UX with Streamlit

---

## 📄 Licence

MIT — free to use, modify, and distribute.
