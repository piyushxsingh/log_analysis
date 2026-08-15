"""
predict.py
----------
Loads trained models and runs inference on new log DataFrames.

Public functions
  load_models()          – load all artefacts from disk
  predict_anomalies()    – Isolation Forest → anomaly column
  predict_severity()     – Random Forest classifier → predicted_severity column
  predict_risk_level()   – rule-based risk score
  generate_summary()     – human-readable text summary
  get_recommendations()  – actionable recommendations per severity
  run_full_prediction()  – convenience: all steps in one call
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─── Default model paths ──────────────────────────────────────────────────────
MODEL_DIR = "models"

_PATHS = {
    "vectorizer":  os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"),
    "iso_forest":  os.path.join(MODEL_DIR, "isolation_forest.pkl"),
    "classifier":  os.path.join(MODEL_DIR, "severity_classifier.pkl"),
    "le":          os.path.join(MODEL_DIR, "label_encoder.pkl"),
}


# ─── Model loading ────────────────────────────────────────────────────────────

def _load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_models(model_dir: str = MODEL_DIR) -> dict:
    """
    Load all model artefacts from disk.

    Returns
    -------
    dict with keys: vectorizer, iso_forest, classifier, le
    """
    paths = {k: os.path.join(model_dir, os.path.basename(v)) for k, v in _PATHS.items()}
    models = {}
    for key, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                "Run  python train.py  first to generate model artefacts."
            )
        models[key] = _load_pkl(path)
    return models


# ─── Anomaly detection ────────────────────────────────────────────────────────

def predict_anomalies(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """
    Add 'anomaly' column to df using Isolation Forest.

    Values:
       1  = normal log
      -1  = anomalous log (suspicious / unusual pattern)
    Also adds 'anomaly_score' (raw decision function, lower = more anomalous).
    """
    df = df.copy()

    texts = df["clean_message"].fillna("").astype(str)
    X     = models["vectorizer"].transform(texts)

    df["anomaly"]       = models["iso_forest"].predict(X)
    df["anomaly_score"] = models["iso_forest"].decision_function(X)

    # Force ERROR/CRITICAL to anomaly regardless of IF prediction
    df.loc[df["severity"].isin(["ERROR", "CRITICAL"]), "anomaly"] = -1

    return df


# ─── Severity classification ─────────────────────────────────────────────────

def predict_severity(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """
    Add 'predicted_severity' column using the Random Forest classifier.
    Falls back to the existing 'severity' column if the model is not available.
    """
    df = df.copy()

    texts = df["clean_message"].fillna("").astype(str)
    X     = models["vectorizer"].transform(texts)

    y_enc = models["classifier"].predict(X)
    df["predicted_severity"] = models["le"].inverse_transform(y_enc)

    # Confidence probability for the predicted class
    proba = models["classifier"].predict_proba(X)
    df["confidence"] = np.max(proba, axis=1).round(3)

    return df


# ─── Risk level ───────────────────────────────────────────────────────────────

_RISK_RULES = {
    "CRITICAL": ("🔴 CRITICAL", 4),
    "ERROR":    ("🟠 HIGH",     3),
    "WARNING":  ("🟡 MEDIUM",   2),
    "INFO":     ("🟢 LOW",      1),
}

def predict_risk_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive a human-readable risk level and numeric risk score from severity.
    Anomalous INFO/WARNING logs are bumped up one risk level.
    """
    df = df.copy()

    sev_col = "predicted_severity" if "predicted_severity" in df.columns else "severity"

    def _risk(row):
        label, score = _RISK_RULES.get(row[sev_col], ("🟢 LOW", 1))
        # Anomaly bump: if marked -1 and not already CRITICAL, raise score
        if row.get("anomaly", 1) == -1 and score < 4:
            score = min(score + 1, 4)
            labels_inv = {v[1]: k for k, v in _RISK_RULES.items()}
            bumped = labels_inv.get(score, label)
            label, _ = _RISK_RULES.get(bumped, (label, score))
        return pd.Series({"risk_label": label, "risk_score": score})

    risk_df = df.apply(_risk, axis=1)
    df["risk_label"] = risk_df["risk_label"]
    df["risk_score"] = risk_df["risk_score"]
    return df


# ─── Summary generator ────────────────────────────────────────────────────────

def generate_summary(df: pd.DataFrame) -> str:
    """
    Return a concise human-readable analysis summary string.
    """
    total     = len(df)
    n_anomaly = (df["anomaly"] == -1).sum() if "anomaly" in df.columns else 0
    sev_counts = df["severity"].value_counts()

    critical = sev_counts.get("CRITICAL", 0)
    error    = sev_counts.get("ERROR",    0)
    warning  = sev_counts.get("WARNING",  0)
    info     = sev_counts.get("INFO",     0)

    risk_pct  = (n_anomaly / total * 100) if total > 0 else 0
    health    = (
        "🔴 Critical — immediate action required"  if critical > 5 or risk_pct > 30 else
        "🟠 Degraded — attention needed"           if error    > 10 or risk_pct > 15 else
        "🟡 Warning — monitor closely"             if warning  > 20 else
        "🟢 Healthy — system operating normally"
    )

    lines = [
        f"📋 Log Analysis Summary",
        f"",
        f"Total Logs Analysed : {total:,}",
        f"Anomalies detected : {n_anomaly:,} ({risk_pct:.1f}%)",
        f"",
        f"Severity Breakdown :",
        f"  - 🔴 CRITICAL : {critical}",
        f"  - 🟠 ERROR    : {error}",
        f"  - 🟡 WARNING  : {warning}",
        f"  - 🟢 INFO     : {info}",
        f"",
        f"System Health : {health}",
    ]

    if n_anomaly > 0 and "predicted_severity" in df.columns:
        top = (
            df[df["anomaly"] == -1]["predicted_severity"]
            .value_counts()
            .head(2)
            .to_dict()
        )
        top_str = ", ".join(f"{k}: {v}" for k, v in top.items())
        lines.append(f"**Top anomaly types**: {top_str}")

    return "\n".join(lines)


# ─── Recommendations ─────────────────────────────────────────────────────────

_RECOMMENDATIONS = {
    "CRITICAL": [
        "🚨 **Immediate escalation required** — page on-call engineer now.",
        "🔒 Check for unauthorised access or security breach.",
        "💾 Verify data integrity and initiate backup if needed.",
        "📣 Notify stakeholders and open a P1 incident ticket.",
        "🔄 Consider rolling back the last deployment if incident is post-deploy.",
    ],
    "ERROR": [
        "🔍 Review full stack traces for the failing component.",
        "♻️  Check service restart logs — may indicate crash loops.",
        "🗄️  Verify database connectivity and query performance.",
        "📊 Cross-reference with APM/metrics for CPU/memory spikes.",
        "📝 Document the error pattern and update runbook.",
    ],
    "WARNING": [
        "📈 Monitor resource utilisation — may precede an outage.",
        "⏱️  Review rate-limiting and retry configurations.",
        "🔑 Rotate credentials/tokens that are nearing expiry.",
        "🧹 Clean up stale sessions and clear caches if needed.",
        "📅 Schedule maintenance window for non-urgent remediation.",
    ],
    "INFO": [
        "✅ System is operating within normal parameters.",
        "📋 Retain logs for compliance and audit trails.",
        "📊 Use this baseline to set alert thresholds.",
    ],
}


def get_recommendations(df: pd.DataFrame) -> dict:
    """
    Return a dict mapping each severity level (that appears in df)
    to a list of recommended actions.
    """
    sev_col    = "predicted_severity" if "predicted_severity" in df.columns else "severity"
    present    = df[sev_col].unique()
    recs       = {}
    priority   = ["CRITICAL", "ERROR", "WARNING", "INFO"]

    for sev in priority:
        if sev in present:
            recs[sev] = _RECOMMENDATIONS.get(sev, [])
    return recs


# ─── Full pipeline ────────────────────────────────────────────────────────────

def run_full_prediction(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """
    Run the complete prediction pipeline:
      1. Anomaly detection
      2. Severity classification
      3. Risk level assignment

    Returns enriched DataFrame.
    """
    df = predict_anomalies(df, models)
    df = predict_severity(df, models)
    df = predict_risk_level(df)
    return df


# ─── Smoke-test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from preprocess import load_and_preprocess

    print("Loading models …")
    models = load_models()

    print("Loading & preprocessing logs …")
    df = load_and_preprocess("data/system_logs.csv")

    print("Running predictions …")
    df = run_full_prediction(df, models)

    print(df[["severity", "predicted_severity", "anomaly", "risk_label", "confidence"]].head(10))
    print("\n" + generate_summary(df))

    recs = get_recommendations(df)
    for sev, actions in recs.items():
        print(f"\n[{sev}] Recommendations:")
        for a in actions:
            print(f"  {a}")
