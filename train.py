"""
train.py
--------
Trains and saves two models:

  1. TF-IDF Vectorizer  – converts cleaned log text to numerical features
  2. Isolation Forest   – unsupervised anomaly detection

Optionally trains a supervised severity classifier (Random Forest)
when labelled data is available.

Run:
    python train.py
    python train.py --logs data/system_logs.csv --out models/
"""

import argparse
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ─── Default paths ────────────────────────────────────────────────────────────
DEFAULT_LOGS   = os.path.join("data", "HDFS_2k.log")
DEFAULT_OUTDIR = "models"


# ─── Helper: save a Python object with pickle ─────────────────────────────────
def save_model(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  💾  Saved → {path}")


# ─── Helper: load a pickled object ───────────────────────────────────────────
def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


# ─── 1. TF-IDF Vectorizer ────────────────────────────────────────────────────
def train_tfidf(texts: pd.Series, max_features: int = 500) -> TfidfVectorizer:
    """
    Fit a TF-IDF vectorizer on the cleaned log messages.

    Parameters
    ----------
    texts        : Series of cleaned log strings
    max_features : vocabulary size cap

    Returns
    -------
    Fitted TfidfVectorizer
    """
    print("\n[1/3] Training TF-IDF Vectorizer …")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),          # unigrams + bigrams
        min_df=2,                    # ignore very rare terms
        sublinear_tf=True,           # apply log(1+tf) scaling
        strip_accents="unicode",
        analyzer="word",
    )
    vectorizer.fit(texts)
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")
    return vectorizer


# ─── 2. Isolation Forest (anomaly detection) ─────────────────────────────────
def train_isolation_forest(
    X,
    contamination: float = 0.15,
    n_estimators: int = 150,
    random_state: int = 42,
) -> IsolationForest:
    """
    Train an Isolation Forest on TF-IDF feature matrix.

    Isolation Forest isolates anomalies by building random trees.
    Points that are easy to isolate (short average path length) are anomalies.

    contamination : expected fraction of anomalies in the data
    """
    print("\n[2/3] Training Isolation Forest …")
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X)

    preds    = model.predict(X)
    n_anom   = (preds == -1).sum()
    n_normal = (preds == 1).sum()
    print(f"  Training anomalies detected : {n_anom}  ({n_anom/len(preds)*100:.1f}%)")
    print(f"  Training normal logs        : {n_normal}")
    return model


# ─── 3. Severity Classifier ──────────────────────────────────────────────────
def train_severity_classifier(
    X,
    y: pd.Series,
    label_encoder: LabelEncoder,
    test_size: float = 0.20,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Train a Random Forest to classify log severity (INFO/WARNING/ERROR/CRITICAL).

    Parameters
    ----------
    X            : TF-IDF feature matrix
    y            : Series of severity labels
    label_encoder: fitted LabelEncoder for the severity labels
    """
    print("\n[3/3] Training Severity Classifier (Random Forest) …")

    y_enc = label_encoder.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        class_weight="balanced",   # handles class imbalance
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Evaluation on test split
    y_pred = clf.predict(X_test)
    print("\n  Classification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )
    return clf


# ─── Main training pipeline ──────────────────────────────────────────────────
def train_all(logs_path: str = DEFAULT_LOGS, out_dir: str = DEFAULT_OUTDIR) -> dict:
    """
    Full training pipeline:
      1. Load & preprocess logs
      2. Train TF-IDF
      3. Train Isolation Forest
      4. Train severity classifier
      5. Save all artefacts to out_dir

    Returns a dict of all trained objects.
    """
    # ── Import here to avoid circular imports when used as a library ──────────
    from preprocess import load_and_preprocess

    print("=" * 60)
    print("  AI-Powered Log Analysis System — Model Training")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\nLoading logs from: {logs_path}")
    df = load_and_preprocess(logs_path)
    print(f"  Rows loaded : {len(df)}")
    print(f"  Severity distribution:\n{df['severity'].value_counts().to_string()}")

    texts    = df["clean_message"]
    severity = df["severity"]

    # ── TF-IDF ────────────────────────────────────────────────────────────────
    vectorizer = train_tfidf(texts, max_features=500)
    X = vectorizer.transform(texts)

    # ── Label encoder (needed before classifier) ──────────────────────────────
    le = LabelEncoder()
    le.fit(["INFO", "WARNING", "ERROR", "CRITICAL"])   # fixed order

    # ── Isolation Forest ──────────────────────────────────────────────────────
    iso_forest = train_isolation_forest(X, contamination=0.15)

    # ── Severity Classifier ───────────────────────────────────────────────────
    classifier = train_severity_classifier(X, severity, le)

    # ── Save artefacts ────────────────────────────────────────────────────────
    print("\nSaving model artefacts …")
    save_model(vectorizer,  os.path.join(out_dir, "tfidf_vectorizer.pkl"))
    save_model(iso_forest,  os.path.join(out_dir, "isolation_forest.pkl"))
    save_model(classifier,  os.path.join(out_dir, "severity_classifier.pkl"))
    save_model(le,          os.path.join(out_dir, "label_encoder.pkl"))

    print("\n✅  Training complete!\n")
    return {
        "vectorizer":  vectorizer,
        "iso_forest":  iso_forest,
        "classifier":  classifier,
        "le":          le,
    }


# ─── CLI entry-point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train log analysis models")
    parser.add_argument("--logs", default=DEFAULT_LOGS,   help="Path to log CSV")
    parser.add_argument("--out",  default=DEFAULT_OUTDIR, help="Model output directory")
    args = parser.parse_args()

    train_all(logs_path=args.logs, out_dir=args.out)
