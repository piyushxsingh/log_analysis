"""
preprocess.py
-------------
Reads raw log files (CSV or plain-text), cleans the text, and returns a
structured Pandas DataFrame ready for feature extraction and modelling.

Supported input formats
  • CSV  – must contain at least a 'message' column;
            optional: 'timestamp', 'severity', 'source', 'host'
  • TXT  – one log line per row, auto-parses common syslog patterns
"""

import re
import io
import pandas as pd
from datetime import datetime


# ─── Regex patterns ───────────────────────────────────────────────────────────

# Matches timestamps like: 2024-01-15 08:23:11  |  Jan 15 08:23:11  |  [2024-01-15T08:23:11]
_TS_PATTERN = re.compile(
    r"""
    (
        # ISO 8601
        \[?\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\]?

        |

        # Syslog
        \[?[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\]?

        |

        # HDFS: 081109 203615 148
        \d{6}\s+\d{6}\s+\d{1,4}

        |

        # Apache/Nginx
        \[?\d{2}/[A-Z][a-z]{2}/\d{4}:\d{2}:\d{2}:\d{2}\s+[+-]\d{4}\]?

        |

        # Epoch (10 or 13 digits)
        \d{10}(?:\d{3})?
    )
    """,
    re.VERBOSE,
)

# Severity keywords
_SEV_PATTERN = re.compile(
    r"\b(INFO|WARNING|WARN|ERROR|ERR|CRITICAL|CRIT|DEBUG|FATAL)\b",
    re.IGNORECASE,
)

# IPv4 addresses
_IP_PATTERN = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")

# Hex values, UUIDs, long numbers (keep short numbers for context)
_HASH_PATTERN = re.compile(r"\b[0-9a-f]{8,}\b", re.IGNORECASE)

# Common special characters to strip (keep hyphens/underscores inside words)
_SPECIAL_PATTERN = re.compile(r"[^\w\s\-/]")

# Multiple whitespace
_WS_PATTERN = re.compile(r"\s+")


# ─── Severity normalisation map ───────────────────────────────────────────────
_SEV_MAP = {
    "warn":     "WARNING",
    "warning":  "WARNING",
    "err":      "ERROR",
    "error":    "ERROR",
    "crit":     "CRITICAL",
    "critical": "CRITICAL",
    "fatal":    "CRITICAL",
    "debug":    "INFO",
    "info":     "INFO",
}


def _normalise_severity(raw: str) -> str:
    """Convert any severity variant to a canonical label."""
    return _SEV_MAP.get(raw.lower(), raw.upper())


def _clean_message(text: str) -> str:
    """
    Apply the full cleaning pipeline to a single log message string:
      1. Strip timestamps
      2. Strip severity keywords (they're stored in a separate column)
      3. Anonymise IPs  → '<ip>'
      4. Replace hashes/UUIDs → '<hash>'
      5. Remove special characters
      6. Lower-case
      7. Collapse whitespace
    """
    text = _TS_PATTERN.sub("", text)
    text = _SEV_PATTERN.sub("", text)
    text = _IP_PATTERN.sub("<ip>", text)
    text = _HASH_PATTERN.sub("<hash>", text)
    text = _SPECIAL_PATTERN.sub(" ", text)
    text = text.lower()
    text = _WS_PATTERN.sub(" ", text).strip()
    return text


# ─── Plain-text parser ────────────────────────────────────────────────────────

def _parse_text_line(line: str) -> dict:
    """
    Parse a single plain-text log line into a dict with keys:
    timestamp, severity, message.
    Falls back gracefully when fields are missing.
    """
    original = line.strip()
    if not original:
        return None

    # Extract timestamp
    ts_match = _TS_PATTERN.search(original)
    timestamp = ts_match.group(0).strip("[]") if ts_match else ""

    # Extract severity
    sev_match = _SEV_PATTERN.search(original)
    severity = _normalise_severity(sev_match.group(0)) if sev_match else "INFO"

    return {
        "timestamp": timestamp,
        "severity":  severity,
        "message":   original,   # raw; cleaned later
        "source":    "unknown",
        "host":      "unknown",
    }


# ─── Public API ───────────────────────────────────────────────────────────────

def load_logs(file_obj) -> pd.DataFrame:
    """
    Load logs from a file-like object or path string.
    Auto-detects CSV vs plain-text format.

    Parameters
    ----------
    file_obj : str | Path | BytesIO | StringIO
        The log file to read.

    Returns
    -------
    pd.DataFrame with columns: timestamp, severity, message, source, host
    """
    # Read raw content
    if isinstance(file_obj, (str,)):
        with open(file_obj, "r", errors="replace") as f:
            raw = f.read()
    else:
        # Streamlit UploadedFile / BytesIO / StringIO
        try:
            raw = file_obj.read().decode("utf-8", errors="replace")
        except AttributeError:
            raw = file_obj.read()

    # ── CSV detection: first non-empty line contains comma-separated headers ──
    first_line = raw.lstrip().split("\n")[0]
    if "," in first_line and any(
        h in first_line.lower() for h in ["message", "severity", "timestamp"]
    ):
        df = pd.read_csv(io.StringIO(raw))
        df.columns = [c.strip().lower() for c in df.columns]

        # Ensure required columns exist
        if "message" not in df.columns:
            raise ValueError("CSV must contain a 'message' column.")
        for col in ["timestamp", "severity", "source", "host"]:
            if col not in df.columns:
                df[col] = "unknown"

    else:
        # ── Plain-text: parse line by line ────────────────────────────────────
        rows = [_parse_text_line(ln) for ln in raw.splitlines()]
        rows = [r for r in rows if r is not None]
        df   = pd.DataFrame(rows)

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline on a raw DataFrame.

    Steps
    -----
    1.  Drop fully empty rows
    2.  Normalise severity labels
    3.  Fill missing fields
    4.  Create `clean_message` column (cleaned text for ML)
    5.  Extract `hour` from timestamp for timeline features
    6.  Add numeric severity encoding for models that need it

    Returns
    -------
    pd.DataFrame – original columns  +  clean_message, hour, severity_code
    """
    df = df.copy()

    # 1. Drop rows where message is empty
    df.dropna(subset=["message"], inplace=True)
    df = df[df["message"].str.strip() != ""]
    df.reset_index(drop=True, inplace=True)

    # 2. Normalise severity
    df["severity"] = (
        df["severity"]
        .fillna("INFO")
        .astype(str)
        .str.strip()
        .apply(_normalise_severity)
    )
    # Keep only known labels
    known = {"INFO", "WARNING", "ERROR", "CRITICAL"}
    df.loc[~df["severity"].isin(known), "severity"] = "INFO"

    # 3. Fill missing metadata
    df["source"] = df["source"].fillna("unknown").astype(str)
    df["host"]   = df["host"].fillna("unknown").astype(str)
    df["timestamp"] = df["timestamp"].fillna("").astype(str)

    # 4. Clean message → `clean_message`
    df["clean_message"] = df["message"].apply(_clean_message)

    # 5. Extract hour (0-23) from timestamp for temporal features
    def _extract_hour(ts: str) -> int:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%b %d %H:%M:%S"):
            try:
                return datetime.strptime(ts.strip("[]"), fmt).hour
            except (ValueError, AttributeError):
                continue
        return -1  # unknown

    df["hour"] = df["timestamp"].apply(_extract_hour)

    # 6. Numeric severity code (used by some visualisations)
    _sev_order = {"INFO": 0, "WARNING": 1, "ERROR": 2, "CRITICAL": 3}
    df["severity_code"] = df["severity"].map(_sev_order).fillna(0).astype(int)

    return df


def load_and_preprocess(file_obj) -> pd.DataFrame:
    """Convenience wrapper: load then preprocess in one call."""
    raw_df = load_logs(file_obj)
    return preprocess(raw_df)


# ─── Quick smoke-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_and_preprocess("data/system_logs.csv")
    print(f"Loaded {len(df)} rows")
    print(df[["severity", "clean_message", "hour"]].head(5))
    print("\nSeverity distribution:")
    print(df["severity"].value_counts())
