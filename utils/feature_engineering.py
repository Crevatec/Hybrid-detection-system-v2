# utils/feature_engineering.py
"""
Feature Engineering for Hybrid Behavioral Detection System.

Extracts:
  - Time-based features from timestamps (hour, minute, day_of_week, etc.)
  - Encoded categorical features (endpoint hash, HTTP method one-hot)
  - Scaled numerical features (latency_ms, status_code)
  - Derived features (latency buckets, status family)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

# Paths for saving/loading fitted scalers & encoders
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "saved_models")


def engineer_features(df: pd.DataFrame, fit: bool = True, verbose: bool = True) -> pd.DataFrame:
    """
    Transform raw DataFrame into a fully numeric feature matrix.

    Parameters
    ----------
    df   : DataFrame with columns timestamp, user_id, ip_address, endpoint,
           method, status_code, latency_ms, label
    fit  : if True, fit (and save) scalers; if False, load saved scalers

    Returns
    -------
    df_out : DataFrame with numeric feature columns + label
    """
    df = df.copy()

    # ── 1. Timestamp features ─────────────────────────────────────────────────
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["hour"]         = df["timestamp"].dt.hour.fillna(0).astype(int)
    df["minute"]       = df["timestamp"].dt.minute.fillna(0).astype(int)
    df["day_of_week"]  = df["timestamp"].dt.dayofweek.fillna(0).astype(int)
    df["day_of_month"] = df["timestamp"].dt.day.fillna(1).astype(int)
    df["month"]        = df["timestamp"].dt.month.fillna(1).astype(int)
    # Cyclical encoding for hour (captures 23→0 wrap-around)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # ── 2. HTTP Method one-hot ────────────────────────────────────────────────
    method_dummies = pd.get_dummies(
        df["method"].fillna("UNKNOWN"), prefix="method", dtype=float
    )
    # Ensure consistent columns across train/test
    for m in ["method_DELETE", "method_GET", "method_PATCH", "method_POST", "method_PUT"]:
        if m not in method_dummies.columns:
            method_dummies[m] = 0.0
    method_dummies = method_dummies[sorted(method_dummies.columns)]

    # ── 3. Endpoint encoding (hash to integer bucket) ─────────────────────────
    df["endpoint_hash"] = df["endpoint"].fillna("").apply(
        lambda x: abs(hash(x)) % 10_000
    ).astype(float)

    # ── 4. Status code features ───────────────────────────────────────────────
    df["status_code"] = pd.to_numeric(df["status_code"], errors="coerce").fillna(200)
    df["status_family"] = (df["status_code"] // 100).astype(float)   # 2,3,4,5
    df["is_error"] = (df["status_code"] >= 400).astype(float)
    df["is_server_error"] = (df["status_code"] >= 500).astype(float)
    df["is_auth_error"] = df["status_code"].isin([401, 403]).astype(float)

    # ── 5. Latency features ───────────────────────────────────────────────────
    df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce").fillna(0)
    df["latency_log"] = np.log1p(df["latency_ms"])
    df["is_high_latency"] = (df["latency_ms"] > 5000).astype(float)

    # ── 6. User / IP features ─────────────────────────────────────────────────
    # Handle both numeric and string formats (e.g. "user_183", "192.168.1.1")
    # Extract numeric part from strings like "user_183" → 183
    def extract_numeric(series):
        # Try direct numeric conversion first
        numeric = pd.to_numeric(series, errors="coerce")
        # For strings like "user_183", extract the trailing number
        if numeric.isna().sum() > len(series) * 0.5:
            numeric = series.astype(str).str.extract(r"(\d+)")[0]
            numeric = pd.to_numeric(numeric, errors="coerce")
        return numeric.fillna(-1)

    def ip_to_numeric(series):
        # Try direct numeric first
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.isna().sum() > len(series) * 0.5:
            # Convert dotted IP "192.168.1.144" → integer
            def ip_int(ip):
                try:
                    parts = str(ip).split(".")
                    if len(parts) == 4:
                        return int(parts[0]) * 16777216 + int(parts[1]) * 65536 + \
                               int(parts[2]) * 256 + int(parts[3])
                except Exception:
                    pass
                return 0
            numeric = series.apply(ip_int).astype(float)
        return numeric.fillna(0)

    df["user_id"]    = extract_numeric(df["user_id"])
    df["ip_address"] = ip_to_numeric(df["ip_address"])

    # ── 7. Assemble feature matrix ────────────────────────────────────────────
    numeric_cols = [
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "day_of_month", "month",
        "endpoint_hash",
        "status_code", "status_family", "is_error", "is_server_error", "is_auth_error",
        "latency_log", "is_high_latency",
        "user_id",
    ]
    df_out = pd.concat([df[numeric_cols], method_dummies], axis=1)

    # ── 8. Scale numerical columns ────────────────────────────────────────────
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    scaler_path = os.path.join(ARTIFACT_DIR, "scaler.pkl")

    scale_cols = ["endpoint_hash", "status_code", "latency_log", "user_id", "day_of_month", "month"]
    scale_cols = [c for c in scale_cols if c in df_out.columns]

    if fit:
        scaler = StandardScaler()
        df_out[scale_cols] = scaler.fit_transform(df_out[scale_cols].astype(float))
        joblib.dump(scaler, scaler_path)
        if verbose:
            print(f"[FeatureEng] Scaler fitted & saved → {scaler_path}")
    else:
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            df_out[scale_cols] = scaler.transform(df_out[scale_cols].astype(float))
        else:
            raise FileNotFoundError("Scaler not found – run training first.")

    df_out["label"] = df["label"].values if "label" in df.columns else 0
    if "label_raw" in df.columns:
        df_out["label_raw"] = df["label_raw"].values

    if verbose:
        print(f"[FeatureEng] Final feature matrix shape: {df_out.shape}")

    return df_out
