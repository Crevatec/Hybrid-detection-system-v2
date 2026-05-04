# utils/data_loader.py
"""
Data Loading and Label Mapping for Hybrid Behavioral Detection System.

Label Strategy:
  - Normal traffic + infrastructure error labels --> Class 0 (Normal)
  - Credential stuffing indicators (BruteForce, SSH-Patator) --> Class 1
  - Business logic attacks (XSS, SQLi, Bot, DDoS, DoS, PortScan) --> Class 2
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# ── Label mapping ────────────────────────────────────────────────────────────
CREDENTIAL_STUFFING_LABELS = {
    # Original dataset labels
    "Web Attack \ufffdBrute Force",
    "Web Attack  Brute Force",
    "Web Attack – Brute Force",
    "Web Attack - Brute Force",
    "SSH-Patator",
    "FTP-Patator",
    # New dataset (api_security_dataset_2026)
    "Credential_Stuffing",
    "Credential Stuffing",
    "credential_stuffing",
    "brute_force",
    "BruteForce",
}

BUSINESS_LOGIC_LABELS = {
    # Original dataset labels
    "Web Attack \ufffdXSS",
    "Web Attack  XSS",
    "Web Attack – XSS",
    "Web Attack - XSS",
    "Web Attack \ufffdSql Injection",
    "Web Attack  Sql Injection",
    "Web Attack – Sql Injection",
    "Web Attack - Sql Injection",
    "Bot",
    "DDoS",
    "DoS slowloris",
    "DoS GoldenEye",
    "DoS Hulk",
    "DoS Slowhttptest",
    "PortScan",
    "Infiltration",
    "Heartbleed",
    # New dataset (api_security_dataset_2026)
    "Business_Logic",
    "Business Logic",
    "business_logic",
    "BusinessLogic",
    "api_abuse",
    "scraping",
    "injection",
}


def map_label(raw_label: str) -> int:
    """Map raw CSV label to 0=Normal, 1=CredentialStuffing, 2=BusinessLogic."""
    label = str(raw_label).strip()
    if label in CREDENTIAL_STUFFING_LABELS:
        return 1
    if label in BUSINESS_LOGIC_LABELS:
        return 2
    return 0


def load_and_prepare_data(
    csv_path: str,
    sample_size: int = 300_000,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True,
):
    """
    Load the hybrid master dataset, map labels, and return train/test splits.

    Parameters
    ----------
    csv_path      : path to hybrid_master_dataset.csv
    sample_size   : max rows to load (stratified sample for speed)
    test_size     : fraction held out for testing
    random_state  : reproducibility seed
    verbose       : print progress info

    Returns
    -------
    df_full       : full processed DataFrame (sampled)
    X_train, X_test, y_train, y_test : numpy arrays, scaled
    feature_names : list of feature column names
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at: {csv_path}\n"
            "Place hybrid_master_dataset.csv in the data/ folder."
        )

    if verbose:
        print(f"[DataLoader] Loading dataset from {csv_path} ...")

    # ── 1. Load a stratified sample (full file is ~3.2 M rows) ───────────────
    total_rows = sum(1 for _ in open(csv_path)) - 1  # exclude header
    if verbose:
        print(f"[DataLoader] Total rows in file: {total_rows:,}")

    if total_rows > sample_size:
        # Random sample with uniform row selection
        skip_idx = sorted(
            np.random.RandomState(random_state).choice(
                range(1, total_rows + 1),
                size=total_rows - sample_size,
                replace=False,
            )
        )
        df = pd.read_csv(csv_path, skiprows=skip_idx)
    else:
        df = pd.read_csv(csv_path)

    if verbose:
        print(f"[DataLoader] Loaded {len(df):,} rows.")

    # ── 2. Map labels ─────────────────────────────────────────────────────────
    df["label_raw"] = df["label"].astype(str)
    df["label"] = df["label_raw"].apply(map_label)

    if verbose:
        counts = df["label"].value_counts().sort_index()
        label_names = {0: "Normal", 1: "CredentialStuffing", 2: "BusinessLogic"}
        for cls, cnt in counts.items():
            pct = 100 * cnt / len(df)
            print(f"  Class {cls} ({label_names[cls]}): {cnt:,} ({pct:.1f}%)")

    # ── 3. Feature engineering is handled by feature_engineering.py ──────────
    from utils.feature_engineering import engineer_features

    df_feat = engineer_features(df, verbose=verbose)

    # ── 4. Split ──────────────────────────────────────────────────────────────
    feature_cols = [c for c in df_feat.columns if c not in ("label", "label_raw")]
    X = df_feat[feature_cols].values.astype(np.float32)
    y = df_feat["label"].values.astype(np.int32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    if verbose:
        print(f"[DataLoader] Train: {X_train.shape}, Test: {X_test.shape}")

    return df_feat, X_train, X_test, y_train, y_test, feature_cols
