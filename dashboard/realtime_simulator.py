# dashboard/realtime_simulator.py
"""
Real-Time Login Stream Simulator.

Reads the CSV in random chunks and yields processed feature rows
one-by-one to simulate a live API traffic feed.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import time
from typing import Generator, Tuple

from utils.feature_engineering import engineer_features


LABEL_NAMES = {0: "Normal", 1: "Credential Stuffing", 2: "Business Logic Attack"}
LABEL_COLORS = {0: "green", 1: "orange", 2: "red"}

CREDENTIAL_STUFFING_LABELS = {
    "Web Attack \ufffdBrute Force", "Web Attack  Brute Force",
    "Web Attack – Brute Force", "Web Attack - Brute Force",
    "SSH-Patator", "FTP-Patator",
}
BUSINESS_LOGIC_LABELS = {
    "Web Attack \ufffdXSS", "Web Attack  XSS", "Web Attack – XSS", "Web Attack - XSS",
    "Web Attack \ufffdSql Injection", "Web Attack  Sql Injection",
    "Web Attack – Sql Injection", "Web Attack - Sql Injection",
    "Bot", "DDoS", "DoS slowloris", "DoS GoldenEye",
    "DoS Hulk", "DoS Slowhttptest", "PortScan",
}


def map_label(raw: str) -> int:
    s = str(raw).strip()
    if s in CREDENTIAL_STUFFING_LABELS:
        return 1
    if s in BUSINESS_LOGIC_LABELS:
        return 2
    return 0


class RealtimeSimulator:
    """
    Yields individual login events from the dataset at a controlled rate.

    Parameters
    ----------
    csv_path     : path to hybrid_master_dataset.csv
    chunk_size   : rows to read per CSV chunk
    delay_sec    : simulated delay between events (0 = as fast as possible)
    random_seed  : for reproducible random sampling
    """

    def __init__(
        self,
        csv_path: str,
        chunk_size: int = 5000,
        delay_sec: float = 0.0,
        random_seed: int = 42,
    ):
        self.csv_path   = csv_path
        self.chunk_size = chunk_size
        self.delay_sec  = delay_sec
        self.rng        = np.random.default_rng(random_seed)

    def stream(self) -> Generator[Tuple[np.ndarray, dict], None, None]:
        """
        Yields (feature_vector [1, n_features], metadata_dict) tuples.

        metadata_dict keys:
          timestamp, endpoint, method, status_code, latency_ms,
          label_raw (original CSV label), label_true (int 0/1/2),
          label_name (str)
        """
        reader = pd.read_csv(self.csv_path, chunksize=self.chunk_size)
        for chunk in reader:
            # Shuffle within chunk for realistic mixed traffic
            chunk = chunk.sample(frac=1, random_state=int(self.rng.integers(1e6)))
            chunk["label_raw"] = chunk["label"].astype(str)
            chunk["label"]     = chunk["label_raw"].apply(map_label)

            try:
                feat_df = engineer_features(chunk, fit=False, verbose=False)
            except FileNotFoundError:
                # Scaler not yet fitted — skip engineering for simulator preview
                feat_df = chunk.copy()
                feat_df["label"] = chunk["label"]

            feature_cols = [c for c in feat_df.columns if c not in ("label", "label_raw")]
            X = feat_df[feature_cols].values.astype(np.float32)

            for i in range(len(chunk)):
                row = chunk.iloc[i]
                meta = {
                    "timestamp":   str(row.get("timestamp", "")),
                    "endpoint":    str(row.get("endpoint", "")),
                    "method":      str(row.get("method", "")),
                    "status_code": int(row.get("status_code", 0) or 0),
                    "latency_ms":  float(row.get("latency_ms", 0) or 0),
                    "label_raw":   str(row.get("label_raw", "Unknown")),
                    "label_true":  int(chunk.iloc[i]["label"]),
                    "label_name":  LABEL_NAMES[int(chunk.iloc[i]["label"])],
                }
                yield X[i:i+1], meta

                if self.delay_sec > 0:
                    time.sleep(self.delay_sec)
