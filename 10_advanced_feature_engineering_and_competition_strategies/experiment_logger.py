"""Lightweight experiment logger for reproducible Kaggle workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib
import json

import pandas as pd


@dataclass
class ExperimentRecord:
    exp_id: str
    seed: int
    feature_set: str
    model_name: str
    cv_score: float
    public_score: float | None = None
    private_score: float | None = None
    notes: str = ""
    params: dict[str, Any] | None = None

    def config_hash(self) -> str:
        payload = json.dumps(self.params or {}, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:12]


class ExperimentLogger:
    """Append-only JSONL logger with convenience dataframe summaries."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: ExperimentRecord) -> None:
        row = asdict(record)
        row["config_hash"] = record.config_hash()
        row["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    def to_frame(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame()
        rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)

    def summary(self) -> pd.DataFrame:
        df = self.to_frame()
        if df.empty:
            return df
        if {"public_score", "private_score"}.issubset(df.columns):
            df["public_private_gap"] = df["private_score"] - df["public_score"]
        return df.sort_values("timestamp_utc").reset_index(drop=True)
