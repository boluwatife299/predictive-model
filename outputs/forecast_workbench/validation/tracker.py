"""
Model Run Tracker — persists forecast runs to a local SQLite database.

Every time a model is run, the prediction is logged with a timestamp.
On subsequent sessions you can load past runs and compare predictions
against what actually happened (model monitoring / backtesting).
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# Database lives next to this file so it survives across sessions
_DB_PATH = Path(__file__).parent / "model_runs.db"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS model_runs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at          TEXT NOT NULL,
            ticker          TEXT NOT NULL,
            asset_class     TEXT NOT NULL,
            model_name      TEXT NOT NULL,
            horizon_days    INTEGER NOT NULL,
            S0              REAL NOT NULL,
            predicted_p50   REAL NOT NULL,
            predicted_p5    REAL NOT NULL,
            predicted_p95   REAL NOT NULL,
            mu              REAL NOT NULL,
            sigma           REAL NOT NULL,
            params          TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def log_run(
    ticker: str,
    asset_class: str,
    result: Any,          # ModelResult
) -> int:
    """
    Persist a model run to the database.
    Returns the new row ID.
    """
    conn = _get_conn()
    cursor = conn.execute(
        """
        INSERT INTO model_runs
            (run_at, ticker, asset_class, model_name, horizon_days,
             S0, predicted_p50, predicted_p5, predicted_p95, mu, sigma, params)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            ticker,
            asset_class,
            result.model_name,
            result.params.get("horizon_days", len(result.dates) - 1),
            result.S0,
            float(result.percentiles[50][-1]),
            float(result.percentiles[5][-1]),
            float(result.percentiles[95][-1]),
            result.mu,
            result.sigma,
            json.dumps(result.params),
        ),
    )
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def load_runs() -> pd.DataFrame:
    """Return all logged runs as a DataFrame, newest first."""
    conn = _get_conn()
    df = pd.read_sql_query(
        "SELECT * FROM model_runs ORDER BY id DESC", conn
    )
    conn.close()
    if df.empty:
        return df
    df["run_at"] = pd.to_datetime(df["run_at"]).dt.tz_convert("UTC")
    return df


def enrich_with_actuals(df: pd.DataFrame, price_series: pd.Series) -> pd.DataFrame:
    """
    Given a DataFrame of past runs and a current price series,
    add columns for the actual terminal price and the prediction error
    for any run whose forecast horizon has already elapsed.
    """
    if df.empty:
        return df

    df = df.copy()
    df["actual_terminal"] = None
    df["error_pct"] = None

    for idx, row in df.iterrows():
        run_date = pd.Timestamp(row["run_at"]).tz_localize(None)
        target_date = run_date + pd.Timedelta(days=int(row["horizon_days"]))
        # Find the closest available price on or after the target date
        future_prices = price_series[price_series.index >= target_date]
        if future_prices.empty:
            continue
        actual = float(future_prices.iloc[0])
        predicted = float(row["predicted_p50"])
        df.at[idx, "actual_terminal"] = actual
        df.at[idx, "error_pct"] = round((predicted - actual) / actual * 100, 2)

    return df
