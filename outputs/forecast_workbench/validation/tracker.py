"""
Model Run Tracker — persists forecast runs to a SQL database.

Every time a model is run, the prediction is logged with a timestamp.
On subsequent sessions you can load past runs and compare predictions
against what actually happened (model monitoring / backtesting).

The backend is resolved in ``store.py``: managed Postgres (Supabase) when
``DATABASE_URL`` is configured, otherwise a local SQLite file. Use Postgres
for any deployed instance — the local SQLite disk is wiped on redeploy.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sqlalchemy import insert, select

from validation.store import get_engine, model_runs


def log_run(
    ticker: str,
    asset_class: str,
    result: Any,          # ModelResult
) -> int:
    """
    Persist a model run to the database. Returns the new row ID.

    Raises on failure — the caller is responsible for deciding whether to
    surface the error to the user (it must, otherwise runs silently vanish).
    """
    engine = get_engine()
    stmt = insert(model_runs).values(
        run_at=datetime.now(timezone.utc),
        ticker=ticker,
        asset_class=asset_class,
        model_name=result.model_name,
        horizon_days=int(result.params.get("horizon_days", len(result.dates) - 1)),
        s0=float(result.S0),
        predicted_p50=float(result.percentiles[50][-1]),
        predicted_p5=float(result.percentiles[5][-1]),
        predicted_p95=float(result.percentiles[95][-1]),
        mu=float(result.mu),
        sigma=float(result.sigma),
        params=json.dumps(result.params),
    )
    with engine.begin() as conn:
        cursor = conn.execute(stmt)
        row_id = cursor.inserted_primary_key[0]
    return int(row_id)


def load_runs() -> pd.DataFrame:
    """Return all logged runs as a DataFrame, newest first."""
    engine = get_engine()
    stmt = select(model_runs).order_by(model_runs.c.id.desc())
    with engine.connect() as conn:
        df = pd.read_sql_query(stmt, conn)
    if df.empty:
        return df
    # Expose the column as "S0" for the UI, mirroring the old schema.
    df = df.rename(columns={"s0": "S0"})
    df["run_at"] = pd.to_datetime(df["run_at"], utc=True)
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
