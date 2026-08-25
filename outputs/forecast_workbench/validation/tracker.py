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
import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import insert, select

from validation.store import get_engine, model_runs


def _finite(*values: float) -> bool:
    """True only if every value is a finite real number (no NaN / inf)."""
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


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
    s0 = float(result.S0)
    p50 = float(result.percentiles[50][-1])
    p5 = float(result.percentiles[5][-1])
    p95 = float(result.percentiles[95][-1])
    mu = float(result.mu)
    sigma = float(result.sigma)

    # Refuse to persist a degenerate run rather than hit a cryptic NOT NULL error.
    if not _finite(s0, p50, p5, p95, mu, sigma):
        raise ValueError(
            f"Forecast for {ticker} contains non-finite values (NaN) — the model "
            "could not estimate valid drift/volatility, usually due to insufficient "
            "price history. Run not saved."
        )

    engine = get_engine()
    stmt = insert(model_runs).values(
        run_at=datetime.now(timezone.utc),
        ticker=ticker,
        asset_class=asset_class,
        model_name=result.model_name,
        horizon_days=int(result.params.get("horizon_days", len(result.dates) - 1)),
        s0=s0,
        predicted_p50=p50,
        predicted_p5=p5,
        predicted_p95=p95,
        mu=mu,
        sigma=sigma,
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


def enrich_with_actuals(
    df: pd.DataFrame,
    price_map: dict[str, pd.Series],
) -> pd.DataFrame:
    """
    Add ``actual_terminal`` and ``error_pct`` columns to a runs DataFrame.

    Each run is scored against **its own ticker's** realised price history, so a
    gold run is compared to gold and an equity run to that equity — never one
    ticker's price smeared across every row (the bug that produced a constant
    "Actual Price" and nonsensical error percentages).

    Parameters
    ----------
    df : DataFrame of past runs. Needs ``ticker``, ``run_at``, ``horizon_days``
         and ``predicted_p50`` columns.
    price_map : ``{ticker: close-price Series indexed by date}`` for every
         ticker to be scored. Tickers absent from the map (e.g. delisted) are
         left unscored rather than mismatched.

    For each run we take the first available close on or after the forecast's
    target date (run date + horizon) and compute the **signed percentage error**
    of the P50 prediction:  ``(predicted − actual) / actual × 100``.
    """
    if df.empty:
        return df

    df = df.copy()
    df["actual_terminal"] = None
    df["error_pct"] = None
    df["scoring_flag"] = None

    for idx, row in df.iterrows():
        series = price_map.get(row["ticker"])
        if series is None or series.empty:
            df.at[idx, "scoring_flag"] = "no price history"
            continue

        run_date = pd.Timestamp(row["run_at"]).tz_localize(None)

        # ``horizon_days`` is TRADING days: the models step at dt = 1/252 and lay
        # their dates out on a bdate_range. Advancing the calendar by the same
        # number of days therefore lands ~30% short (30 trading days is about 42
        # calendar days), and every run was being scored against a price from too
        # early in the path. Offset in business days to match how it was forecast.
        horizon = int(row["horizon_days"])
        target_date = pd.Timestamp(
            np.busday_offset(run_date.date(), horizon, roll="forward")
        )

        future_prices = series[series.index >= target_date]
        if future_prices.empty:
            df.at[idx, "scoring_flag"] = "horizon not yet elapsed"
            continue

        actual = float(future_prices.iloc[0])
        predicted = float(row["predicted_p50"])

        if not np.isfinite(actual) or not np.isfinite(predicted):
            df.at[idx, "scoring_flag"] = "non-finite value"
            continue
        if actual <= 0:
            df.at[idx, "scoring_flag"] = "non-positive actual"
            continue

        # Entry-price integrity check.
        #
        # A ratio band on predicted/actual is the wrong test: the failure that
        # actually occurs here is a run logged against a stale entry price from a
        # previously selected instrument, and a $310 forecast on an $85 instrument
        # is only a 3.6x discrepancy, well inside any band loose enough to permit a
        # genuinely volatile asset. The precise invariant is different and cheap to
        # check: S0 was, by definition, the instrument's own price at the moment the
        # run was logged. If the logged S0 does not match that instrument's close on
        # the run date, the run does not belong to the instrument it is filed under
        # and its error is meaningless regardless of magnitude.
        #
        # The tolerance is generous because S0 may be an intraday or pre-market quote
        # against a daily close, and crypto runs are logged around the clock. It is
        # nowhere near loose enough to admit a different instrument.
        S0 = float(row["S0"]) if "S0" in row and row["S0"] is not None else float("nan")
        at_run = series[series.index <= run_date]
        if np.isfinite(S0) and S0 > 0 and not at_run.empty:
            ref = float(at_run.iloc[-1])
            drift = abs(S0 / ref - 1.0) if ref > 0 else float("inf")
            if drift > 0.25:
                df.at[idx, "actual_terminal"] = round(actual, 4)
                df.at[idx, "scoring_flag"] = (
                    f"entry-price mismatch: logged S0 {S0:,.2f} vs {row['ticker']} "
                    f"close {ref:,.2f} on {run_date.date()} ({drift*100:,.0f}% apart)"
                )
                continue

        # Residual scale guard for everything the entry check cannot see, such as a
        # split or redenomination inside the horizon: a realised move beyond 8x is a
        # corporate action or a bad series, not a forecast error, and one such row
        # moves a mean by orders of magnitude while the median does not notice.
        ratio = predicted / actual
        if not (0.125 <= ratio <= 8.0):
            df.at[idx, "actual_terminal"] = round(actual, 4)
            df.at[idx, "scoring_flag"] = (
                f"scale mismatch: predicted/actual = {ratio:,.1f}x"
            )
            continue

        df.at[idx, "actual_terminal"] = round(actual, 4)
        df.at[idx, "error_pct"] = round((predicted - actual) / actual * 100, 2)

    return df

    df = df.copy()
    df["actual_terminal"] = None
    df["error_pct"] = None

    for idx, row in df.iterrows():
        series = price_map.get(row["ticker"])
        if series is None or series.empty:
            continue
        run_date = pd.Timestamp(row["run_at"]).tz_localize(None)
        target_date = run_date + pd.Timedelta(days=int(row["horizon_days"]))
        # First available close on or after the target date.
        future_prices = series[series.index >= target_date]
        if future_prices.empty:
            continue
        actual = float(future_prices.iloc[0])
        if actual == 0:
            continue
        predicted = float(row["predicted_p50"])
        df.at[idx, "actual_terminal"] = round(actual, 4)
        df.at[idx, "error_pct"] = round((predicted - actual) / actual * 100, 2)

    return df
