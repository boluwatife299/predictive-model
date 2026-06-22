"""
Model-driven strategy — the bridge between the Model Zoo and the backtester.

At each rebalance bar it refits the chosen forecasting model on the *trailing*
window (data up to that bar only — no lookahead), reads the forecast, and turns
it into a decision:

    expected return = P50 terminal / S0 − 1
        > +threshold        → go long
        < −threshold        → go short (if shorting enabled)
        otherwise           → flat

When the model produces a genuine distribution (everything except single-path
GBM) the P5/P95 terminal levels become the protective stop and profit target —
so the same maths that draws the forecast cone also sizes the risk.

This is the wiring the workbench was missing: a forecast is no longer a chart, it
is an instruction that gets scored over history.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.preprocessor import Preprocessor
from models import REGISTRY
from strategy.base import Signal, Strategy


class ModelForecastStrategy(Strategy):
    name = "Model Forecast (Model Zoo)"
    description = (
        "Refits a chosen Model-Zoo model on a rolling window and trades its "
        "forecast: long when expected return clears a threshold, with the P5/P95 "
        "cone as stop/target."
    )

    def __init__(
        self,
        model_key: str = "gbm",
        horizon_days: int = 21,
        lookback_window: int = 252,
        rebalance_every: int = 10,
        entry_threshold_pct: float = 2.0,
        use_cone_stops: bool = True,
    ) -> None:
        self.model_key = model_key
        self.horizon_days = int(horizon_days)
        self.lookback_window = int(lookback_window)
        self.rebalance_every = max(1, int(rebalance_every))
        self.entry_threshold = float(entry_threshold_pct) / 100.0
        self.use_cone_stops = bool(use_cone_stops)
        self.warmup = self.lookback_window

        # Diagnostics surfaced to the UI after a run.
        self.n_refits = 0
        self.n_failures = 0

    # ------------------------------------------------------------------
    def prepare(self, df: pd.DataFrame) -> None:
        """Precompute per-bar direction / stop / target by walking forward."""
        self.df = df
        n = len(df)
        self.dir_arr = np.zeros(n, dtype=int)
        self.stop_arr = np.full(n, np.nan)
        self.tgt_arr = np.full(n, np.nan)
        self.exp_ret_arr = np.full(n, np.nan)

        ModelClass = REGISTRY[self.model_key]
        # Hold a small tail out of the μ/σ estimate; the forecast still starts
        # from the current bar via predict_forward().
        val_days = max(5, self.lookback_window // 8)

        last_dir, last_stop, last_tgt, last_exp = 0, np.nan, np.nan, np.nan
        for i in range(n):
            due = i >= self.lookback_window and (
                (i - self.lookback_window) % self.rebalance_every == 0
            )
            if due:
                window = df.iloc[i - self.lookback_window + 1 : i + 1]
                try:
                    prep = Preprocessor(validation_days=val_days)
                    prep.fit(window)
                    model = ModelClass(horizon_days=self.horizon_days)
                    model.fit(prep)
                    res = model.predict_forward()
                    s0 = float(res.S0)
                    p50 = float(res.percentiles[50][-1])
                    p5 = float(res.percentiles[5][-1])
                    p95 = float(res.percentiles[95][-1])
                    if not np.isfinite([s0, p50, p5, p95]).all() or s0 <= 0:
                        raise ValueError("non-finite forecast")

                    last_exp = p50 / s0 - 1.0
                    if last_exp > self.entry_threshold:
                        last_dir = 1
                    elif last_exp < -self.entry_threshold:
                        last_dir = -1
                    else:
                        last_dir = 0

                    band = (p95 - p5) / s0
                    if self.use_cone_stops and last_dir != 0 and band > 0.02:
                        if last_dir > 0:
                            last_stop, last_tgt = p5, p95
                        else:
                            last_stop, last_tgt = p95, p5
                    else:
                        last_stop, last_tgt = np.nan, np.nan
                    self.n_refits += 1
                except Exception:
                    last_dir, last_stop, last_tgt, last_exp = 0, np.nan, np.nan, np.nan
                    self.n_failures += 1

            self.dir_arr[i] = last_dir
            self.stop_arr[i] = last_stop
            self.tgt_arr[i] = last_tgt
            self.exp_ret_arr[i] = last_exp

    def signal(self, i: int, position: int) -> Signal:
        d = int(self.dir_arr[i])
        stop = self.stop_arr[i]
        tgt = self.tgt_arr[i]
        exp = self.exp_ret_arr[i]
        reason = (
            f"{self.model_key}: expected {exp * 100:+.1f}% over {self.horizon_days}d"
            if np.isfinite(exp)
            else "warming up"
        )
        return Signal(
            direction=d,
            stop=float(stop) if np.isfinite(stop) else None,
            target=float(tgt) if np.isfinite(tgt) else None,
            reason=reason,
        )

    @classmethod
    def param_schema(cls) -> dict[str, tuple]:
        return {
            "horizon_days": (
                21, 5, 90, 1,
                "Forecast horizon the model projects at each rebalance.",
            ),
            "lookback_window": (
                252, 60, 504, 10,
                "Trailing bars the model is refit on (the rolling training window).",
            ),
            "rebalance_every": (
                10, 1, 30, 1,
                "Refit/redecide every N bars. Larger = faster backtest, fewer trades.",
            ),
            "entry_threshold_pct": (
                2.0, 0.0, 15.0, 0.5,
                "Minimum expected return (%) to take a position.",
            ),
        }
