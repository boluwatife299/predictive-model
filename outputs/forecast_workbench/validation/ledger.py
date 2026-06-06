"""
Validation Ledger — compare forecasts to realised price action.

Generates a structured performance report containing:
  - Point-in-time metrics (MAPE, RMSE, directional accuracy)
  - Signed error (over- vs under-prediction)
  - Analytical breakdown with lessons learned
  - A DataFrame suitable for table display in Streamlit
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from config.settings import OVER_PREDICT_THRESHOLD, UNDER_PREDICT_THRESHOLD


# ── Lesson library ─────────────────────────────────────────────────────────────

_LESSONS: dict[str, dict[str, str]] = {
    "over_vol": {
        "headline": "Model over-predicted due to a volatility spike",
        "rationale": (
            "The realised volatility exceeded historical σ calibrated at model "
            "fit. GBM / MC assume constant volatility, so a sudden spike (e.g. "
            "earnings surprise, macro shock) pushes actual prices below the "
            "modelled distribution."
        ),
        "lesson": (
            "Consider regime-switching models or GARCH-fitted σ to capture "
            "time-varying volatility. Widen confidence intervals before known "
            "event dates."
        ),
    },
    "under_vol": {
        "headline": "Model under-predicted due to a volatility compression",
        "rationale": (
            "Realised vol was lower than historical σ. The market entered a "
            "low-vol regime (e.g. summer drift, low macro uncertainty), causing "
            "actual prices to trend more smoothly than the diffusion term implied."
        ),
        "lesson": (
            "In trending, low-vol environments, momentum or trend-following "
            "overlays outperform pure diffusion models. Monitor VIX / implied vol "
            "as a real-time σ input."
        ),
    },
    "over_drift": {
        "headline": "Model over-predicted — drift (μ) was too optimistic",
        "rationale": (
            "The historical drift used for calibration included a bull-market "
            "window. Mean-reversion or a change in macro conditions caused the "
            "asset to trend below GBM's expected path."
        ),
        "lesson": (
            "Trim the calibration window or use a rolling 60-day drift rather "
            "than full-period μ. Regime detection (e.g. Hidden Markov Model) can "
            "flag when historical drift is no longer representative."
        ),
    },
    "under_drift": {
        "headline": "Model under-predicted — actual drift exceeded the forecast",
        "rationale": (
            "The asset experienced stronger-than-expected momentum or a "
            "fundamental re-rating (e.g. earnings beat, index inclusion) that "
            "lifted prices above the modelled path."
        ),
        "lesson": (
            "Supplement GBM with analyst price target data or momentum signals. "
            "Bayesian updating — adjusting μ as new information arrives — can "
            "keep forecasts current."
        ),
    },
    "accurate": {
        "headline": "Model performed within acceptable tolerance",
        "rationale": (
            "The realised price stayed close to the modelled median. The "
            "historical calibration window captured the asset's behaviour well "
            "during this forecast period."
        ),
        "lesson": (
            "Maintain the current calibration cadence. Continue monitoring for "
            "regime changes that may degrade model fit. Track whether accuracy "
            "persists across multiple forecast windows before raising confidence."
        ),
    },
}


# ── Data class ─────────────────────────────────────────────────────────────────

@dataclass
class LedgerReport:
    ticker: str
    model_name: str
    horizon_days: int
    S0: float
    predicted_terminal: float
    actual_terminal: float
    signed_error_pct: float          # positive = over-predict
    abs_error_pct: float
    mape: float
    rmse: float
    directional_accuracy: float      # fraction of days direction was correct
    lesson_key: str
    lesson: dict[str, str]
    metrics_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def over_predicted(self) -> bool:
        return self.signed_error_pct > 0

    @property
    def summary_sentence(self) -> str:
        direction = "over" if self.over_predicted else "under"
        return (
            f"The {self.model_name} {direction}-predicted the terminal price of "
            f"{self.ticker} by {abs(self.signed_error_pct):.1f}% over "
            f"{self.horizon_days} trading days."
        )


# ── Main class ─────────────────────────────────────────────────────────────────

class ValidationLedger:
    """
    Compare a ModelResult to actual out-of-sample price data.

    Usage:
        ledger = ValidationLedger(ticker="AAPL", model_name="GBM")
        report = ledger.evaluate(result, actual_close)
    """

    def __init__(self, ticker: str, model_name: str) -> None:
        self.ticker = ticker
        self.model_name = model_name

    def evaluate(
        self,
        result,                     # ModelResult
        actual_close: pd.Series,
    ) -> LedgerReport:
        """
        result       : ModelResult from a model.predict() call.
        actual_close : pd.Series of realised close prices indexed by date,
                       covering (some subset of) the forecast horizon.
        """
        horizon = result.params.get("horizon_days", len(result.dates) - 1)

        # Align dates
        forecast_dates = result.dates[1:]   # skip t=0 (S0)
        median_path    = pd.Series(
            result.percentiles[50][1:], index=forecast_dates
        )

        shared_idx = forecast_dates.intersection(actual_close.index)
        if shared_idx.empty:
            raise ValueError(
                "No overlapping dates between forecast and actual data. "
                "Make sure the validation window aligns with the forecast horizon."
            )

        pred_aligned = median_path.reindex(shared_idx)
        act_aligned  = actual_close.reindex(shared_idx)

        # ── Scalar metrics ────────────────────────────────────────────────────
        S0                = result.S0
        pred_terminal     = float(pred_aligned.iloc[-1])
        act_terminal      = float(act_aligned.iloc[-1])
        signed_err        = (pred_terminal - act_terminal) / act_terminal
        abs_err           = abs(signed_err)

        pct_errors = (pred_aligned - act_aligned).abs() / act_aligned
        mape = float(pct_errors.mean())

        rmse = float(
            np.sqrt(((pred_aligned - act_aligned) ** 2).mean())
        )

        # Directional accuracy: did model get daily direction right?
        pred_direction = np.sign(pred_aligned.diff().dropna())
        act_direction  = np.sign(act_aligned.diff().dropna())
        both_diff_idx  = pred_direction.index.intersection(act_direction.index)
        dir_acc = float(
            (pred_direction[both_diff_idx] == act_direction[both_diff_idx]).mean()
        )

        # ── Lesson selection ─────────────────────────────────────────────────
        lesson_key = self._select_lesson(
            signed_err_pct=signed_err,
            result=result,
            actual_close=act_aligned,
        )
        lesson = _LESSONS[lesson_key]

        # ── Metrics table ────────────────────────────────────────────────────
        metrics_table = pd.DataFrame({
            "Metric": [
                "Starting Price (S₀)",
                "Predicted Terminal",
                "Actual Terminal",
                "Signed Error",
                "MAPE",
                "RMSE",
                "Directional Accuracy",
                "Annualised μ (model)",
                "Annualised σ (model)",
            ],
            "Value": [
                f"${S0:,.2f}",
                f"${pred_terminal:,.2f}",
                f"${act_terminal:,.2f}",
                f"{signed_err*100:+.2f}%",
                f"{mape*100:.2f}%",
                f"${rmse:,.4f}",
                f"{dir_acc*100:.1f}%",
                f"{result.mu*100:.2f}%",
                f"{result.sigma*100:.2f}%",
            ],
        })

        return LedgerReport(
            ticker=self.ticker,
            model_name=self.model_name,
            horizon_days=horizon,
            S0=S0,
            predicted_terminal=pred_terminal,
            actual_terminal=act_terminal,
            signed_error_pct=signed_err * 100,
            abs_error_pct=abs_err * 100,
            mape=mape * 100,
            rmse=rmse,
            directional_accuracy=dir_acc * 100,
            lesson_key=lesson_key,
            lesson=lesson,
            metrics_table=metrics_table,
            extra={
                "pred_series": pred_aligned,
                "act_series":  act_aligned,
            },
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _select_lesson(
        signed_err_pct: float,
        result,
        actual_close: pd.Series,
    ) -> str:
        """
        Heuristic rule engine to pick the most explanatory lesson.

        signed_err_pct > 0  → model over-predicted
        signed_err_pct < 0  → model under-predicted
        """
        over  = OVER_PREDICT_THRESHOLD
        under = UNDER_PREDICT_THRESHOLD

        if abs(signed_err_pct) < over:
            return "accurate"

        # Estimate realised vol from the actual series
        log_ret_act = np.log(actual_close / actual_close.shift(1)).dropna()
        realised_vol = float(log_ret_act.std() * np.sqrt(252))
        model_vol    = result.sigma

        vol_spike = realised_vol > model_vol * 1.2
        vol_crush = realised_vol < model_vol * 0.8

        if signed_err_pct > 0:
            return "over_vol" if vol_spike else "over_drift"
        else:
            return "under_vol" if vol_crush else "under_drift"
