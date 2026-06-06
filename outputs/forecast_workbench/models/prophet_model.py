"""
Facebook Prophet.

Best for: assets with strong trend + seasonal structure
          (e.g. gold, commodities, BTC with halving cycles, seasonal ETFs).

How it works:
    Prophet decomposes the time series into:
        y(t) = trend(t) + seasonality(t) + holidays(t) + noise

    Trend: piecewise linear or logistic growth with automatic changepoint detection.
    Seasonality: Fourier series for weekly and yearly patterns.
    Uncertainty: Monte Carlo samples from posterior (uncertainty_samples).

    changepoint_prior_scale: higher = more flexible trend (more changepoints allowed).
    Larger values adapt to recent regime changes; smaller = smoother trend.

Note: Prophet is designed for business/web metrics but works well for
financial series with strong trend and seasonality. It is NOT a stochastic
price model — it finds structure. Use alongside GBM/Monte Carlo, not instead.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from models.base_model import BaseModel, ModelResult

try:
    from prophet import Prophet as _Prophet
    _PROPHET_OK = True
except ImportError:
    _PROPHET_OK = False

warnings.filterwarnings("ignore")

_N_PATHS = 200


class ProphetModel(BaseModel):
    name = "Prophet"

    def __init__(
        self,
        horizon_days: int = 30,
        changepoint_prior_scale: int = 5,   # slider: 1-20, divided by 100 internally
        seed: int = 42,
    ) -> None:
        self.horizon_days = horizon_days
        self.changepoint_prior_scale = changepoint_prior_scale / 100.0
        self.seed = seed

        self._mu: float = 0.0
        self._sigma: float = 0.0
        self._S0: float = 0.0
        self._last_date: pd.Timestamp | None = None
        self._live_S0: float = 0.0
        self._live_last_date: pd.Timestamp | None = None
        self._train_series: pd.Series | None = None
        self._full_series: pd.Series | None = None

    # ------------------------------------------------------------------

    def fit(self, preprocessor) -> "ProphetModel":
        if not _PROPHET_OK:
            raise RuntimeError(
                "prophet is required. Run: pip install prophet"
            )
        self._mu = preprocessor.mu
        self._sigma = preprocessor.sigma
        self._S0 = preprocessor.S0
        self._last_date = preprocessor.train.index[-1]
        self._live_S0 = preprocessor.live_S0
        self._live_last_date = preprocessor.live_last_date
        self._train_series = preprocessor.train["Close"]
        self._full_series  = preprocessor.clean["Close"]
        return self

    def predict(self) -> ModelResult:
        return self._run(self._train_series, self._S0, self._last_date)

    def predict_forward(self) -> ModelResult:
        return self._run(self._full_series, self._live_S0, self._live_last_date)

    # ------------------------------------------------------------------

    def _run(
        self,
        series: pd.Series,
        S0: float,
        last_date: pd.Timestamp,
    ) -> ModelResult:
        T = self.horizon_days
        rng = np.random.default_rng(self.seed)

        df_prophet = pd.DataFrame({
            "ds": series.index.tz_localize(None),
            "y":  series.values,
        })

        try:
            m = _Prophet(
                changepoint_prior_scale=self.changepoint_prior_scale,
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                uncertainty_samples=_N_PATHS,
                stan_backend="CMDSTANPY" if False else None,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(df_prophet)

            future = m.make_future_dataframe(periods=T, freq="B")
            fc = m.predict(future)

            # Extract the T forward rows only
            fc_forward = fc.tail(T)
            yhat       = fc_forward["yhat"].values
            yhat_lower = fc_forward["yhat_lower"].values
            yhat_upper = fc_forward["yhat_upper"].values

        except Exception as exc:
            # Fallback: simple linear extrapolation
            slope = float(np.polyfit(np.arange(len(series)), series.values, 1)[0])
            yhat = S0 + slope * np.arange(1, T + 1)
            band = self._sigma / np.sqrt(252) * S0 * np.sqrt(np.arange(1, T + 1))
            yhat_lower = yhat - 1.96 * band
            yhat_upper = yhat + 1.96 * band

        # Simulate paths between lower and upper bounds
        yhat_lower = np.maximum(yhat_lower, S0 * 0.05)
        mid  = yhat
        half = (yhat_upper - yhat_lower) / 2.0
        std_approx = half / 1.96

        noise = rng.normal(0, 1, (_N_PATHS, T))
        price_paths = mid[np.newaxis, :] + noise * std_approx[np.newaxis, :]
        price_paths = np.maximum(price_paths, S0 * 0.05)

        paths = np.hstack([np.full((_N_PATHS, 1), S0), price_paths])
        dates = pd.bdate_range(start=last_date, periods=T + 1)

        return ModelResult(
            paths=paths,
            dates=dates,
            S0=S0,
            mu=self._mu,
            sigma=self._sigma,
            model_name=self.name,
            params={
                "horizon_days": T,
                "changepoint_prior_scale": self.changepoint_prior_scale,
            },
            metadata={
                "point_forecast_terminal": round(float(yhat[-1]), 4),
                "lower_80_terminal": round(float(yhat_lower[-1]), 4),
                "upper_80_terminal": round(float(yhat_upper[-1]), 4),
            },
        )

    @classmethod
    def param_schema(cls) -> dict:
        return {
            "horizon_days": (30, 5, 252, 5,
                "Trading days to forecast forward."),
            "changepoint_prior_scale": (5, 1, 20, 1,
                "Trend flexibility (1-20, scaled to 0.01-0.20). "
                "Higher = more changepoints allowed, more responsive to recent trend."),
            "seed": (42, 0, 9999, 1,
                "Random seed."),
        }
