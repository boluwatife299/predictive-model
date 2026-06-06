"""
ARIMA — AutoRegressive Integrated Moving Average.

Best for: assets with trend and autocorrelation in returns.
          The most foundational time-series model — every quant should
          understand it before anything fancier.

Mathematics:
    ARIMA(p, d, q) on the log-price series:

    d  : number of differences to make series stationary
         (d=1 → model works on log-returns, not log-prices)
    p  : AR order — how many lagged values predict today
         φ₁y_{t-1} + φ₂y_{t-2} + ... + φ_p·y_{t-p}
    q  : MA order — how many lagged forecast errors predict today
         θ₁ε_{t-1} + ... + θ_q·ε_{t-q}

    Full equation:
    y_t = c + Σφ_i·y_{t-i} + Σθ_j·ε_{t-j} + ε_t

Forecast uncertainty grows with horizon (wider confidence intervals further out).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from models.base_model import BaseModel, ModelResult

try:
    from statsmodels.tsa.arima.model import ARIMA as _ARIMA
    _STATSMODELS_OK = True
except ImportError:
    _STATSMODELS_OK = False

warnings.filterwarnings("ignore")

# Number of simulated paths (ARIMA is deterministic; we add forecast-error noise)
_N_PATHS = 200


class ARIMAModel(BaseModel):
    name = "ARIMA"

    def __init__(
        self,
        p: int = 2,
        d: int = 1,
        q: int = 2,
        horizon_days: int = 30,
        seed: int = 42,
    ) -> None:
        self.p = p
        self.d = d
        self.q = q
        self.horizon_days = horizon_days
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

    def fit(self, preprocessor) -> "ARIMAModel":
        if not _STATSMODELS_OK:
            raise RuntimeError(
                "statsmodels is required for ARIMA. "
                "Run: pip install statsmodels"
            )
        self._mu = preprocessor.mu
        self._sigma = preprocessor.sigma
        self._S0 = preprocessor.S0
        self._last_date = preprocessor.train.index[-1]
        self._live_S0 = preprocessor.live_S0
        self._live_last_date = preprocessor.live_last_date
        self._train_series = preprocessor.train["Close"]
        self._full_series = preprocessor.clean["Close"]
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

        # Fit on log prices (ensures positive forecasts after exp)
        log_series = np.log(series)

        try:
            model = _ARIMA(log_series, order=(self.p, self.d, self.q))
            fitted = model.fit()
            fc = fitted.get_forecast(steps=T)
            fc_frame = fc.summary_frame(alpha=0.05)
            mean_log = fc_frame["mean"].values
            se_log   = fc_frame["mean_se"].values
        except Exception as exc:
            # Fallback: simple drift forecast if ARIMA fails to converge
            daily_log_ret = float(np.log(series).diff().mean())
            mean_log = np.log(S0) + daily_log_ret * np.arange(1, T + 1)
            se_log   = np.full(T, self._sigma / np.sqrt(252))

        # Point forecast in price space
        point_forecast = np.exp(mean_log)

        # Simulate N paths by adding heteroskedastic noise in log space
        # Uncertainty grows with sqrt(t) — consistent with random walk theory
        time_scale = np.sqrt(np.arange(1, T + 1))
        noise = rng.normal(0, 1, (_N_PATHS, T)) * se_log[np.newaxis, :] * time_scale[np.newaxis, :]
        log_paths = mean_log[np.newaxis, :] + noise
        price_paths = np.exp(log_paths)

        # Prepend S0
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
                "p": self.p,
                "d": self.d,
                "q": self.q,
                "horizon_days": T,
            },
            metadata={
                "point_forecast_terminal": round(float(point_forecast[-1]), 4),
                "forecast_stderr_terminal": round(float(se_log[-1]), 6),
            },
        )

    @classmethod
    def param_schema(cls) -> dict:
        return {
            "p": (2, 0, 5, 1,
                "AR order — lags of past values used. Start with 1-2."),
            "d": (1, 0, 2, 1,
                "Differencing order. 1 = model returns (standard for prices)."),
            "q": (2, 0, 5, 1,
                "MA order — lags of past forecast errors used."),
            "horizon_days": (30, 5, 252, 5,
                "Trading days to forecast forward."),
            "seed": (42, 0, 9999, 1,
                "Random seed for path simulation noise."),
        }
