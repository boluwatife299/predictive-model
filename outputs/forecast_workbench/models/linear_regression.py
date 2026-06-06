"""
Linear Regression (OLS Baseline).

The simplest possible ML model — and your benchmark.
If your complex models (LSTM, XGBoost, GARCH) can't beat this,
they're overfitting or misconfigured.

Features used:
    - Lagged closing prices (lag_1 to lag_lookback)
    - Rolling mean (20-day)
    - Rolling std (20-day)
    - Time index (captures linear trend)

Forecast method: recursive multi-step
    At each step t, predict price_{t+1} using the last `lookback` prices.
    Append the prediction, shift the window, repeat for T steps.

Uncertainty: residual std from training fit, scaled by √t to model
             growing uncertainty with horizon.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models.base_model import BaseModel, ModelResult

try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

_N_PATHS = 200
_LOOKBACK = 20


def _make_features(prices: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """Build feature matrix X and target y from a price array."""
    X, y = [], []
    for i in range(lookback, len(prices)):
        window = prices[i - lookback: i]
        feat = list(window)                        # lag features
        feat.append(np.mean(window))               # rolling mean
        feat.append(np.std(window))                # rolling std
        feat.append(float(i))                      # time index
        X.append(feat)
        y.append(prices[i])
    return np.array(X, dtype=float), np.array(y, dtype=float)


class LinearRegressionModel(BaseModel):
    name = "Linear Regression"

    def __init__(
        self,
        lookback: int = 20,
        horizon_days: int = 30,
        seed: int = 42,
    ) -> None:
        self.lookback = lookback
        self.horizon_days = horizon_days
        self.seed = seed

        self._mu: float = 0.0
        self._sigma: float = 0.0
        self._S0: float = 0.0
        self._last_date: pd.Timestamp | None = None
        self._live_S0: float = 0.0
        self._live_last_date: pd.Timestamp | None = None
        self._train_prices: np.ndarray | None = None
        self._full_prices: np.ndarray | None = None

    # ------------------------------------------------------------------

    def fit(self, preprocessor) -> "LinearRegressionModel":
        if not _SKLEARN_OK:
            raise RuntimeError(
                "scikit-learn is required. Run: pip install scikit-learn"
            )
        self._mu = preprocessor.mu
        self._sigma = preprocessor.sigma
        self._S0 = preprocessor.S0
        self._last_date = preprocessor.train.index[-1]
        self._live_S0 = preprocessor.live_S0
        self._live_last_date = preprocessor.live_last_date
        self._train_prices = preprocessor.train["Close"].values
        self._full_prices  = preprocessor.clean["Close"].values
        return self

    def predict(self) -> ModelResult:
        return self._run(self._train_prices, self._S0, self._last_date)

    def predict_forward(self) -> ModelResult:
        return self._run(self._full_prices, self._live_S0, self._live_last_date)

    # ------------------------------------------------------------------

    def _run(
        self,
        prices: np.ndarray,
        S0: float,
        last_date: pd.Timestamp,
    ) -> ModelResult:
        T = self.horizon_days
        lb = min(self.lookback, len(prices) - 2)
        rng = np.random.default_rng(self.seed)

        X, y = _make_features(prices, lb)
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)

        model = Ridge(alpha=1.0)
        model.fit(X_sc, y)
        residual_std = float(np.std(y - model.predict(X_sc)))

        # Recursive multi-step forecast
        history = list(prices[-lb:])
        point_forecast = []
        n_hist = len(prices)

        for step in range(T):
            window = np.array(history[-lb:])
            feat = list(window) + [np.mean(window), np.std(window), float(n_hist + step)]
            feat_sc = scaler.transform([feat])
            pred = float(model.predict(feat_sc)[0])
            point_forecast.append(pred)
            history.append(pred)

        point_forecast = np.array(point_forecast)

        # Simulate N paths: add noise scaled by sqrt(t) for growing uncertainty
        time_scale = np.sqrt(np.arange(1, T + 1))
        noise = rng.normal(0, 1, (_N_PATHS, T)) * residual_std * time_scale[np.newaxis, :]
        price_paths = point_forecast[np.newaxis, :] + noise
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
                "lookback_days": lb,
                "horizon_days": T,
                "regularisation_alpha": 1.0,
            },
            metadata={
                "residual_std": round(residual_std, 4),
                "point_forecast_terminal": round(float(point_forecast[-1]), 4),
            },
        )

    @classmethod
    def param_schema(cls) -> dict:
        return {
            "lookback": (20, 5, 60, 5,
                "Number of past days used as lag features."),
            "horizon_days": (30, 5, 252, 5,
                "Trading days to forecast forward."),
            "seed": (42, 0, 9999, 1,
                "Random seed."),
        }
