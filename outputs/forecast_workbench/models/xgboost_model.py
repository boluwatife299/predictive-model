"""
XGBoost — Gradient Boosted Decision Trees.

Surprisingly competitive against deep learning for tabular/time-series data.
Captures non-linear relationships between lagged prices and future prices.
Often beats LSTM on shorter financial series.

How it works:
    1. Create lag features: price_{t-1}, ..., price_{t-lookback},
       rolling mean, rolling std, log-returns.
    2. Train XGBRegressor to predict price_t from those features.
    3. Recursive T-step forecast: each prediction becomes a new lag feature.

Key risk: recursive errors compound — uncertainty grows faster than linear
          models because each step uses a predicted (not observed) input.

Use the residual std from training as the uncertainty envelope.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models.base_model import BaseModel, ModelResult

try:
    import xgboost as xgb
    _XGB_OK = True
except ImportError:
    _XGB_OK = False

_N_PATHS = 200
_LOOKBACK = 20


def _make_xgb_features(prices: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    log_ret = np.diff(np.log(prices))
    for i in range(lookback, len(prices)):
        window = prices[i - lookback: i]
        ret_window = log_ret[max(i - lookback - 1, 0): i - 1]
        feat = (
            list(window)
            + [np.mean(window), np.std(window + 1e-8)]
            + [np.mean(ret_window[-5:]) if len(ret_window) >= 5 else 0.0]
            + [float(i)]
        )
        X.append(feat)
        y.append(prices[i])
    return np.array(X, dtype=float), np.array(y, dtype=float)


class XGBoostModel(BaseModel):
    name = "XGBoost"

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        lookback: int = 20,
        horizon_days: int = 30,
        seed: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
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

    def fit(self, preprocessor) -> "XGBoostModel":
        if not _XGB_OK:
            raise RuntimeError(
                "xgboost is required. Run: pip install xgboost"
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

        X, y = _make_xgb_features(prices, lb)

        model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.seed,
            verbosity=0,
        )
        model.fit(X, y)
        residual_std = float(np.std(y - model.predict(X)))

        # Recursive forecast
        history = list(prices)
        point_forecast = []
        log_ret_hist = list(np.diff(np.log(prices)))

        for step in range(T):
            window = np.array(history[-lb:])
            ret_window = np.array(log_ret_hist[-lb:])
            feat = (
                list(window)
                + [np.mean(window), np.std(window + 1e-8)]
                + [np.mean(ret_window[-5:]) if len(ret_window) >= 5 else 0.0]
                + [float(len(prices) + step)]
            )
            pred = float(model.predict(np.array([feat]))[0])
            point_forecast.append(pred)
            history.append(pred)
            log_ret_hist.append(np.log(pred / history[-2]) if history[-2] > 0 else 0.0)

        point_forecast = np.array(point_forecast)

        # Uncertainty envelope
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
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "lookback_days": lb,
                "horizon_days": T,
            },
            metadata={
                "residual_std": round(residual_std, 4),
                "point_forecast_terminal": round(float(point_forecast[-1]), 4),
            },
        )

    @classmethod
    def param_schema(cls) -> dict:
        return {
            "n_estimators": (100, 50, 500, 50,
                "Number of boosting trees. More = slower but potentially better."),
            "max_depth": (3, 2, 8, 1,
                "Max tree depth. Deeper = more complex. Keep ≤ 5 to avoid overfit."),
            "lookback": (20, 5, 60, 5,
                "Past days used as lag features."),
            "horizon_days": (30, 5, 252, 5,
                "Trading days to forecast forward."),
            "seed": (42, 0, 9999, 1,
                "Random seed."),
        }
