"""
LSTM — Long Short-Term Memory Neural Network.

Best for: assets with complex non-linear temporal patterns that linear
          models miss. Captures long-range dependencies in price sequences.

Architecture:
    Input: sequence of `lookback` normalised closing prices → shape (lookback, 1)
    LSTM layer: `units` hidden cells with forget/input/output gates
    Dense output layer: 1 neuron → predicted next normalised price

Training:
    Loss: MSE on normalised prices
    Optimiser: Adam
    Epochs: configurable (more = better fit, slower)

Forecast: recursive T-step
    At each step, use the last `lookback` prices (including prior predictions)
    to predict the next price, then slide the window forward.

Uncertainty: residual std from training, applied as Gaussian noise across N paths.

IMPORTANT: LSTM requires tensorflow. On Streamlit Cloud this may be slow
on the first run as the model compiles. Subsequent runs are faster.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from models.base_model import BaseModel, ModelResult

try:
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.preprocessing import MinMaxScaler
    _TF_OK = True
    tf.get_logger().setLevel("ERROR")
except ImportError:
    _TF_OK = False

warnings.filterwarnings("ignore")

_N_PATHS = 100   # fewer paths: LSTM is slower


def _build_lstm(lookback: int, units: int) -> "keras.Model":
    model = keras.Sequential([
        keras.layers.Input(shape=(lookback, 1)),
        keras.layers.LSTM(units, return_sequences=False),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def _make_sequences(data: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback: i])
        y.append(data[i])
    return np.array(X)[..., np.newaxis], np.array(y)


class LSTMModel(BaseModel):
    name = "LSTM Neural Network"

    def __init__(
        self,
        lookback: int = 30,
        units: int = 50,
        epochs: int = 10,
        horizon_days: int = 30,
        seed: int = 42,
    ) -> None:
        self.lookback = lookback
        self.units = units
        self.epochs = epochs
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

    def fit(self, preprocessor) -> "LSTMModel":
        if not _TF_OK:
            raise RuntimeError(
                "tensorflow and scikit-learn are required for LSTM. "
                "Run: pip install tensorflow scikit-learn"
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
        tf.random.set_seed(self.seed)

        # Normalise
        scaler = MinMaxScaler()
        prices_sc = scaler.fit_transform(prices.reshape(-1, 1)).flatten()

        # Sequences
        X, y = _make_sequences(prices_sc, lb)
        if len(X) < 10:
            raise ValueError(
                f"Not enough data for LSTM with lookback={lb}. "
                "Reduce lookback or increase the historical window."
            )

        # Train
        model = _build_lstm(lb, self.units)
        model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=32,
            verbose=0,
            validation_split=0.1,
        )

        train_preds = model.predict(X, verbose=0).flatten()
        residual_std_sc = float(np.std(y - train_preds))

        # Recursive forecast in scaled space
        window = list(prices_sc[-lb:])
        point_forecast_sc = []
        for _ in range(T):
            inp = np.array(window[-lb:])[np.newaxis, :, np.newaxis]
            next_sc = float(model.predict(inp, verbose=0)[0, 0])
            point_forecast_sc.append(next_sc)
            window.append(next_sc)

        point_forecast_sc = np.array(point_forecast_sc)

        # Simulate paths in scaled space
        time_scale = np.sqrt(np.arange(1, T + 1))
        noise = rng.normal(0, 1, (_N_PATHS, T)) * residual_std_sc * time_scale[np.newaxis, :]
        paths_sc = point_forecast_sc[np.newaxis, :] + noise

        # Inverse transform each path to price space
        S0_sc = float(scaler.transform([[S0]])[0, 0])
        price_paths = scaler.inverse_transform(paths_sc.reshape(-1, 1)).reshape(_N_PATHS, T)
        price_paths = np.maximum(price_paths, S0 * 0.05)

        paths = np.hstack([np.full((_N_PATHS, 1), S0), price_paths])
        dates = pd.bdate_range(start=last_date, periods=T + 1)

        point_forecast_price = scaler.inverse_transform(
            point_forecast_sc.reshape(-1, 1)
        ).flatten()

        return ModelResult(
            paths=paths,
            dates=dates,
            S0=S0,
            mu=self._mu,
            sigma=self._sigma,
            model_name=self.name,
            params={
                "lookback": lb,
                "units": self.units,
                "epochs": self.epochs,
                "horizon_days": T,
            },
            metadata={
                "residual_std_scaled": round(residual_std_sc, 6),
                "point_forecast_terminal": round(float(point_forecast_price[-1]), 4),
            },
        )

    @classmethod
    def param_schema(cls) -> dict:
        return {
            "lookback": (30, 10, 90, 5,
                "Sequence length fed into the LSTM at each step."),
            "units": (50, 16, 128, 16,
                "LSTM hidden units. More units = more capacity but slower."),
            "epochs": (10, 5, 50, 5,
                "Training epochs. More = better fit, longer wait."),
            "horizon_days": (30, 5, 90, 5,
                "Trading days to forecast forward."),
            "seed": (42, 0, 9999, 1,
                "Random seed."),
        }
