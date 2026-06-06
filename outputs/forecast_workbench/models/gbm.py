"""
Geometric Brownian Motion — single path simulation.

dS = μ S dt + σ S dW

Discretised with Euler-Maruyama:
    S(t+dt) = S(t) * exp( (μ - σ²/2)*dt + σ*√dt*Z )
    Z ~ N(0,1)

The model handles its own data cleaning by accepting a Preprocessor
object and reading `.mu`, `.sigma`, `.S0` directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models.base_model import BaseModel, ModelResult


class GBMModel(BaseModel):
    name = "Geometric Brownian Motion (GBM)"

    def __init__(self, horizon_days: int = 30, seed: int | None = None) -> None:
        self.horizon_days = horizon_days
        self.seed = seed

        # Set after fit()
        self._mu: float = 0.0
        self._sigma: float = 0.0
        self._S0: float = 0.0
        self._last_date: pd.Timestamp | None = None

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def fit(self, preprocessor) -> "GBMModel":
        self._mu = preprocessor.mu
        self._sigma = preprocessor.sigma
        self._S0 = preprocessor.S0
        self._last_date = preprocessor.train.index[-1]
        # Live values for forward forecasting
        self._live_S0 = preprocessor.live_S0
        self._live_last_date = preprocessor.live_last_date
        return self

    def predict(self) -> ModelResult:
        T = self.horizon_days
        dt = 1 / 252  # daily steps, annualised

        rng = np.random.default_rng(self.seed)
        Z = rng.standard_normal(T)

        drift = (self._mu - 0.5 * self._sigma ** 2) * dt
        diffusion = self._sigma * np.sqrt(dt) * Z

        log_returns = drift + diffusion
        price_path = self._S0 * np.exp(np.concatenate([[0.0], np.cumsum(log_returns)]))

        # Shape (1, T+1) — single path wrapped in 2-D for ModelResult
        paths = price_path[np.newaxis, :]

        dates = pd.bdate_range(start=self._last_date, periods=T + 1)

        return ModelResult(
            paths=paths,
            dates=dates,
            S0=self._S0,
            mu=self._mu,
            sigma=self._sigma,
            model_name=self.name,
            params={
                "horizon_days": T,
                "annualised_mu": round(self._mu, 4),
                "annualised_sigma": round(self._sigma, 4),
                "seed": self.seed,
            },
        )

    def predict_forward(self) -> ModelResult:
        """Same as predict() but starts from today's price and date."""
        T = self.horizon_days
        dt = 1 / 252

        rng = np.random.default_rng(self.seed)
        Z = rng.standard_normal(T)

        drift = (self._mu - 0.5 * self._sigma ** 2) * dt
        diffusion = self._sigma * np.sqrt(dt) * Z

        log_returns = drift + diffusion
        price_path = self._live_S0 * np.exp(np.concatenate([[0.0], np.cumsum(log_returns)]))

        paths = price_path[np.newaxis, :]
        dates = pd.bdate_range(start=self._live_last_date, periods=T + 1)

        return ModelResult(
            paths=paths,
            dates=dates,
            S0=self._live_S0,
            mu=self._mu,
            sigma=self._sigma,
            model_name=self.name,
            params={
                "horizon_days": T,
                "annualised_mu": round(self._mu, 4),
                "annualised_sigma": round(self._sigma, 4),
                "seed": self.seed,
            },
        )

    @classmethod
    def param_schema(cls) -> dict:
        return {
            "horizon_days": (
                30, 5, 252, 5,
                "Number of trading days to simulate forward."
            ),
            "seed": (
                42, 0, 9999, 1,
                "Random seed for reproducibility (0 = random each run)."
            ),
        }
