"""
Monte Carlo Simulation — probabilistic multi-path distribution.

Uses the same GBM discretisation as the single-path model but runs
N independent paths to build a full price distribution.

Output surfaces:
  - All N raw paths
  - Percentile fan (5th / 25th / 50th / 75th / 95th)
  - Expected price (mean of terminal distribution)
  - Value at Risk proxy (5th percentile terminal price)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models.base_model import BaseModel, ModelResult


class MonteCarloModel(BaseModel):
    name = "Monte Carlo Simulation"

    def __init__(
        self,
        horizon_days: int = 30,
        num_paths: int = 500,
        seed: int | None = None,
    ) -> None:
        self.horizon_days = horizon_days
        self.num_paths = num_paths
        self.seed = seed

        self._mu: float = 0.0
        self._sigma: float = 0.0
        self._S0: float = 0.0
        self._last_date: pd.Timestamp | None = None

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def fit(self, preprocessor) -> "MonteCarloModel":
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
        N = self.num_paths
        dt = 1 / 252

        rng = np.random.default_rng(self.seed)
        # Shape: (N, T)
        Z = rng.standard_normal((N, T))

        drift = (self._mu - 0.5 * self._sigma ** 2) * dt
        diffusion = self._sigma * np.sqrt(dt) * Z

        log_returns = drift + diffusion          # (N, T)
        cum_log_ret = np.cumsum(log_returns, axis=1)  # (N, T)

        # Prepend t=0 column (all paths start at S0)
        paths = self._S0 * np.exp(
            np.hstack([np.zeros((N, 1)), cum_log_ret])
        )  # (N, T+1)

        dates = pd.bdate_range(start=self._last_date, periods=T + 1)

        terminal = paths[:, -1]
        var_5 = float(np.percentile(terminal, 5))
        cvar_5 = float(terminal[terminal <= var_5].mean())

        return ModelResult(
            paths=paths,
            dates=dates,
            S0=self._S0,
            mu=self._mu,
            sigma=self._sigma,
            model_name=self.name,
            params={
                "horizon_days": T,
                "num_paths": N,
                "annualised_mu": round(self._mu, 4),
                "annualised_sigma": round(self._sigma, 4),
                "seed": self.seed,
            },
            metadata={
                "VaR_5pct_terminal": round(var_5, 4),
                "CVaR_5pct_terminal": round(cvar_5, 4),
                "expected_terminal": round(float(np.mean(terminal)), 4),
            },
        )

    def predict_forward(self) -> ModelResult:
        """Same as predict() but starts from today's price and date."""
        T = self.horizon_days
        N = self.num_paths
        dt = 1 / 252

        rng = np.random.default_rng(self.seed)
        Z = rng.standard_normal((N, T))

        drift = (self._mu - 0.5 * self._sigma ** 2) * dt
        diffusion = self._sigma * np.sqrt(dt) * Z

        log_returns = drift + diffusion
        cum_log_ret = np.cumsum(log_returns, axis=1)

        paths = self._live_S0 * np.exp(
            np.hstack([np.zeros((N, 1)), cum_log_ret])
        )

        dates = pd.bdate_range(start=self._live_last_date, periods=T + 1)

        terminal = paths[:, -1]
        var_5 = float(np.percentile(terminal, 5))
        cvar_5 = float(terminal[terminal <= var_5].mean())

        return ModelResult(
            paths=paths,
            dates=dates,
            S0=self._live_S0,
            mu=self._mu,
            sigma=self._sigma,
            model_name=self.name,
            params={
                "horizon_days": T,
                "num_paths": N,
                "annualised_mu": round(self._mu, 4),
                "annualised_sigma": round(self._sigma, 4),
                "seed": self.seed,
            },
            metadata={
                "VaR_5pct_terminal": round(var_5, 4),
                "CVaR_5pct_terminal": round(cvar_5, 4),
                "expected_terminal": round(float(np.mean(terminal)), 4),
            },
        )

    @classmethod
    def param_schema(cls) -> dict:
        return {
            "horizon_days": (
                30, 5, 252, 5,
                "Number of trading days to simulate forward."
            ),
            "num_paths": (
                500, 50, 5000, 50,
                "Number of simulation paths. More paths = smoother distribution."
            ),
            "seed": (
                42, 0, 9999, 1,
                "Random seed (0 = new random each run)."
            ),
        }
