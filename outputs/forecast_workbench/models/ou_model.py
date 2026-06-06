"""
Ornstein-Uhlenbeck Mean Reversion Model.

Best for: interest rates, VIX, commodity spreads — assets that don't
          trend indefinitely but pull back toward a long-run average.

Mathematics:
    dX = θ(μ - X)dt + σ dW

    θ  : speed of mean reversion (how fast price snaps back)
    μ  : long-run equilibrium price
    σ  : diffusion (volatility)
    dW : Wiener process increment

Discrete Euler-Maruyama step:
    X_{t+1} = X_t + θ(μ - X_t)Δt + σ√Δt · Z,   Z ~ N(0,1)

Parameter estimation (OLS on the AR(1) discretisation):
    ΔX_t = a + b·X_t + ε_t
    θ = -b/Δt,   μ = a / (θ·Δt),   σ = std(ε) / √Δt
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models.base_model import BaseModel, ModelResult


class OUModel(BaseModel):
    name = "Ornstein-Uhlenbeck (Mean Reversion)"

    def __init__(
        self,
        horizon_days: int = 30,
        num_paths: int = 500,
        seed: int = 42,
    ) -> None:
        self.horizon_days = horizon_days
        self.num_paths = num_paths
        self.seed = seed

        self._theta: float = 0.0
        self._mu_ou: float = 0.0
        self._sigma_ou: float = 0.0
        self._mu_hist: float = 0.0
        self._sigma_hist: float = 0.0
        self._S0: float = 0.0
        self._last_date: pd.Timestamp | None = None
        self._live_S0: float = 0.0
        self._live_last_date: pd.Timestamp | None = None

    # ------------------------------------------------------------------

    def fit(self, preprocessor) -> "OUModel":
        close = preprocessor.train["Close"]
        self._mu_hist = preprocessor.mu
        self._sigma_hist = preprocessor.sigma
        self._S0 = preprocessor.S0
        self._last_date = preprocessor.train.index[-1]
        self._live_S0 = preprocessor.live_S0
        self._live_last_date = preprocessor.live_last_date

        self._theta, self._mu_ou, self._sigma_ou = self._estimate_ou(close)
        return self

    def predict(self) -> ModelResult:
        return self._run(self._S0, self._last_date)

    def predict_forward(self) -> ModelResult:
        return self._run(self._live_S0, self._live_last_date)

    # ------------------------------------------------------------------

    def _run(self, S0: float, last_date: pd.Timestamp) -> ModelResult:
        T = self.horizon_days
        dt = 1 / 252
        rng = np.random.default_rng(self.seed)
        N = self.num_paths

        paths = np.zeros((N, T + 1))
        paths[:, 0] = S0

        for t in range(T):
            X = paths[:, t]
            Z = rng.standard_normal(N)
            drift = self._theta * (self._mu_ou - X) * dt
            diffusion = self._sigma_ou * np.sqrt(dt) * Z
            paths[:, t + 1] = np.maximum(X + drift + diffusion, 1e-6)

        dates = pd.bdate_range(start=last_date, periods=T + 1)

        terminal = paths[:, -1]
        var_5 = float(np.percentile(terminal, 5))

        return ModelResult(
            paths=paths,
            dates=dates,
            S0=S0,
            mu=self._mu_hist,
            sigma=self._sigma_hist,
            model_name=self.name,
            params={
                "horizon_days": T,
                "num_paths": N,
                "theta_reversion_speed": round(self._theta, 4),
                "mu_long_run_mean": round(self._mu_ou, 4),
                "sigma_diffusion": round(self._sigma_ou, 4),
            },
            metadata={
                "long_run_mean": round(self._mu_ou, 4),
                "reversion_speed_theta": round(self._theta, 4),
                "half_life_days": round(np.log(2) / self._theta * 252, 1)
                    if self._theta > 0 else float("inf"),
                "VaR_5pct_terminal": round(var_5, 4),
            },
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_ou(close: pd.Series) -> tuple[float, float, float]:
        """OLS on ΔX = a + b·X + ε to recover θ, μ, σ."""
        dt = 1 / 252
        X = close.values
        dX = np.diff(X)
        X_lag = X[:-1]

        # OLS
        A = np.column_stack([np.ones_like(X_lag), X_lag])
        try:
            coeffs, residuals, _, _ = np.linalg.lstsq(A, dX, rcond=None)
        except Exception:
            return 1.0, float(np.mean(X)), float(np.std(dX))

        a, b = coeffs
        theta = max(-b / dt, 1e-4)          # reversion speed (positive)
        mu_ou = a / (theta * dt)            # long-run mean
        resid = dX - (a + b * X_lag)
        sigma_ou = float(np.std(resid) / np.sqrt(dt))
        return theta, float(mu_ou), sigma_ou

    @classmethod
    def param_schema(cls) -> dict:
        return {
            "horizon_days": (30, 5, 252, 5,
                "Trading days to simulate forward."),
            "num_paths": (500, 50, 2000, 50,
                "Number of simulation paths."),
            "seed": (42, 0, 9999, 1,
                "Random seed (0 = new each run)."),
        }
