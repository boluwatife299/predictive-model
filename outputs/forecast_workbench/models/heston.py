"""
Heston Stochastic Volatility Model.

Used by options desks for pricing derivatives.  Unlike GBM (constant σ),
Heston lets volatility itself follow a mean-reverting stochastic process.

Mathematics:
    dS = μ S dt + √V · S · dW₁
    dV = κ(θ - V)dt + ξ√V · dW₂
    Corr(dW₁, dW₂) = ρ

    κ  : mean-reversion speed of variance
    θ  : long-run variance (target vol² = θ)
    ξ  : vol of vol (how much variance itself fluctuates)
    ρ  : correlation between price and vol shocks
         (typically negative for equities: price drops → vol spikes)
    V₀ : initial variance

Feller condition for non-negative variance: 2κθ > ξ²

Discretised (Euler-Maruyama):
    V_{t+1} = max(V_t + κ(θ-V_t)dt + ξ√V_t·√dt·Z₂, 0)
    S_{t+1} = S_t · exp((μ - 0.5·V_t)dt + √V_t·√dt·Z₁)
    Z₁,Z₂ correlated via Cholesky: Z₂ = ρ·Z₁ + √(1-ρ²)·Z_ind
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models.base_model import BaseModel, ModelResult


class HestonModel(BaseModel):
    name = "Heston Stochastic Volatility"

    def __init__(
        self,
        horizon_days: int = 30,
        num_paths: int = 500,
        kappa: float = 2,
        xi: float = 50,        # stored as pct for slider, divided by 100 inside
        rho: int = -70,        # stored as int pct for slider, divided by 100 inside
        seed: int = 42,
    ) -> None:
        self.horizon_days = horizon_days
        self.num_paths = num_paths
        self.kappa = kappa
        self.xi = xi / 100.0
        self.rho = rho / 100.0
        self.seed = seed

        self._mu: float = 0.0
        self._sigma: float = 0.0
        self._theta: float = 0.0    # long-run variance = sigma²
        self._v0: float = 0.0       # initial variance
        self._S0: float = 0.0
        self._last_date: pd.Timestamp | None = None
        self._live_S0: float = 0.0
        self._live_last_date: pd.Timestamp | None = None

    # ------------------------------------------------------------------

    def fit(self, preprocessor) -> "HestonModel":
        self._mu = preprocessor.mu
        self._sigma = preprocessor.sigma
        self._theta = self._sigma ** 2      # long-run variance
        self._v0 = self._sigma ** 2         # start at historical variance
        self._S0 = preprocessor.S0
        self._last_date = preprocessor.train.index[-1]
        self._live_S0 = preprocessor.live_S0
        self._live_last_date = preprocessor.live_last_date
        return self

    def predict(self) -> ModelResult:
        return self._run(self._S0, self._last_date)

    def predict_forward(self) -> ModelResult:
        return self._run(self._live_S0, self._live_last_date)

    # ------------------------------------------------------------------

    def _run(self, S0: float, last_date: pd.Timestamp) -> ModelResult:
        T = self.horizon_days
        N = self.num_paths
        dt = 1 / 252
        rng = np.random.default_rng(self.seed)

        kappa = self.kappa
        theta = self._theta
        xi = self.xi
        rho = self.rho
        sqrt_1_rho2 = np.sqrt(max(1 - rho ** 2, 1e-8))

        S = np.full(N, S0, dtype=float)
        V = np.full(N, self._v0, dtype=float)

        paths = np.zeros((N, T + 1))
        paths[:, 0] = S0

        for t in range(T):
            Z1 = rng.standard_normal(N)
            Z_ind = rng.standard_normal(N)
            Z2 = rho * Z1 + sqrt_1_rho2 * Z_ind

            sqrt_V = np.sqrt(np.maximum(V, 0))
            # Price step
            S = S * np.exp(
                (self._mu - 0.5 * V) * dt + sqrt_V * np.sqrt(dt) * Z1
            )
            # Variance step (full-truncation: absorbing boundary at 0)
            V = np.maximum(
                V + kappa * (theta - V) * dt + xi * sqrt_V * np.sqrt(dt) * Z2,
                0.0,
            )
            paths[:, t + 1] = S

        dates = pd.bdate_range(start=last_date, periods=T + 1)
        terminal = paths[:, -1]
        var_5 = float(np.percentile(terminal, 5))

        # Check Feller condition
        feller_ok = 2 * kappa * theta > xi ** 2

        return ModelResult(
            paths=paths,
            dates=dates,
            S0=S0,
            mu=self._mu,
            sigma=self._sigma,
            model_name=self.name,
            params={
                "horizon_days": T,
                "num_paths": N,
                "kappa": kappa,
                "theta_long_run_var": round(theta, 6),
                "xi_vol_of_vol": xi,
                "rho_price_vol_corr": rho,
            },
            metadata={
                "long_run_vol_pct": round(np.sqrt(theta) * 100, 2),
                "feller_condition_satisfied": feller_ok,
                "VaR_5pct_terminal": round(var_5, 4),
            },
        )

    @classmethod
    def param_schema(cls) -> dict:
        return {
            "horizon_days": (30, 5, 252, 5,
                "Trading days to simulate forward."),
            "num_paths": (500, 50, 2000, 50,
                "Number of simulation paths."),
            "kappa": (2, 1, 10, 1,
                "Speed of variance mean reversion. Higher = faster snap-back."),
            "xi": (50, 10, 200, 10,
                "Vol-of-vol (%). How much variance itself fluctuates."),
            "rho": (-70, -99, 0, 1,
                "Price-vol correlation (%). Negative = vol rises when price falls (typical equities)."),
            "seed": (42, 0, 9999, 1,
                "Random seed."),
        }
