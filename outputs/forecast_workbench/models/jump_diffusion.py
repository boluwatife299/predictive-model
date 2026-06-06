"""
Merton Jump Diffusion Model.

Best for: crypto, single stocks around earnings / news events —
          assets where sudden large moves (jumps) occur on top of
          continuous diffusion.

Mathematics:
    dS = (μ - λk̄)S dt + σS dW + S dJ

    μ     : drift
    σ     : diffusion volatility
    λ     : jump intensity (expected jumps per year)
    dJ    : compound Poisson process; jump sizes log-normal N(μ_j, σ_j²)
    k̄     : E[e^J - 1] = exp(μ_j + 0.5*σ_j²) - 1  (compensator)

Discretised step:
    ln(S_{t+1}/S_t) = (μ - λk̄ - 0.5σ²)dt + σ√dt·Z + Σ_{i=1}^{N_t} Y_i
    N_t ~ Poisson(λ·dt),   Y_i ~ N(μ_j, σ_j²)

Jump parameter estimation:
    Identify days with |log_return| > jump_threshold * σ as "jump days".
    λ = count(jump_days) / years_of_data
    μ_j, σ_j = mean, std of those jump log-returns
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models.base_model import BaseModel, ModelResult


class JumpDiffusionModel(BaseModel):
    name = "Merton Jump Diffusion"

    def __init__(
        self,
        horizon_days: int = 30,
        num_paths: int = 500,
        jump_threshold: float = 2.5,
        seed: int = 42,
    ) -> None:
        self.horizon_days = horizon_days
        self.num_paths = num_paths
        self.jump_threshold = jump_threshold
        self.seed = seed

        self._mu: float = 0.0
        self._sigma: float = 0.0
        self._lambda: float = 0.0
        self._mu_j: float = 0.0
        self._sigma_j: float = 0.0
        self._S0: float = 0.0
        self._last_date: pd.Timestamp | None = None
        self._live_S0: float = 0.0
        self._live_last_date: pd.Timestamp | None = None

    # ------------------------------------------------------------------

    def fit(self, preprocessor) -> "JumpDiffusionModel":
        close = preprocessor.train["Close"]
        log_ret = np.log(close / close.shift(1)).dropna()

        self._mu = preprocessor.mu
        self._sigma = preprocessor.sigma
        self._S0 = preprocessor.S0
        self._last_date = preprocessor.train.index[-1]
        self._live_S0 = preprocessor.live_S0
        self._live_last_date = preprocessor.live_last_date

        self._lambda, self._mu_j, self._sigma_j = self._estimate_jumps(
            log_ret, self._sigma, self.jump_threshold
        )
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

        lam = self._lambda
        mu_j = self._mu_j
        sig_j = self._sigma_j
        # Compensator: E[e^Y - 1]
        k_bar = np.exp(mu_j + 0.5 * sig_j ** 2) - 1
        adj_mu = self._mu - lam * k_bar

        paths = np.zeros((N, T + 1))
        paths[:, 0] = S0

        for t in range(T):
            Z = rng.standard_normal(N)
            # Poisson jumps this step
            n_jumps = rng.poisson(lam * dt, N)
            jump_sum = np.array([
                rng.normal(mu_j, sig_j, max(n, 1)).sum() if n > 0 else 0.0
                for n in n_jumps
            ])
            log_ret = ((adj_mu - 0.5 * self._sigma ** 2) * dt
                       + self._sigma * np.sqrt(dt) * Z
                       + jump_sum)
            paths[:, t + 1] = paths[:, t] * np.exp(log_ret)

        dates = pd.bdate_range(start=last_date, periods=T + 1)
        terminal = paths[:, -1]
        var_5 = float(np.percentile(terminal, 5))

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
                "jump_threshold_sigma": self.jump_threshold,
                "annualised_mu": round(self._mu, 4),
                "annualised_sigma": round(self._sigma, 4),
                "lambda_jumps_per_year": round(lam, 4),
                "mu_j_avg_jump": round(mu_j, 4),
                "sigma_j_jump_std": round(sig_j, 4),
            },
            metadata={
                "expected_jumps_per_year": round(lam, 2),
                "avg_jump_size_pct": round((np.exp(mu_j) - 1) * 100, 2),
                "VaR_5pct_terminal": round(var_5, 4),
            },
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_jumps(
        log_ret: pd.Series,
        sigma: float,
        threshold: float,
    ) -> tuple[float, float, float]:
        daily_sig = sigma / np.sqrt(252)
        jump_mask = log_ret.abs() > threshold * daily_sig
        jump_returns = log_ret[jump_mask]

        years = len(log_ret) / 252
        lam = float(len(jump_returns) / max(years, 0.01))
        mu_j = float(jump_returns.mean()) if len(jump_returns) > 0 else 0.0
        sig_j = float(jump_returns.std()) if len(jump_returns) > 1 else daily_sig
        return lam, mu_j, sig_j

    @classmethod
    def param_schema(cls) -> dict:
        return {
            "horizon_days": (30, 5, 252, 5,
                "Trading days to simulate forward."),
            "num_paths": (500, 50, 2000, 50,
                "Number of simulation paths."),
            "jump_threshold": (2, 1, 5, 1,
                "Multiples of daily σ used to classify a return as a jump. "
                "Lower = more jumps detected."),
            "seed": (42, 0, 9999, 1,
                "Random seed (0 = new each run)."),
        }
