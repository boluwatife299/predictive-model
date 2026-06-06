"""
GARCH — Generalised AutoRegressive Conditional Heteroskedasticity.

Directly fixes GBM's biggest flaw: GBM assumes constant volatility.
GARCH lets volatility cluster — high-vol days tend to follow high-vol days
(as seen in every real market).

Mathematics (GARCH(p,q)):
    r_t = σ_t · ε_t,     ε_t ~ N(0,1)

    σ_t² = ω  +  Σ_{i=1}^{p} α_i · r_{t-i}²   (ARCH terms)
              +  Σ_{j=1}^{q} β_j · σ_{t-j}²    (GARCH terms)

    Persistence: α + β < 1 for stationarity.
    Half-life of vol shock: log(0.5) / log(α + β) days.

Forecast workflow:
    1. Fit GARCH on historical log-returns.
    2. Extract T-step ahead variance forecast: σ̂₁², σ̂₂², ..., σ̂_T².
    3. Simulate N GBM paths using time-varying σ_t = √(σ̂_t²) at each step.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from models.base_model import BaseModel, ModelResult

try:
    from arch import arch_model as _arch_model
    _ARCH_OK = True
except ImportError:
    _ARCH_OK = False

warnings.filterwarnings("ignore")


class GARCHModel(BaseModel):
    name = "GARCH"

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        horizon_days: int = 30,
        num_paths: int = 500,
        seed: int = 42,
    ) -> None:
        self.p = p
        self.q = q
        self.horizon_days = horizon_days
        self.num_paths = num_paths
        self.seed = seed

        self._mu: float = 0.0
        self._sigma: float = 0.0
        self._S0: float = 0.0
        self._last_date: pd.Timestamp | None = None
        self._live_S0: float = 0.0
        self._live_last_date: pd.Timestamp | None = None
        self._train_returns: pd.Series | None = None
        self._full_returns: pd.Series | None = None

    # ------------------------------------------------------------------

    def fit(self, preprocessor) -> "GARCHModel":
        if not _ARCH_OK:
            raise RuntimeError(
                "arch is required for GARCH. Run: pip install arch"
            )
        self._mu = preprocessor.mu
        self._sigma = preprocessor.sigma
        self._S0 = preprocessor.S0
        self._last_date = preprocessor.train.index[-1]
        self._live_S0 = preprocessor.live_S0
        self._live_last_date = preprocessor.live_last_date

        train_close = preprocessor.train["Close"]
        full_close  = preprocessor.clean["Close"]
        # arch expects percentage log-returns
        self._train_returns = (np.log(train_close / train_close.shift(1)).dropna() * 100)
        self._full_returns  = (np.log(full_close  / full_close.shift(1)).dropna() * 100)
        return self

    def predict(self) -> ModelResult:
        return self._run(self._train_returns, self._S0, self._last_date)

    def predict_forward(self) -> ModelResult:
        return self._run(self._full_returns, self._live_S0, self._live_last_date)

    # ------------------------------------------------------------------

    def _run(
        self,
        returns_pct: pd.Series,
        S0: float,
        last_date: pd.Timestamp,
    ) -> ModelResult:
        T = self.horizon_days
        N = self.num_paths
        dt = 1 / 252
        rng = np.random.default_rng(self.seed)

        # Fit GARCH
        try:
            am = _arch_model(
                returns_pct,
                vol="Garch",
                p=self.p,
                q=self.q,
                dist="normal",
                mean="Constant",
            )
            res = am.fit(disp="off", show_warning=False)
            # T-step ahead variance forecast (percentage²)
            fc = res.forecast(horizon=T, reindex=False)
            var_forecast_pct2 = fc.variance.values[-1]   # shape (T,)
            # Convert from percentage² to daily sigma
            daily_sigma_t = np.sqrt(var_forecast_pct2) / 100.0  # fraction
            garch_params = {
                "omega": round(float(res.params.get("omega", 0)), 6),
                "alpha[1]": round(float(res.params.get("alpha[1]", 0)), 4),
                "beta[1]":  round(float(res.params.get("beta[1]", 0)), 4),
            }
            persistence = garch_params.get("alpha[1]", 0) + garch_params.get("beta[1]", 0)
        except Exception:
            # Fallback: constant vol from historical sigma
            daily_sigma_t = np.full(T, self._sigma / np.sqrt(252))
            garch_params = {}
            persistence = float("nan")

        # Simulate N paths with time-varying σ_t
        mu_daily = self._mu / 252
        paths = np.zeros((N, T + 1))
        paths[:, 0] = S0

        for t in range(T):
            sig_t = daily_sigma_t[t]
            Z = rng.standard_normal(N)
            log_ret = (mu_daily - 0.5 * sig_t ** 2) + sig_t * Z
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
                "p": self.p,
                "q": self.q,
                "horizon_days": T,
                "num_paths": N,
                **garch_params,
            },
            metadata={
                "persistence_alpha_plus_beta": round(persistence, 4)
                    if not np.isnan(persistence) else "N/A",
                "avg_forecast_annualised_vol_pct": round(
                    float(np.mean(daily_sigma_t)) * np.sqrt(252) * 100, 2
                ),
                "VaR_5pct_terminal": round(var_5, 4),
            },
        )

    @classmethod
    def param_schema(cls) -> dict:
        return {
            "p": (1, 1, 3, 1,
                "ARCH order — lags of squared returns. Usually 1."),
            "q": (1, 1, 3, 1,
                "GARCH order — lags of past variance. Usually 1."),
            "horizon_days": (30, 5, 252, 5,
                "Trading days to forecast forward."),
            "num_paths": (500, 50, 2000, 50,
                "Number of GBM paths using GARCH-estimated vol."),
            "seed": (42, 0, 9999, 1,
                "Random seed."),
        }
