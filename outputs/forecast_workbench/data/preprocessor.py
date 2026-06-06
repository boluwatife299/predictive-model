"""
Preprocessor — cleaning and feature extraction.

Responsibilities:
  - Forward-fill / back-fill gaps (weekends, holidays)
  - Compute log returns
  - Estimate annualised drift (μ) and volatility (σ)
  - Split into training window and out-of-sample validation window
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class Preprocessor:
    """
    Stateful preprocessor: call `fit(df)` once, then access
    `.clean`, `.log_returns`, `.mu`, `.sigma`, `.S0`.
    """

    TRADING_DAYS = 252

    def __init__(self, validation_days: int = 30) -> None:
        self.validation_days = validation_days

        self.clean: pd.DataFrame = pd.DataFrame()
        self.log_returns: pd.Series = pd.Series(dtype=float)
        self.mu: float = 0.0
        self.sigma: float = 0.0
        self.S0: float = 0.0

        self._train_df: pd.DataFrame = pd.DataFrame()
        self._val_df: pd.DataFrame = pd.DataFrame()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        """
        Clean the raw OHLCV frame and estimate drift / volatility
        from the *training* portion (everything before the last
        `validation_days` bars).
        """
        self.clean = self._clean(df)

        # Split train / validation
        if len(self.clean) <= self.validation_days:
            self._train_df = self.clean.copy()
            self._val_df = pd.DataFrame()
        else:
            self._train_df = self.clean.iloc[: -self.validation_days]
            self._val_df = self.clean.iloc[-self.validation_days :]

        self.log_returns = self._log_returns(self._train_df["Close"])
        self.mu, self.sigma = self._estimate_params(self.log_returns)
        # S0 / last_date for validation (end of training split)
        self.S0 = float(self._train_df["Close"].iloc[-1])
        # live_S0 / live_last_date for forward forecasting (end of full dataset = today)
        self.live_S0 = float(self.clean["Close"].iloc[-1])
        self.live_last_date = self.clean.index[-1]
        return self

    @property
    def train(self) -> pd.DataFrame:
        return self._train_df

    @property
    def validation(self) -> pd.DataFrame:
        return self._val_df

    @property
    def has_validation(self) -> bool:
        return not self._val_df.empty

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out = out[~out.index.duplicated(keep="first")]
        out.sort_index(inplace=True)
        # Forward-fill then back-fill so no NaN remain
        out.ffill(inplace=True)
        out.bfill(inplace=True)
        # Drop any remaining all-NaN rows
        out.dropna(how="all", inplace=True)
        return out

    @staticmethod
    def _log_returns(close: pd.Series) -> pd.Series:
        return np.log(close / close.shift(1)).dropna()

    def _estimate_params(self, log_ret: pd.Series) -> tuple[float, float]:
        """
        Annualised drift (μ) and volatility (σ) from daily log returns.
        mu  = mean(log_ret) * T  +  0.5 * sigma^2          (GBM drift)
        sigma = std(log_ret) * sqrt(T)
        """
        T = self.TRADING_DAYS
        daily_sigma = float(log_ret.std())
        daily_mu = float(log_ret.mean())

        sigma = daily_sigma * np.sqrt(T)
        # Ito's correction: annualised drift of the *price process*
        mu = daily_mu * T + 0.5 * sigma ** 2
        return mu, sigma
