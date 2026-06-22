"""
Technical strategies — rule-based, fast, no model refit.

These read only price/indicator state and are cheap enough to backtest over
thousands of bars. They are the honest baselines: if a model-driven strategy
can't beat a simple moving-average cross, the model isn't adding edge.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.base import Signal, Strategy


class SMACrossStrategy(Strategy):
    """
    Trend following. Long when the fast SMA is above the slow SMA; short (or flat
    if shorting is disabled) when it is below. Exits happen naturally on the cross.
    """

    name = "SMA Crossover (trend)"
    description = (
        "Long when the fast moving average is above the slow one, flat/short "
        "below. A classic trend-following baseline."
    )

    def __init__(self, fast: int = 20, slow: int = 50) -> None:
        self.fast = int(fast)
        self.slow = int(slow)
        self.warmup = max(self.fast, self.slow) + 1

    def prepare(self, df: pd.DataFrame) -> None:
        self.df = df
        close = df["Close"]
        self._fast = close.rolling(self.fast).mean().to_numpy()
        self._slow = close.rolling(self.slow).mean().to_numpy()

    def signal(self, i: int, position: int) -> Signal:
        f, s = self._fast[i], self._slow[i]
        if np.isnan(f) or np.isnan(s):
            return Signal(0, reason="warming up")
        if f > s:
            return Signal(+1, reason=f"fast SMA {f:.2f} > slow SMA {s:.2f}")
        return Signal(-1, reason=f"fast SMA {f:.2f} < slow SMA {s:.2f}")

    @classmethod
    def param_schema(cls) -> dict[str, tuple]:
        return {
            "fast": (20, 5, 100, 1, "Fast moving-average window (days)."),
            "slow": (50, 10, 250, 5, "Slow moving-average window (days)."),
        }


class RSIReversionStrategy(Strategy):
    """
    Mean reversion. Buy oversold (RSI below ``low``), sell/short overbought (RSI
    above ``high``), and flatten once RSI returns through the neutral ``exit``
    level. Hysteresis (enter at the extreme, exit at neutral) avoids whipsaw.
    """

    name = "RSI Mean-Reversion"
    description = (
        "Buy oversold, sell overbought, exit back at neutral RSI. A counter-trend "
        "baseline — pairs well with the Ornstein-Uhlenbeck view of the world."
    )

    def __init__(
        self, period: int = 14, low: int = 30, high: int = 70, exit: int = 50
    ) -> None:
        self.period = int(period)
        self.low = float(low)
        self.high = float(high)
        self.exit = float(exit)
        self.warmup = self.period + 1

    def prepare(self, df: pd.DataFrame) -> None:
        self.df = df
        close = df["Close"]
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(self.period).mean()
        loss = (-delta.clip(upper=0)).rolling(self.period).mean()
        rs = gain / loss.replace(0, np.nan)
        self._rsi = (100 - 100 / (1 + rs)).to_numpy()

    def signal(self, i: int, position: int) -> Signal:
        rsi = self._rsi[i]
        if np.isnan(rsi):
            return Signal(0, reason="warming up")
        if position > 0:
            if rsi >= self.exit:
                return Signal(0, reason=f"RSI {rsi:.0f} back to neutral — exit long")
            return Signal(+1, reason=f"RSI {rsi:.0f} — hold long")
        if position < 0:
            if rsi <= self.exit:
                return Signal(0, reason=f"RSI {rsi:.0f} back to neutral — exit short")
            return Signal(-1, reason=f"RSI {rsi:.0f} — hold short")
        # Flat: look for a fresh extreme.
        if rsi <= self.low:
            return Signal(+1, reason=f"RSI {rsi:.0f} oversold — go long")
        if rsi >= self.high:
            return Signal(-1, reason=f"RSI {rsi:.0f} overbought — go short")
        return Signal(0, reason=f"RSI {rsi:.0f} — no edge")

    @classmethod
    def param_schema(cls) -> dict[str, tuple]:
        return {
            "period": (14, 2, 50, 1, "RSI lookback window (days)."),
            "low": (30, 5, 45, 1, "Oversold threshold — go long below this."),
            "high": (70, 55, 95, 1, "Overbought threshold — go short above this."),
            "exit": (50, 40, 60, 1, "Neutral level — flatten when RSI returns here."),
        }
