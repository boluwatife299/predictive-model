"""
Strategy layer — turns market state (and optionally a model forecast) into a
trading *decision*.

A forecast on its own is inert: a price cone you cannot act on. A Strategy is the
rule that converts that view into a position — go long / short / flat, and where
to place the protective stop and profit target. The backtest engine then scores
the strategy across history (walk-forward, no lookahead), so you measure a
*tested edge* rather than a forecast's prettiness.

Contract
--------
1. ``prepare(df)`` is called once with the full clean OHLCV frame. Precompute any
   indicators or forecasts here — but only in a *causal* way (a value at bar t may
   use bars ≤ t only). Rolling means, EWMAs and RSI are causal; a forward fill of
   a future value is not.
2. ``signal(i, position)`` returns the desired position *as of* bar ``i`` given the
   strategy is currently holding ``position`` (so a strategy can express "hold").
   It must read nothing beyond bar ``i``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class Signal:
    """A desired position plus optional risk levels, emitted for one bar."""

    direction: int = 0            # +1 long, -1 short, 0 flat / exit
    stop: float | None = None     # absolute price for the protective stop
    target: float | None = None   # absolute price for the profit target
    reason: str = ""              # human-readable rationale (shown in replay)


class Strategy(ABC):
    """Shared interface for every strategy."""

    name: str = "Strategy"
    description: str = ""
    warmup: int = 50              # bars of history needed before the first signal

    # ------------------------------------------------------------------
    def prepare(self, df: pd.DataFrame) -> None:
        """Cache the frame and precompute indicators. Override as needed."""
        self.df = df

    @abstractmethod
    def signal(self, i: int, position: int) -> Signal:
        """Desired position as of bar ``i`` (no lookahead beyond ``i``)."""

    @classmethod
    def param_schema(cls) -> dict[str, tuple]:
        """Sidebar widget specs: ``{name: (default, min, max, step, tooltip)}``."""
        return {}
