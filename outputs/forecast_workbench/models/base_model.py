"""
Abstract base class for all models in the Model Zoo.

Every model must implement:
  - fit(preprocessor)  → self
  - predict()          → ModelResult
  - param_schema()     → dict of {name: (default, min, max, step, description)}
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ModelResult:
    """
    Standardised output from any Model Zoo model.

    Attributes:
        paths         : 2-D array of shape (n_paths, horizon+1).
                        Each row is one simulated price path.
                        First element of every row = S0 (current price).
        dates         : DatetimeIndex of length horizon+1 (includes today).
        S0            : Starting price.
        mu            : Annualised drift used by the model.
        sigma         : Annualised volatility used by the model.
        model_name    : Human-readable model name.
        params        : Dict of parameters actually used.
        percentiles   : Pre-computed percentile bands {pct: array}.
        metadata      : Anything extra the model wants to surface.
    """
    paths: np.ndarray
    dates: pd.DatetimeIndex
    S0: float
    mu: float
    sigma: float
    model_name: str
    params: dict[str, Any] = field(default_factory=dict)
    percentiles: dict[int, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.percentiles:
            for p in (5, 25, 50, 75, 95):
                self.percentiles[p] = np.percentile(self.paths, p, axis=0)

    @property
    def terminal_prices(self) -> np.ndarray:
        """Final price of every path."""
        return self.paths[:, -1]

    @property
    def expected_price(self) -> float:
        return float(np.mean(self.terminal_prices))

    @property
    def price_at_percentile(self) -> dict[int, float]:
        return {p: float(arr[-1]) for p, arr in self.percentiles.items()}


class BaseModel(ABC):
    """Shared interface for every Model Zoo entry."""

    name: str = "Base"

    @abstractmethod
    def fit(self, preprocessor: Any) -> "BaseModel":
        """Consume a fitted Preprocessor and cache parameters."""

    @abstractmethod
    def predict(self) -> ModelResult:
        """Run the simulation and return a ModelResult."""

    @classmethod
    @abstractmethod
    def param_schema(cls) -> dict[str, tuple]:
        """
        Return widget specs consumed by the Streamlit sidebar.

        Format:
            { "param_name": (default, min, max, step, "Tooltip text") }
        """
