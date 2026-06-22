"""Strategy registry — maps display names to strategy classes."""
from strategy.base import Signal, Strategy
from strategy.model_strategy import ModelForecastStrategy
from strategy.technical import RSIReversionStrategy, SMACrossStrategy

STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    "SMA Crossover (trend)": SMACrossStrategy,
    "RSI Mean-Reversion": RSIReversionStrategy,
    "Model Forecast (Model Zoo)": ModelForecastStrategy,
}

# Strategies that internally refit a Model-Zoo model (slow; need a model picker).
MODEL_DRIVEN = {"Model Forecast (Model Zoo)"}

__all__ = [
    "Signal",
    "Strategy",
    "SMACrossStrategy",
    "RSIReversionStrategy",
    "ModelForecastStrategy",
    "STRATEGY_REGISTRY",
    "MODEL_DRIVEN",
]
