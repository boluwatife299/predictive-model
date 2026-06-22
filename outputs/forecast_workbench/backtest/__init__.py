"""Backtest package — walk-forward engine, metrics, and trade-sequence MC."""
from backtest.engine import BacktestResult, Trade, run_backtest
from backtest.metrics import compute_metrics, metrics_table
from backtest.montecarlo import MonteCarloResult, trade_sequence_mc

__all__ = [
    "run_backtest",
    "BacktestResult",
    "Trade",
    "compute_metrics",
    "metrics_table",
    "trade_sequence_mc",
    "MonteCarloResult",
]
