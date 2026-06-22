"""
Trade-sequence Monte Carlo — is the edge real, or did you get lucky?

A single backtest is one realised ordering of trades. Reshuffle or resample that
same set of trades thousands of times and you get a *distribution* of outcomes:
how good the lucky runs look, how bad the unlucky ones get, and — most usefully —
the drawdown you should actually brace for. This is the decision-grade Monte Carlo
(distinct from the price-path MC in the Forecast tab, which simulates prices, not
a strategy's P&L).

Two modes:
- ``bootstrap`` resamples trades *with replacement* (asks: given this edge, what
  range of futures is plausible over N trades?).
- ``shuffle`` reorders the *same* trades (asks: how much of my result was just the
  lucky sequence — e.g. wins clustering early?).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MonteCarloResult:
    final_returns: np.ndarray        # terminal return of each simulated run
    max_drawdowns: np.ndarray        # max drawdown of each simulated run
    sample_curves: list[np.ndarray]  # a handful of equity curves for plotting
    prob_loss: float                 # P(final return < 0)
    n_sims: int
    n_trades: int
    mode: str
    pctiles: dict[str, dict[int, float]] = field(default_factory=dict)


def _equity_and_dd(seq: np.ndarray) -> tuple[float, float]:
    eq = np.cumprod(1.0 + seq)
    roll = np.maximum.accumulate(eq)
    dd = float((eq / roll - 1.0).min())
    return float(eq[-1] - 1.0), dd


def trade_sequence_mc(
    trade_returns: list[float],
    n_sims: int = 2000,
    n_trades: int | None = None,
    mode: str = "bootstrap",
    seed: int = 42,
) -> MonteCarloResult | None:
    """
    trade_returns : realised per-trade fractional returns from a backtest.
    n_sims        : number of simulated trade sequences.
    n_trades      : length of each simulated sequence (default = number of real
                    trades). Only used in 'bootstrap' mode.
    mode          : 'bootstrap' (resample with replacement) or 'shuffle' (reorder).
    """
    tr = np.asarray(trade_returns, dtype=float)
    if tr.size < 2:
        return None

    rng = np.random.default_rng(seed)
    k = n_trades or tr.size

    finals = np.empty(n_sims)
    maxdds = np.empty(n_sims)
    sample_curves: list[np.ndarray] = []

    for s in range(n_sims):
        if mode == "shuffle":
            seq = rng.permutation(tr)
        else:
            seq = rng.choice(tr, size=k, replace=True)
        finals[s], maxdds[s] = _equity_and_dd(seq)
        if s < 200:
            sample_curves.append(np.cumprod(1.0 + seq))

    pctiles = {
        "final": {p: float(np.percentile(finals, p)) for p in (5, 25, 50, 75, 95)},
        "maxdd": {p: float(np.percentile(maxdds, p)) for p in (5, 25, 50, 75, 95)},
    }

    return MonteCarloResult(
        final_returns=finals,
        max_drawdowns=maxdds,
        sample_curves=sample_curves,
        prob_loss=float((finals < 0).mean()),
        n_sims=n_sims,
        n_trades=int(k),
        mode=mode,
        pctiles=pctiles,
    )
