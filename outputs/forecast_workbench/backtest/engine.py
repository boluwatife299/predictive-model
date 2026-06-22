"""
Walk-forward backtest engine.

This is the loop fxreplay's whole product is built around: step through history
one bar at a time, let the strategy decide using *only* what was known at that
bar, simulate the resulting trades with realistic stops/costs, and record what
actually happened.

Execution model
---------------
- A decision made at the close of bar ``i`` takes effect from bar ``i+1`` — the
  position set this bar earns/loses the next bar's move. No same-bar lookahead.
- While in a position, intrabar stops and targets are checked against each bar's
  High/Low. If the bar gaps straight through the level, the fill happens at the
  open (you can't get filled at a price the market skipped).
- One position at a time, sized to full equity (a clean, comparable unit). A
  round-trip pays ``cost_bps`` per side (commission + slippage proxy).

The output is a trade ledger, an equity curve, a buy-&-hold benchmark, and the
per-bar return stream — everything the metrics and Monte-Carlo layers consume.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from strategy.base import Strategy


@dataclass
class Trade:
    direction: int          # +1 long, -1 short
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    bars_held: int
    return_pct: float       # net of round-trip cost, direction-adjusted (%)
    r_multiple: float       # return / initial risk (nan if no stop was set)
    exit_reason: str        # 'stop' | 'target' | 'signal' | 'end-of-test'


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity: pd.Series        # strategy equity curve, starts at 1.0
    benchmark: pd.Series     # buy-&-hold equity over the same window
    returns: pd.Series       # per-bar strategy returns
    positions: pd.Series     # position carried into each bar
    params: dict = field(default_factory=dict)

    @property
    def trade_returns(self) -> list[float]:
        """Per-trade fractional returns — the input to trade-sequence MC."""
        return [t.return_pct / 100.0 for t in self.trades]

    def trades_dataframe(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        rows = [{
            "Direction": "Long" if t.direction > 0 else "Short",
            "Entry Date": t.entry_date,
            "Entry Price": round(t.entry_price, 4),
            "Exit Date": t.exit_date,
            "Exit Price": round(t.exit_price, 4),
            "Bars Held": t.bars_held,
            "Return %": round(t.return_pct, 2),
            "R-Multiple": round(t.r_multiple, 2) if np.isfinite(t.r_multiple) else None,
            "Exit Reason": t.exit_reason,
        } for t in self.trades]
        return pd.DataFrame(rows)


def run_backtest(
    df: pd.DataFrame,
    strategy: Strategy,
    cost_bps: float = 5.0,
    allow_short: bool = False,
) -> BacktestResult:
    """
    Backtest ``strategy`` over the OHLCV frame ``df``.

    cost_bps    : per-side cost in basis points (5 = 0.05% each way).
    allow_short : if False, short signals are treated as 'go flat'.
    """
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    n = len(df)
    if n < strategy.warmup + 2:
        raise ValueError(
            f"Not enough data: need > {strategy.warmup + 2} bars for this strategy, "
            f"have {n}. Increase the historical window or lower the lookback."
        )

    strategy.prepare(df)

    close = df["Close"].to_numpy(dtype=float)
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    open_ = df["Open"].to_numpy(dtype=float)
    dates = df.index

    cost = cost_bps * 1e-4
    start = max(strategy.warmup, 1)

    equity = 1.0
    eq_vals: list[float] = []
    pos_vals: list[int] = []
    bar_rets: list[float] = []

    pos = 0
    entry_price = entry_i = 0
    entry_date = None
    stop = target = None
    init_risk = np.nan
    trades: list[Trade] = []

    def book_trade(exit_price: float, exit_i: int, reason: str) -> None:
        gross = pos * (exit_price / entry_price - 1.0)
        net = (1.0 + gross) * (1.0 - cost) ** 2 - 1.0
        r_mult = (gross / init_risk) if (init_risk and np.isfinite(init_risk)) else np.nan
        trades.append(Trade(
            direction=pos,
            entry_date=entry_date,
            entry_price=entry_price,
            exit_date=dates[exit_i],
            exit_price=exit_price,
            bars_held=exit_i - entry_i,
            return_pct=net * 100.0,
            r_multiple=r_mult,
            exit_reason=reason,
        ))

    for i in range(start, n):
        prev_c = close[i - 1]
        bar_ret = 0.0
        just_exited = False

        # 1) Mark the open position; check intrabar stop / target.
        if pos != 0:
            exit_price = None
            reason = ""
            if pos > 0:
                if stop is not None and low[i] <= stop:
                    exit_price = stop if open_[i] >= stop else open_[i]
                    reason = "stop"
                elif target is not None and high[i] >= target:
                    exit_price = target if open_[i] <= target else open_[i]
                    reason = "target"
            else:
                if stop is not None and high[i] >= stop:
                    exit_price = stop if open_[i] <= stop else open_[i]
                    reason = "stop"
                elif target is not None and low[i] <= target:
                    exit_price = target if open_[i] >= target else open_[i]
                    reason = "target"

            if exit_price is not None:
                bar_ret = pos * (exit_price / prev_c - 1.0)
                equity *= (1.0 + bar_ret) * (1.0 - cost)
                book_trade(exit_price, i, reason)
                pos = 0
                stop = target = None
                init_risk = np.nan
                just_exited = True
            else:
                bar_ret = pos * (close[i] / prev_c - 1.0)
                equity *= (1.0 + bar_ret)

        # 2) Decide at the close of bar i (effective next bar).
        if pos == 0 and not just_exited:
            sig = strategy.signal(i, pos)
            d = sig.direction
            if d < 0 and not allow_short:
                d = 0
            if d != 0:
                pos = d
                entry_price = close[i]
                entry_date = dates[i]
                entry_i = i
                stop = sig.stop
                target = sig.target
                init_risk = (
                    abs(entry_price - stop) / entry_price
                    if stop not in (None, 0)
                    else np.nan
                )
                equity *= (1.0 - cost)  # entry cost
        elif pos != 0:
            sig = strategy.signal(i, pos)
            d = sig.direction
            if d < 0 and not allow_short:
                d = 0
            if d != pos:  # signal flattened or flipped → exit at close
                equity *= (1.0 - cost)
                book_trade(close[i], i, "signal")
                pos = 0
                stop = target = None
                init_risk = np.nan

        eq_vals.append(equity)
        pos_vals.append(pos)
        bar_rets.append(bar_ret)

    # Close any still-open position at the final bar.
    if pos != 0:
        equity *= (1.0 - cost)
        book_trade(close[n - 1], n - 1, "end-of-test")
        eq_vals[-1] = equity
        pos_vals[-1] = 0

    idx = dates[start:n]
    equity_s = pd.Series(eq_vals, index=idx, name="Strategy")
    returns_s = pd.Series(bar_rets, index=idx, name="Return")
    positions_s = pd.Series(pos_vals, index=idx, name="Position")
    benchmark_s = pd.Series(close[start:n] / prev_close_at(close, start), index=idx, name="Buy & Hold")

    return BacktestResult(
        trades=trades,
        equity=equity_s,
        benchmark=benchmark_s,
        returns=returns_s,
        positions=positions_s,
        params={
            "strategy": strategy.name,
            "cost_bps": cost_bps,
            "allow_short": allow_short,
            "bars_tested": int(n - start),
        },
    )


def prev_close_at(close: np.ndarray, start: int) -> float:
    """Buy-&-hold reference price: the close just before the test starts."""
    return float(close[start - 1]) if start > 0 else float(close[0])
