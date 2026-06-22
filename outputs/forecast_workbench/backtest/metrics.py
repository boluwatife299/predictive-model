"""
Performance metrics — the numbers that decide whether a strategy has an edge.

Forecast-quality metrics (MAPE, directional accuracy) answer "was the prediction
close?". These answer the only question that pays: "does trading this rule make
money, and how much pain on the way?" You can be right 45% of the time and profit
with good payoff, or right 60% and bleed out with bad risk/reward — so win rate
alone is a trap. Expectancy, profit factor and max drawdown are the load-bearing
numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.engine import BacktestResult

TRADING_DAYS = 252


def _safe(x: float) -> float:
    return float(x) if np.isfinite(x) else float("nan")


def compute_metrics(bt: BacktestResult) -> dict[str, float]:
    eq = bt.equity
    rets = bt.returns.to_numpy(dtype=float)
    tr = np.array(bt.trade_returns, dtype=float)  # fractional per-trade returns

    out: dict[str, float] = {}

    # ── Equity-curve metrics ────────────────────────────────────────────────
    total_return = float(eq.iloc[-1] - 1.0) if len(eq) else float("nan")
    n_bars = len(eq)
    years = n_bars / TRADING_DAYS if n_bars else float("nan")
    cagr = (eq.iloc[-1] ** (1.0 / years) - 1.0) if (years and years > 0) else float("nan")

    roll_max = eq.cummax()
    drawdown = eq / roll_max - 1.0
    max_dd = float(drawdown.min()) if len(drawdown) else float("nan")

    vol = float(np.std(rets, ddof=1)) if len(rets) > 1 else float("nan")
    mean_r = float(np.mean(rets)) if len(rets) else float("nan")
    sharpe = (mean_r / vol * np.sqrt(TRADING_DAYS)) if (vol and vol > 0) else float("nan")
    downside = rets[rets < 0]
    dvol = float(np.std(downside, ddof=1)) if len(downside) > 1 else float("nan")
    sortino = (mean_r / dvol * np.sqrt(TRADING_DAYS)) if (dvol and dvol > 0) else float("nan")
    calmar = (cagr / abs(max_dd)) if (max_dd and max_dd < 0) else float("nan")
    exposure = float((bt.positions != 0).mean()) if len(bt.positions) else float("nan")

    # ── Benchmark ───────────────────────────────────────────────────────────
    bench_return = float(bt.benchmark.iloc[-1] - 1.0) if len(bt.benchmark) else float("nan")

    # ── Trade-level metrics ─────────────────────────────────────────────────
    n_trades = int(len(tr))
    if n_trades:
        wins = tr[tr > 0]
        losses = tr[tr <= 0]
        win_rate = len(wins) / n_trades
        avg_win = float(np.mean(wins)) if len(wins) else 0.0
        avg_loss = float(np.mean(losses)) if len(losses) else 0.0
        payoff = (avg_win / abs(avg_loss)) if avg_loss != 0 else float("nan")
        gross_win = float(wins.sum())
        gross_loss = float(abs(losses.sum()))
        profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("nan")
        expectancy = float(np.mean(tr))      # avg return per trade
        best = float(tr.max())
        worst = float(tr.min())
    else:
        win_rate = avg_win = avg_loss = payoff = profit_factor = float("nan")
        expectancy = best = worst = float("nan")

    out.update({
        "total_return": _safe(total_return),
        "cagr": _safe(cagr),
        "max_drawdown": _safe(max_dd),
        "sharpe": _safe(sharpe),
        "sortino": _safe(sortino),
        "calmar": _safe(calmar),
        "exposure": _safe(exposure),
        "benchmark_return": _safe(bench_return),
        "excess_return": _safe(total_return - bench_return),
        "n_trades": n_trades,
        "win_rate": _safe(win_rate),
        "avg_win": _safe(avg_win),
        "avg_loss": _safe(avg_loss),
        "payoff_ratio": _safe(payoff),
        "profit_factor": _safe(profit_factor),
        "expectancy": _safe(expectancy),
        "best_trade": _safe(best),
        "worst_trade": _safe(worst),
    })
    return out


def metrics_table(m: dict[str, float]) -> pd.DataFrame:
    """Format the metrics dict into a tidy two-column display frame."""
    def pct(x: float) -> str:
        return f"{x * 100:+.2f}%" if np.isfinite(x) else "—"

    def num(x: float, d: int = 2) -> str:
        return f"{x:.{d}f}" if np.isfinite(x) else "—"

    rows = [
        ("Total return", pct(m["total_return"])),
        ("Buy & hold return", pct(m["benchmark_return"])),
        ("Excess vs buy & hold", pct(m["excess_return"])),
        ("CAGR", pct(m["cagr"])),
        ("Max drawdown", pct(m["max_drawdown"])),
        ("Sharpe (ann.)", num(m["sharpe"])),
        ("Sortino (ann.)", num(m["sortino"])),
        ("Calmar", num(m["calmar"])),
        ("Time in market", pct(m["exposure"])),
        ("Trades", f"{int(m['n_trades'])}"),
        ("Win rate", pct(m["win_rate"])),
        ("Expectancy / trade", pct(m["expectancy"])),
        ("Profit factor", num(m["profit_factor"])),
        ("Payoff ratio (avg win / avg loss)", num(m["payoff_ratio"])),
        ("Avg win", pct(m["avg_win"])),
        ("Avg loss", pct(m["avg_loss"])),
        ("Best / worst trade", f"{pct(m['best_trade'])}  /  {pct(m['worst_trade'])}"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])
