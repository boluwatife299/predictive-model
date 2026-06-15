"""
Behavioural & technical signals.

Turns a raw OHLCV DataFrame into a structured read of *how the asset is
behaving right now* — trend, momentum, volatility regime, drawdown, positioning
— and distils that into plain-English tailwinds (supportive) and headwinds
(adverse) that contextualise the model's forecast.

Everything here is rule-based and deterministic. The Phase-5 AI layer consumes
this same dict to write a richer narrative, so signals stay reproducible and
auditable rather than hallucinated.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BehaviouralReport:
    metrics: dict[str, float] = field(default_factory=dict)   # raw numbers
    tailwinds: list[str] = field(default_factory=list)        # supportive factors
    headwinds: list[str] = field(default_factory=list)        # adverse factors
    regime: str = ""                                          # one-line summary


def _rsi(close: pd.Series, window: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    val = rsi.iloc[-1]
    return float(val) if pd.notna(val) else 50.0


def _safe_return(close: pd.Series, lookback: int) -> float | None:
    if len(close) <= lookback:
        return None
    past = close.iloc[-lookback - 1]
    if past == 0:
        return None
    return float(close.iloc[-1] / past - 1.0)


def compute_signals(df: pd.DataFrame) -> BehaviouralReport:
    """Derive behavioural/technical signals from an OHLCV DataFrame."""
    rep = BehaviouralReport()
    close = df["Close"].dropna()
    if len(close) < 30:
        rep.regime = "Insufficient history for a behavioural read."
        return rep

    last = float(close.iloc[-1])
    sma20 = float(close.rolling(20).mean().iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else np.nan
    sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else np.nan

    log_ret = np.log(close / close.shift(1)).dropna()
    real_vol_30 = float(log_ret.tail(30).std() * np.sqrt(252)) if len(log_ret) >= 30 else np.nan
    real_vol_full = float(log_ret.std() * np.sqrt(252)) if len(log_ret) > 1 else np.nan

    rsi = _rsi(close)
    r_1m = _safe_return(close, 21)
    r_3m = _safe_return(close, 63)
    r_6m = _safe_return(close, 126)

    running_max = close.cummax()
    drawdown = float(last / running_max.iloc[-1] - 1.0)
    high_52w = float(close.tail(252).max())
    low_52w = float(close.tail(252).min())
    dist_high = last / high_52w - 1.0 if high_52w else np.nan
    dist_low = last / low_52w - 1.0 if low_52w else np.nan

    rep.metrics = {
        "last": last,
        "sma20": sma20, "sma50": sma50, "sma200": sma200,
        "rsi14": rsi,
        "ret_1m": r_1m, "ret_3m": r_3m, "ret_6m": r_6m,
        "real_vol_30": real_vol_30, "real_vol_full": real_vol_full,
        "drawdown_from_high": drawdown,
        "dist_52w_high": dist_high, "dist_52w_low": dist_low,
        "high_52w": high_52w, "low_52w": low_52w,
    }

    # ── Trend / moving-average alignment ───────────────────────────────────
    if not np.isnan(sma50) and not np.isnan(sma200):
        if last > sma50 > sma200:
            rep.tailwinds.append(
                "Price is above both the 50- and 200-day moving averages — a "
                "classic uptrend alignment (golden-cross structure)."
            )
        elif last < sma50 < sma200:
            rep.headwinds.append(
                "Price is below both the 50- and 200-day moving averages — a "
                "downtrend alignment (death-cross structure)."
            )
    if last > sma20:
        rep.tailwinds.append("Trading above its 20-day average — short-term momentum is positive.")
    else:
        rep.headwinds.append("Trading below its 20-day average — short-term momentum is negative.")

    # ── Momentum (multi-horizon returns) ───────────────────────────────────
    if r_3m is not None:
        if r_3m > 0.10:
            rep.tailwinds.append(f"Strong 3-month momentum (+{r_3m*100:.1f}%).")
        elif r_3m < -0.10:
            rep.headwinds.append(f"Weak 3-month momentum ({r_3m*100:.1f}%).")

    # ── RSI (overbought / oversold) ────────────────────────────────────────
    if rsi >= 70:
        rep.headwinds.append(f"RSI is {rsi:.0f} — overbought; pullback risk is elevated.")
    elif rsi <= 30:
        rep.tailwinds.append(f"RSI is {rsi:.0f} — oversold; mean-reversion bounce is possible.")

    # ── Volatility regime ──────────────────────────────────────────────────
    if not np.isnan(real_vol_30) and not np.isnan(real_vol_full) and real_vol_full > 0:
        ratio = real_vol_30 / real_vol_full
        if ratio > 1.3:
            rep.headwinds.append(
                f"Recent 30-day volatility ({real_vol_30*100:.0f}%) is well above its "
                f"longer-run level — an expanding-volatility regime widens forecast risk."
            )
        elif ratio < 0.7:
            rep.tailwinds.append(
                f"Recent volatility ({real_vol_30*100:.0f}%) is compressed versus its "
                f"longer-run level — a calmer regime tightens the forecast cone."
            )

    # ── Drawdown / positioning ─────────────────────────────────────────────
    if drawdown < -0.20:
        rep.headwinds.append(
            f"Currently {drawdown*100:.0f}% off its recent high — in a meaningful drawdown."
        )
    if dist_high is not None and dist_high > -0.03:
        rep.tailwinds.append("Within 3% of its 52-week high — pressing into breakout territory.")
    if dist_low is not None and dist_low < 0.05:
        rep.headwinds.append("Within 5% of its 52-week low — testing major support.")

    # ── Regime one-liner ───────────────────────────────────────────────────
    trend = "uptrend" if last > sma20 and (np.isnan(sma50) or last > sma50) else (
        "downtrend" if last < sma20 and (np.isnan(sma50) or last < sma50) else "rangebound"
    )
    vol_word = (
        "elevated volatility" if not np.isnan(real_vol_30) and real_vol_30 > 0.40
        else "moderate volatility" if not np.isnan(real_vol_30) and real_vol_30 > 0.20
        else "low volatility"
    )
    rep.regime = f"{trend.capitalize()} · {vol_word} · RSI {rsi:.0f}"
    return rep
