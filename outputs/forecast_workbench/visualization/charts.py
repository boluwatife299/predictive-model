"""
Plotly chart builders for the Forecast Workbench.

All functions return a go.Figure so callers can further customise
before handing to st.plotly_chart().
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.settings import CHART_THEME
from models.base_model import ModelResult

# Colour palette
PALETTE = {
    "historical":  "#7EB8F7",
    "median":      "#F7C948",
    "cone_outer":  "rgba(120,180,255,0.12)",
    "cone_inner":  "rgba(120,180,255,0.22)",
    "actual":      "#56D364",
    "predicted":   "#F7C948",
    "error":       "#FF7B7B",
    "band_fill":   "rgba(248,201,72,0.15)",
}


# ── Historical price chart ────────────────────────────────────────────────────

def plot_historical(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Candlestick + volume sub-panel for the historical price."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.03,
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"], high=df["High"],
            low=df["Low"],   close=df["Close"],
            name="OHLC",
            increasing_line_color="#56D364",
            decreasing_line_color="#FF7B7B",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Bar(
            x=df.index, y=df["Volume"],
            name="Volume",
            marker_color=PALETTE["historical"],
            opacity=0.5,
        ),
        row=2, col=1,
    )

    fig.update_layout(
        title=f"{ticker} — Historical Price",
        template=CHART_THEME,
        xaxis_rangeslider_visible=False,
        height=500,
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=40, r=20, t=60, b=20),
    )
    return fig


# ── Monte Carlo fan chart ─────────────────────────────────────────────────────

def plot_monte_carlo(
    result: ModelResult,
    historical_close: pd.Series | None = None,
    max_display_paths: int = 100,
) -> go.Figure:
    """
    Fan chart showing:
      - Faint individual paths (up to max_display_paths)
      - 5-95 and 25-75 percentile bands
      - Median path
      - Optional historical close leading into the forecast
    """
    fig = go.Figure()

    dates = result.dates

    # Historical lead-in
    if historical_close is not None:
        fig.add_trace(go.Scatter(
            x=historical_close.index,
            y=historical_close.values,
            mode="lines",
            name="Historical Close",
            line=dict(color=PALETTE["historical"], width=2),
        ))

    # Individual paths (thinned for performance)
    n_show = min(max_display_paths, result.paths.shape[0])
    for i in range(n_show):
        fig.add_trace(go.Scatter(
            x=dates, y=result.paths[i],
            mode="lines",
            line=dict(color="rgba(200,200,200,0.08)", width=1),
            showlegend=False,
            hoverinfo="skip",
        ))

    # 5-95 outer band
    p5  = result.percentiles[5]
    p95 = result.percentiles[95]
    fig.add_trace(go.Scatter(
        x=list(dates) + list(dates[::-1]),
        y=list(p95) + list(p5[::-1]),
        fill="toself",
        fillcolor=PALETTE["cone_outer"],
        line=dict(color="rgba(0,0,0,0)"),
        name="5-95th pct",
        hoverinfo="skip",
    ))

    # 25-75 inner band
    p25 = result.percentiles[25]
    p75 = result.percentiles[75]
    fig.add_trace(go.Scatter(
        x=list(dates) + list(dates[::-1]),
        y=list(p75) + list(p25[::-1]),
        fill="toself",
        fillcolor=PALETTE["cone_inner"],
        line=dict(color="rgba(0,0,0,0)"),
        name="25-75th pct",
        hoverinfo="skip",
    ))

    # Median
    fig.add_trace(go.Scatter(
        x=dates, y=result.percentiles[50],
        mode="lines",
        name="Median path",
        line=dict(color=PALETTE["median"], width=2.5),
    ))

    # Vertical line at simulation start
    fig.add_vline(
        x=dates[0].isoformat(),
        line_dash="dash",
        line_color="rgba(255,255,255,0.3)",
        annotation_text="Forecast start",
        annotation_position="top right",
    )

    fig.update_layout(
        title=f"{result.model_name} — {result.params.get('horizon_days', '?')}d Forecast",
        template=CHART_THEME,
        xaxis_title="Date",
        yaxis_title="Price",
        height=520,
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=40, r=20, t=60, b=20),
        hovermode="x unified",
    )
    return fig


# ── GBM single-path chart ─────────────────────────────────────────────────────

def plot_gbm(
    result: ModelResult,
    historical_close: pd.Series | None = None,
) -> go.Figure:
    """Single-path GBM with optional historical lead-in."""
    fig = go.Figure()

    if historical_close is not None:
        fig.add_trace(go.Scatter(
            x=historical_close.index,
            y=historical_close.values,
            mode="lines",
            name="Historical Close",
            line=dict(color=PALETTE["historical"], width=2),
        ))

    fig.add_trace(go.Scatter(
        x=result.dates,
        y=result.paths[0],
        mode="lines",
        name="GBM Path",
        line=dict(color=PALETTE["median"], width=2.5),
    ))

    fig.add_vline(
        x=result.dates[0].isoformat(),
        line_dash="dash",
        line_color="rgba(255,255,255,0.3)",
        annotation_text="Forecast start",
        annotation_position="top right",
    )

    fig.update_layout(
        title=f"GBM — {result.params.get('horizon_days', '?')}d Single Path",
        template=CHART_THEME,
        xaxis_title="Date",
        yaxis_title="Price",
        height=480,
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=40, r=20, t=60, b=20),
    )
    return fig


# ── Validation comparison chart ───────────────────────────────────────────────

def plot_validation(
    predicted_series: pd.Series,
    actual_series: pd.Series,
    model_name: str,
    confidence_bands: dict[tuple[int, int], np.ndarray] | None = None,
) -> go.Figure:
    """
    Overlay predicted (median) vs actual price with optional confidence bands.
    Shaded error ribbon between actual and predicted.
    """
    fig = go.Figure()

    # Confidence bands (optional)
    if confidence_bands:
        for (lo_pct, hi_pct), band_arr in confidence_bands.items():
            dates_shared = predicted_series.index
            lo_arr = band_arr[0]
            hi_arr = band_arr[1]
            fig.add_trace(go.Scatter(
                x=list(dates_shared) + list(dates_shared[::-1]),
                y=list(hi_arr) + list(lo_arr[::-1]),
                fill="toself",
                fillcolor=PALETTE["band_fill"],
                line=dict(color="rgba(0,0,0,0)"),
                name=f"{lo_pct}-{hi_pct}th pct",
                hoverinfo="skip",
            ))

    # Actual price
    fig.add_trace(go.Scatter(
        x=actual_series.index,
        y=actual_series.values,
        mode="lines",
        name="Actual Price",
        line=dict(color=PALETTE["actual"], width=2.5),
    ))

    # Predicted price
    fig.add_trace(go.Scatter(
        x=predicted_series.index,
        y=predicted_series.values,
        mode="lines",
        name="Predicted (Median)",
        line=dict(color=PALETTE["predicted"], width=2.5, dash="dot"),
    ))

    # Shaded divergence fill between predicted and actual
    both_idx = predicted_series.index.intersection(actual_series.index)
    pred_aligned = predicted_series.reindex(both_idx)
    act_aligned  = actual_series.reindex(both_idx)
    error        = pred_aligned - act_aligned

    # Green where model under-predicted, red where over-predicted
    over_mask  = error > 0
    under_mask = error <= 0

    for mask, colour, label in [
        (over_mask,  "rgba(255,123,123,0.25)", "Over-predicted"),
        (under_mask, "rgba(86,211,100,0.20)",  "Under-predicted"),
    ]:
        seg_idx  = both_idx[mask]
        seg_pred = pred_aligned[mask]
        seg_act  = act_aligned[mask]
        if seg_idx.empty:
            continue
        fig.add_trace(go.Scatter(
            x=list(seg_idx) + list(seg_idx[::-1]),
            y=list(seg_pred) + list(seg_act[::-1]),
            fill="toself",
            fillcolor=colour,
            line=dict(color="rgba(0,0,0,0)"),
            name=label,
            hoverinfo="skip",
        ))

    fig.update_layout(
        title=f"Validation — {model_name}: Predicted vs Actual",
        template=CHART_THEME,
        xaxis_title="Date",
        yaxis_title="Price",
        height=480,
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=40, r=20, t=60, b=20),
        hovermode="x unified",
    )
    return fig


# ── Terminal distribution histogram ──────────────────────────────────────────

def plot_terminal_distribution(result: ModelResult) -> go.Figure:
    """Histogram of terminal (end-of-horizon) prices from Monte Carlo."""
    terminal = result.terminal_prices
    s0 = result.S0

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=terminal,
        nbinsx=60,
        marker_color=PALETTE["historical"],
        opacity=0.8,
        name="Terminal Price Distribution",
    ))

    # Vertical lines for S0 and median
    for val, label, colour in [
        (s0,                     "S₀ (entry)",  PALETTE["actual"]),
        (float(np.median(terminal)), "Median", PALETTE["predicted"]),
    ]:
        fig.add_vline(
            x=val, line_dash="dash", line_color=colour,
            annotation_text=label, annotation_position="top",
        )

    fig.update_layout(
        title=f"Terminal Price Distribution — Day {result.params.get('horizon_days', '?')}",
        template=CHART_THEME,
        xaxis_title="Price",
        yaxis_title="Frequency",
        height=380,
        margin=dict(l=40, r=20, t=60, b=20),
        showlegend=False,
    )
    return fig


# ── Backtest: equity curve vs buy & hold ──────────────────────────────────────

def plot_equity_curve(equity: pd.Series, benchmark: pd.Series) -> go.Figure:
    """Strategy equity curve overlaid on a buy-&-hold benchmark (both start at 1)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=benchmark.index, y=benchmark.values, mode="lines",
        name="Buy & hold", line=dict(color=PALETTE["historical"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values, mode="lines",
        name="Strategy", line=dict(color=PALETTE["median"], width=2.5),
    ))
    fig.add_hline(y=1.0, line_dash="dash", line_color="rgba(255,255,255,0.25)")
    fig.update_layout(
        title="Equity Curve — Strategy vs Buy & Hold",
        template=CHART_THEME, xaxis_title="Date", yaxis_title="Growth of $1",
        height=420, legend=dict(orientation="h", y=1.02),
        margin=dict(l=40, r=20, t=60, b=20), hovermode="x unified",
    )
    return fig


# ── Backtest: underwater drawdown ─────────────────────────────────────────────

def plot_drawdown(equity: pd.Series) -> go.Figure:
    """Underwater plot — percentage below the running peak."""
    dd = (equity / equity.cummax() - 1.0) * 100.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values, mode="lines", fill="tozeroy",
        name="Drawdown", line=dict(color=PALETTE["error"], width=1.5),
        fillcolor="rgba(255,123,123,0.25)",
    ))
    fig.update_layout(
        title="Drawdown (underwater)",
        template=CHART_THEME, xaxis_title="Date", yaxis_title="Drawdown %",
        height=300, margin=dict(l=40, r=20, t=60, b=20),
        showlegend=False, hovermode="x unified",
    )
    return fig


# ── Backtest: trade-sequence Monte Carlo ──────────────────────────────────────

def plot_mc_distribution(final_returns: np.ndarray, pctiles: dict[int, float]) -> go.Figure:
    """Histogram of simulated final returns with P5/P50/P95 markers."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=final_returns * 100.0, nbinsx=60,
        marker_color=PALETTE["historical"], opacity=0.8, name="Final return",
    ))
    for p, colour, dash in [
        (5, PALETTE["error"], "dash"),
        (50, PALETTE["median"], "solid"),
        (95, PALETTE["actual"], "dash"),
    ]:
        fig.add_vline(
            x=pctiles[p] * 100.0, line_dash=dash, line_color=colour,
            annotation_text=f"P{p} {pctiles[p]*100:.0f}%", annotation_position="top",
        )
    fig.add_vline(x=0, line_dash="dot", line_color="rgba(255,255,255,0.4)")
    fig.update_layout(
        title="Trade-Sequence Monte Carlo — Distribution of Final Returns",
        template=CHART_THEME, xaxis_title="Final return %", yaxis_title="Frequency",
        height=380, margin=dict(l=40, r=20, t=60, b=20), showlegend=False,
    )
    return fig


# ── Replay: progressive price with trade markers ──────────────────────────────

def plot_replay(
    df_visible: pd.DataFrame,
    ticker: str,
    markers: list[dict] | None = None,
) -> go.Figure:
    """
    Price line revealed up to the current replay bar, with entry/exit markers.
    ``markers`` is a list of {date, price, kind} where kind ∈ {long, short, exit}.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_visible.index, y=df_visible["Close"], mode="lines",
        name="Close", line=dict(color=PALETTE["historical"], width=2),
    ))
    style = {
        "long":  ("triangle-up", PALETTE["actual"], "Long"),
        "short": ("triangle-down", PALETTE["error"], "Short"),
        "exit":  ("x", PALETTE["median"], "Exit"),
    }
    for kind, (symbol, colour, label) in style.items():
        pts = [m for m in (markers or []) if m["kind"] == kind]
        if not pts:
            continue
        fig.add_trace(go.Scatter(
            x=[m["date"] for m in pts], y=[m["price"] for m in pts],
            mode="markers", name=label,
            marker=dict(symbol=symbol, color=colour, size=12,
                        line=dict(width=1, color="rgba(0,0,0,0.4)")),
        ))
    fig.update_layout(
        title=f"{ticker} — Replay",
        template=CHART_THEME, xaxis_title="Date", yaxis_title="Price",
        height=460, legend=dict(orientation="h", y=1.02),
        margin=dict(l=40, r=20, t=60, b=20), hovermode="x unified",
    )
    return fig
