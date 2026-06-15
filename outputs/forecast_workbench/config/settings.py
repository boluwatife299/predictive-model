"""
Global configuration for Forecast Workbench.
Override any value via environment variables or a .env file.
"""
from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

# ── Data ──────────────────────────────────────────────────────────────────────
DEFAULT_PERIOD   = "1y"          # yfinance historical window
DEFAULT_INTERVAL = "1d"          # bar size

# ── Model defaults ─────────────────────────────────────────────────────────────
MC_NUM_PATHS     = 500           # Monte Carlo paths
MC_HORIZON_DAYS  = 30            # forward simulation horizon
GBM_HORIZON_DAYS = 30

# ── Validation ─────────────────────────────────────────────────────────────────
OVER_PREDICT_THRESHOLD = 0.05    # flag if model over-predicts by >5 %
UNDER_PREDICT_THRESHOLD = 0.05

# ── UI ─────────────────────────────────────────────────────────────────────────
APP_TITLE   = "Forecast Workbench"
CHART_THEME = "plotly_dark"

ASSET_CLASSES = {
    "Equities":      ["AAPL", "MSFT", "NVDA", "TSLA", "SPY", "QQQ"],
    "Crypto":        ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"],
    "Fixed Income":  ["TLT", "IEF", "SHY", "HYG", "LQD"],
    "Macro":         ["GLD", "SLV", "USO", "DX-Y.NYB", "^VIX"],
    "Commodities":   None,   # uses the curated COMMODITY_CATALOG dropdown instead
}

# ── Commodities catalog ─────────────────────────────────────────────────────────
# Curated list of tradable precious-metals and agriculture futures (Yahoo "=F"
# front-month continuous contracts). The UI renders this as a dropdown — and ONLY
# for the Commodities asset class — keyed by friendly name. Grouped for display.
COMMODITY_CATALOG: dict[str, list[tuple[str, str]]] = {
    "Precious Metals": [
        ("Gold",      "GC=F"),
        ("Silver",    "SI=F"),
        ("Platinum",  "PL=F"),
        ("Palladium", "PA=F"),
        ("Copper",    "HG=F"),
    ],
    "Agriculture — Grains & Oilseeds": [
        ("Corn",          "ZC=F"),
        ("Wheat",         "ZW=F"),
        ("Soybeans",      "ZS=F"),
        ("Soybean Oil",   "ZL=F"),
        ("Soybean Meal",  "ZM=F"),
        ("Oats",          "ZO=F"),
        ("Rough Rice",    "ZR=F"),
    ],
    "Agriculture — Softs": [
        ("Coffee",        "KC=F"),
        ("Sugar #11",     "SB=F"),
        ("Cocoa",         "CC=F"),
        ("Cotton",        "CT=F"),
        ("Orange Juice",  "OJ=F"),
        ("Lumber",        "LBS=F"),
    ],
    "Agriculture — Livestock": [
        ("Live Cattle",   "LE=F"),
        ("Feeder Cattle", "GF=F"),
        ("Lean Hogs",     "HE=F"),
    ],
}

# Flattened {symbol: "Group · Friendly Name"} lookup for labels/explanations.
COMMODITY_NAMES: dict[str, str] = {
    sym: f"{name}"
    for group, items in COMMODITY_CATALOG.items()
    for name, sym in items
}

MODEL_ZOO = {
    # ── Stochastic / Classical ─────────────────────────────────────────
    "Geometric Brownian Motion (GBM)":   "gbm",
    "Monte Carlo Simulation":            "monte_carlo",
    "Ornstein-Uhlenbeck (Mean Reversion)": "ou",
    "Merton Jump Diffusion":             "jump_diffusion",
    "Heston Stochastic Volatility":      "heston",
    # ── Statistical Time Series ────────────────────────────────────────
    "ARIMA":                             "arima",
    "GARCH":                             "garch",
    # ── Machine Learning ───────────────────────────────────────────────
    "Linear Regression (Baseline)":      "linear_regression",
    "XGBoost":                           "xgboost",
    "Prophet":                           "prophet",
    "LSTM Neural Network":               "lstm",
}

# Models that produce a single deterministic path (use simple line chart)
SINGLE_PATH_MODELS = {"gbm"}

# Models that need heavy ML libraries (show spinner warning)
HEAVY_MODELS = {"lstm", "xgboost", "prophet"}
