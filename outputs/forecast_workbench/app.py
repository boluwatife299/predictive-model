"""
Forecast Workbench — main Streamlit application.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st

# Make sure local packages are importable when launched from the project root
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import APP_TITLE, ASSET_CLASSES, MODEL_ZOO, SINGLE_PATH_MODELS, HEAVY_MODELS
from data.fetcher import DataFetcher
from data.preprocessor import Preprocessor
from models import REGISTRY
from validation.ledger import ValidationLedger
from validation.tracker import enrich_with_actuals, load_runs, log_run
from visualization.charts import (
    plot_gbm,
    plot_historical,
    plot_monte_carlo,
    plot_terminal_distribution,
    plot_validation,
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 12px 16px;
    }
    h2 { letter-spacing: 0.3px; }
    .lesson-card {
        background: rgba(248,201,72,0.08);
        border-left: 3px solid #F7C948;
        border-radius: 4px;
        padding: 14px 18px;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️  Workbench Controls")
    st.divider()

    # Asset class & ticker
    asset_class = st.selectbox("Asset Class", list(ASSET_CLASSES.keys()))
    default_tickers = ASSET_CLASSES[asset_class]
    ticker_input = st.text_input(
        "Ticker / Symbol",
        value=default_tickers[0],
        help="Equity/ETF: any Yahoo Finance ticker. Crypto: e.g. BTC-USD or BTC/USDT",
    )
    ticker = ticker_input.strip().upper()

    # Data window
    period_options = {"6 months": "6mo", "1 year": "1y", "2 years": "2y", "3 years": "3y"}
    period_label = st.selectbox("Historical Window", list(period_options.keys()), index=1)
    period = period_options[period_label]

    validation_days = st.slider(
        "Validation Window (days)",
        min_value=5, max_value=90, value=30, step=5,
        help="Hold out the last N days as out-of-sample ground truth.",
    )

    st.divider()

    # Model selection
    model_label = st.selectbox("Model", list(MODEL_ZOO.keys()))
    model_key = MODEL_ZOO[model_label]
    ModelClass = REGISTRY[model_key]

    # Dynamic parameter widgets from schema
    st.subheader("Model Parameters")
    schema = ModelClass.param_schema()
    user_params: dict = {}
    for param_name, (default, lo, hi, step, tooltip) in schema.items():
        user_params[param_name] = st.slider(
            param_name.replace("_", " ").title(),
            min_value=lo,
            max_value=hi,
            value=default,
            step=step,
            help=tooltip,
        )

    st.divider()
    run_btn = st.button("▶  Run Model", use_container_width=True, type="primary")

    st.divider()

    # ── Auto-refresh ──────────────────────────────────────────────────────────
    st.subheader("Live Data")
    auto_refresh = st.toggle("Auto-refresh data", value=False)
    refresh_interval = 5
    if auto_refresh:
        refresh_interval = st.selectbox(
            "Refresh every",
            options=[1, 5, 15, 30],
            index=1,
            format_func=lambda x: f"{x} min",
        )
        st.caption(f"Data will refresh every {refresh_interval} minute(s).")

# ── Main area ─────────────────────────────────────────────────────────────────

st.title(f"📈 {APP_TITLE}")
st.caption(
    "A quantitative environment for testing financial models, visualising "
    "market dislocations, and learning from forecast outcomes."
)

# Session state init
if "result" not in st.session_state:
    st.session_state.result = None          # validation result (training-end based)
if "result_forward" not in st.session_state:
    st.session_state.result_forward = None  # forecast result (today-based)
if "preprocessor" not in st.session_state:
    st.session_state.preprocessor = None
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "last_ticker" not in st.session_state:
    st.session_state.last_ticker = None

# ── Load data ─────────────────────────────────────────────────────────────────

fetcher = DataFetcher()

# ttl matches the refresh interval so auto-refresh always pulls fresh data
cache_ttl = refresh_interval * 60 if auto_refresh else 300

@st.cache_data(ttl=cache_ttl, show_spinner=False)
def load_data(ticker: str, asset_class: str, period: str, val_days: int):
    df = fetcher.fetch(ticker, asset_class, period=period)
    prep = Preprocessor(validation_days=val_days)
    prep.fit(df)
    return df, prep

with st.spinner(f"Fetching {ticker} data…"):
    try:
        raw_df, prep = load_data(ticker, asset_class, period, validation_days)
        st.session_state.raw_df = raw_df
        st.session_state.preprocessor = prep
        data_ok = True
    except Exception as exc:
        st.error(f"Data fetch failed: {exc}")
        data_ok = False

# ── Tabs ──────────────────────────────────────────────────────────────────────

if data_ok:
    tab_hist, tab_forecast, tab_validate, tab_history = st.tabs(
        ["📊 Historical Price", "🔮 Forecast", "🎯 Validation & Lessons", "🗂️ Model History"]
    )

    with tab_hist:
        st.plotly_chart(
            plot_historical(raw_df, ticker),
            use_container_width=True,
            config={"displayModeBar": True},
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Latest Close",  f"${prep.S0:,.2f}")
        col2.metric("Ann. Drift μ",  f"{prep.mu*100:.2f}%")
        col3.metric("Ann. Vol σ",    f"{prep.sigma*100:.2f}%")
        col4.metric(
            "Training Bars",
            f"{len(prep.train):,}",
            delta=f"−{validation_days}d val",
        )

        if auto_refresh:
            st.info(f"Auto-refresh ON — data updates every {refresh_interval} min.")

# ── Run model ─────────────────────────────────────────────────────────────────

    if run_btn:
        spinner_msg = (
            f"Running {model_label}… (this may take 20-30s for ML models)"
            if model_key in HEAVY_MODELS
            else f"Running {model_label}…"
        )
        with st.spinner(spinner_msg):
            try:
                model = ModelClass(**user_params)
                model.fit(prep)
                result = model.predict()                    # validation (training-end)
                result_forward = model.predict_forward()    # forecast (today)
                st.session_state.result = result
                st.session_state.result_forward = result_forward
                st.session_state.last_ticker = ticker
                st.session_state.run_error = None
                try:
                    log_run(ticker=ticker, asset_class=asset_class, result=result_forward)
                except Exception:
                    pass
            except Exception as exc:
                st.session_state.run_error = str(exc)

    if st.session_state.get("run_error"):
        st.error(f"Model error: {st.session_state.run_error}")

    result = st.session_state.result
    result_forward = st.session_state.result_forward

# ── Forecast tab ──────────────────────────────────────────────────────────────

    with tab_forecast:
        if result_forward is None:
            st.info("Configure the model in the sidebar and press **▶ Run Model** to generate a forecast.")
        else:
            # Use the full dataset (including validation period) as the lead-in
            # so the chart shows up to today, then projects forward
            lead_close = prep.clean["Close"].iloc[-60:]

            if model_key in SINGLE_PATH_MODELS:
                fig = plot_gbm(result_forward, historical_close=lead_close)
            else:
                fig = plot_monte_carlo(result_forward, historical_close=lead_close)

            st.plotly_chart(fig, use_container_width=True)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Entry Price (S₀)",     f"${result_forward.S0:,.2f}")
            col2.metric("Expected Terminal",    f"${result_forward.expected_price:,.2f}")
            col3.metric("5th Pct Terminal",     f"${result_forward.price_at_percentile[5]:,.2f}")
            col4.metric("95th Pct Terminal",    f"${result_forward.price_at_percentile[95]:,.2f}")

            if model_key not in SINGLE_PATH_MODELS:
                st.plotly_chart(
                    plot_terminal_distribution(result_forward),
                    use_container_width=True,
                )

            with st.expander("Model Parameters Used"):
                st.json(result_forward.params)
            if result_forward.metadata:
                with st.expander("Risk Metrics"):
                    st.json(result_forward.metadata)

# ── Validation tab ────────────────────────────────────────────────────────────

    with tab_validate:
        if result is None:
            st.info("Run a model first to see the validation report.")
        elif not prep.has_validation:
            st.warning(
                "Not enough data for a validation window. "
                "Increase the historical window or reduce the validation period."
            )
        else:
            actual_val = prep.validation["Close"]

            ledger = ValidationLedger(ticker=ticker, model_name=result.model_name)
            with st.spinner("Generating validation report…"):
                try:
                    report = ledger.evaluate(result, actual_val)
                    report_ok = True
                except ValueError as exc:
                    st.error(str(exc))
                    report_ok = False

            if report_ok:
                colour = "red" if report.over_predicted else "green"
                st.markdown(
                    f"<h3 style='color:{colour}'>"
                    f"{'▲ Over-predicted' if report.over_predicted else '▼ Under-predicted'} "
                    f"by {report.abs_error_pct:.1f}%</h3>",
                    unsafe_allow_html=True,
                )
                st.caption(report.summary_sentence)

                pred_series = report.extra["pred_series"]
                act_series  = report.extra["act_series"]

                conf_bands = None
                if model_key not in SINGLE_PATH_MODELS:
                    shared = pred_series.index
                    p5_arr  = result.percentiles[5][1:len(shared)+1]
                    p95_arr = result.percentiles[95][1:len(shared)+1]
                    conf_bands = {(5, 95): np.array([p5_arr, p95_arr])}

                st.plotly_chart(
                    plot_validation(
                        predicted_series=pred_series,
                        actual_series=act_series,
                        model_name=result.model_name,
                        confidence_bands=conf_bands,
                    ),
                    use_container_width=True,
                )

                st.subheader("Performance Ledger")
                st.dataframe(
                    report.metrics_table,
                    use_container_width=True,
                    hide_index=True,
                )

                lesson = report.lesson
                st.subheader("Analytical Breakdown & Lessons Learned")
                st.markdown(
                    f"""
                    <div class="lesson-card">
                        <strong>📌 {lesson['headline']}</strong><br><br>
                        <strong>Strategic Rationale:</strong><br>
                        {lesson['rationale']}<br><br>
                        <strong>Lesson Learned:</strong><br>
                        {lesson['lesson']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.expander("Model Behaviour Explained"):
                    _MODEL_EXPLANATIONS = {
                        "gbm": """
**Geometric Brownian Motion (GBM)**

$$dS = \\mu S \\, dt + \\sigma S \\, dW_t$$

- **μ** (drift): annualised expected return from historical log returns.
- **σ** (vol): annualised std of log returns.
- **dWₜ**: Wiener process — the random shock.

**Strengths:** tractable, positive prices guaranteed, foundation of Black-Scholes.
**Weaknesses:** constant μ and σ, no fat tails, no jumps, no mean reversion.
""",
                        "monte_carlo": """
**Monte Carlo Simulation**

Runs *N* independent GBM paths. At each step:

$$S_i(t+\\Delta t) = S_i(t) \\cdot \\exp\\!\\left[\\left(\\mu - \\tfrac{\\sigma^2}{2}\\right)\\Delta t + \\sigma\\sqrt{\\Delta t}\\,Z\\right],\\quad Z \\sim \\mathcal{N}(0,1)$$

- **P5 path** = downside scenario (VaR proxy).
- **P95 path** = upside scenario.
- **Fan width** = uncertainty; wider = higher σ.

**Weaknesses:** same as GBM — constant vol, no jumps, no regime changes.
""",
                        "ou": """
**Ornstein-Uhlenbeck (Mean Reversion)**

$$dX = \\theta(\\mu - X)\\,dt + \\sigma\\,dW$$

- **θ** (reversion speed): how fast price snaps back to equilibrium.
- **μ** (long-run mean): the equilibrium price level.
- **Half-life** = ln(2)/θ days — how long a shock takes to halve.

**Best for:** VIX, interest rates, commodity spreads, pairs-trading spreads.
**Weakness:** poor for trending assets; assumes prices always revert.
""",
                        "jump_diffusion": """
**Merton Jump Diffusion**

GBM + a compound Poisson jump process:

$$dS = (\\mu - \\lambda\\bar{k})S\\,dt + \\sigma S\\,dW + S\\,dJ$$

- **λ** (intensity): expected jumps per year.
- **μⱼ, σⱼ**: mean and std of log-jump sizes.
- **k̄** = E[e^Y − 1]: compensator keeping drift unbiased.

**Best for:** crypto, single stocks around earnings, anything with fat tails.
**Weakness:** still constant vol between jumps; jump timing is random, not event-driven.
""",
                        "heston": """
**Heston Stochastic Volatility**

$$dS = \\mu S\\,dt + \\sqrt{V}\\,S\\,dW_1$$
$$dV = \\kappa(\\theta - V)\\,dt + \\xi\\sqrt{V}\\,dW_2,\\quad \\text{Corr}(dW_1,dW_2)=\\rho$$

- **κ**: variance mean-reversion speed.
- **θ**: long-run variance (√θ = long-run vol).
- **ξ** (xi): vol-of-vol — how much variance fluctuates.
- **ρ**: typically negative for equities (price falls → vol spikes).
- **Feller condition:** 2κθ > ξ² for variance to stay non-negative.

**Best for:** options pricing, assets where the vol smile matters.
""",
                        "arima": """
**ARIMA(p, d, q)**

Works on the log-price series after *d* differences:

$$y_t = c + \\sum_{i=1}^p \\phi_i y_{t-i} + \\sum_{j=1}^q \\theta_j \\varepsilon_{t-j} + \\varepsilon_t$$

- **p** AR lags: how many past values predict today.
- **d** differences: 1 = model returns (standard for prices).
- **q** MA lags: how many past forecast errors predict today.

**Best for:** assets with autocorrelated returns or detectable patterns.
**Weakness:** linear only, no volatility clustering, forecasts revert to mean quickly.
""",
                        "garch": """
**GARCH(p, q)**

Fixes GBM's biggest flaw — lets volatility cluster:

$$r_t = \\sigma_t \\varepsilon_t, \\quad \\sigma_t^2 = \\omega + \\sum_{i=1}^p \\alpha_i r_{t-i}^2 + \\sum_{j=1}^q \\beta_j \\sigma_{t-j}^2$$

- **α** (ARCH): weight on recent squared shock — how fast vol reacts.
- **β** (GARCH): weight on past variance — how long vol persists.
- **Persistence** = α + β. Close to 1 = long-memory vol.
- **Half-life** of vol shock = log(0.5) / log(α + β) days.

Price paths are simulated using time-varying σₜ from the GARCH forecast.
""",
                        "linear_regression": """
**Linear Regression (Ridge OLS Baseline)**

Features: lag₁…lag_k prices, rolling mean, rolling std, time index.

$$\\hat{P}_{t+1} = \\beta_0 + \\beta_1 P_t + \\beta_2 P_{t-1} + \\ldots + \\beta_k \\bar{P} + \\varepsilon$$

Recursive multi-step forecast: each prediction feeds the next as a new lag.

**Use this as your benchmark.** If LSTM or XGBoost can't beat it, they're overfit.
**Weakness:** linear only; accumulating error in recursive forecasting.
""",
                        "xgboost": """
**XGBoost (Gradient Boosted Trees)**

Builds an ensemble of decision trees, each correcting the previous one's residuals:

$$\\hat{y} = \\sum_{k=1}^K f_k(x),\\quad f_k \\in \\mathcal{F}$$

Features: lagged prices, rolling mean/std, log-returns.
Recursive T-step forecast; each step uses prior predictions as new lags.

- **n_estimators**: number of trees (more = richer model).
- **max_depth**: tree complexity. Keep ≤ 5 to avoid overfitting.

**Strengths:** handles non-linearity, interactions, and regime-like behavior.
**Weakness:** not natively sequential; recursive errors compound.
""",
                        "prophet": """
**Prophet (Meta / Facebook)**

Additive decomposition:

$$y(t) = \\text{trend}(t) + \\text{seasonality}(t) + \\text{holidays}(t) + \\varepsilon_t$$

- **Trend**: piecewise linear with automatic changepoint detection.
- **Seasonality**: Fourier series for weekly and yearly cycles.
- **changepoint_prior_scale**: higher = more flexible trend.

**Best for:** assets with strong seasonality (gold, commodities, BTC cycles).
**Weakness:** designed for business metrics, not stochastic price processes.
""",
                        "lstm": """
**LSTM Neural Network (Long Short-Term Memory)**

A recurrent neural network designed to capture long-range dependencies:

$$h_t = \\text{LSTM}(x_t, h_{t-1}, c_{t-1})$$

Gates control what the network remembers vs forgets:
- **Forget gate**: discard irrelevant history.
- **Input gate**: add new information.
- **Output gate**: produce the hidden state.

Architecture here: Input(lookback, 1) → LSTM(units) → Dropout(0.1) → Dense(1).
Prices are MinMax-normalised before training and inverse-transformed after.

**Best for:** assets with complex non-linear temporal patterns.
**Weakness:** needs significant data, slow to train, black-box.
""",
                    }
                    explanation = _MODEL_EXPLANATIONS.get(model_key, "")
                    if explanation:
                        st.markdown(explanation)

# ── Model History tab ─────────────────────────────────────────────────────────

    with tab_history:
        st.subheader("Past Model Runs")
        st.caption(
            "Every time you press ▶ Run Model, the prediction is saved here. "
            "Once the forecast horizon has elapsed, the actual price is fetched "
            "and the error is calculated automatically."
        )

        runs_df = load_runs()

        if runs_df.empty:
            st.info("No runs logged yet. Run a model to start tracking.")
        else:
            # Try to enrich completed runs with actuals from the current ticker
            try:
                price_series = raw_df["Close"]
                runs_df = enrich_with_actuals(runs_df, price_series)
            except Exception:
                pass

            display_cols = [
                "run_at", "ticker", "asset_class", "model_name",
                "horizon_days", "S0", "predicted_p50",
                "predicted_p5", "predicted_p95",
                "actual_terminal", "error_pct",
            ]
            display_cols = [c for c in display_cols if c in runs_df.columns]

            rename_map = {
                "run_at": "Run At (UTC)",
                "ticker": "Ticker",
                "asset_class": "Asset Class",
                "model_name": "Model",
                "horizon_days": "Horizon (days)",
                "S0": "Entry Price",
                "predicted_p50": "Predicted (P50)",
                "predicted_p5": "Predicted (P5)",
                "predicted_p95": "Predicted (P95)",
                "actual_terminal": "Actual Price",
                "error_pct": "Error %",
            }

            display_df = runs_df[display_cols].rename(columns=rename_map)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Summary stats for completed runs
            completed = runs_df[runs_df["error_pct"].notna()]
            if not completed.empty:
                st.subheader("Completed Run Stats")
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Runs Scored", len(completed))
                col2.metric("Mean Error %", f"{completed['error_pct'].mean():.2f}%")
                col3.metric(
                    "Runs Within ±5%",
                    f"{(completed['error_pct'].abs() <= 5).sum()} / {len(completed)}",
                )

# ── Auto-refresh loop ─────────────────────────────────────────────────────────

if auto_refresh:
    time.sleep(refresh_interval * 60)
    st.rerun()
