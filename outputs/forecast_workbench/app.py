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
import pandas as pd
import streamlit as st

# Make sure local packages are importable when launched from the project root
sys.path.insert(0, str(Path(__file__).parent))

from analysis import llm
from analysis.drivers import get_drivers
from analysis.fundamentals import get_fundamentals
from analysis.model_docs import MODEL_DOCS
from analysis.param_docs import DATA_USAGE, PARAM_DOCS
from analysis.signals import compute_signals
from config.settings import (
    APP_TITLE,
    ASSET_CLASSES,
    COMMODITY_CATALOG,
    HEAVY_MODELS,
    MODEL_ZOO,
    SINGLE_PATH_MODELS,
)
from data.fetcher import DataFetcher
from data.preprocessor import Preprocessor
from models import REGISTRY, UNAVAILABLE
from validation.ledger import ValidationLedger
from validation.store import get_status
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

    if asset_class == "Commodities":
        # Commodities get a curated dropdown (no free-text) — only this class.
        commodity_options: list[tuple[str, str]] = []   # (display_label, symbol)
        for group, items in COMMODITY_CATALOG.items():
            for name, sym in items:
                commodity_options.append((f"{group} · {name}  ({sym})", sym))

        labels = [lbl for lbl, _ in commodity_options]
        chosen_label = st.selectbox(
            "Commodity",
            labels,
            help="Curated precious-metals and agriculture futures (Yahoo front-month).",
        )
        ticker = dict(commodity_options)[chosen_label]
    else:
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

    # Model selection — exclude models whose deps aren't installed
    available_model_labels = [
        label for label, key in MODEL_ZOO.items()
        if key not in UNAVAILABLE
    ]
    if UNAVAILABLE:
        unavailable_names = ", ".join(
            label for label, key in MODEL_ZOO.items() if key in UNAVAILABLE
        )
        st.caption(
            f"ℹ️ Not available in this environment (heavy deps): {unavailable_names}. "
            "Install locally with `pip install prophet tensorflow-cpu`."
        )
    model_label = st.selectbox("Model", available_model_labels)
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

                # Reject degenerate forecasts (NaN/inf) before they reach the
                # chart or the database — almost always insufficient history.
                check_vals = [
                    result_forward.S0, result_forward.mu, result_forward.sigma,
                    result_forward.percentiles[50][-1],
                    result_forward.percentiles[5][-1],
                    result_forward.percentiles[95][-1],
                ]
                if not all(np.isfinite(v) for v in check_vals):
                    raise ValueError(
                        f"The forecast for {ticker} came back invalid (NaN). This "
                        f"usually means {ticker} has too little price history to "
                        "estimate drift and volatility. Try a longer historical "
                        "window, or pick a more liquid ticker."
                    )

                st.session_state.result = result
                st.session_state.result_forward = result_forward
                st.session_state.last_ticker = ticker
                st.session_state.run_error = None
                try:
                    log_run(ticker=ticker, asset_class=asset_class, result=result_forward)
                    st.session_state.log_error = None
                except Exception as log_exc:
                    # Don't fail the forecast, but don't silently lose the run either.
                    st.session_state.log_error = str(log_exc)
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

            st.divider()

            # ── Why this forecast — model explainer ────────────────────────
            doc = MODEL_DOCS.get(model_key)
            if doc:
                st.subheader(f"🧠 Understanding the model — {doc['title']}")
                st.markdown(f"**What it does.** {doc['what']}")
                with st.expander("How it's modelled (the maths)"):
                    st.markdown(doc["how"])
                st.markdown(f"**Why / when to use it.** {doc['why']}")

            st.divider()

            # ── Parameters & scenario analysis (study aid) ─────────────────
            st.subheader("📐 Parameters & scenario analysis")
            st.markdown(DATA_USAGE)
            st.caption("Each parameter below shows your current value and what changing it does.")
            schema_now = ModelClass.param_schema()
            for pname in schema_now:
                pdoc = PARAM_DOCS.get(pname)
                pretty = pname.replace("_", " ").title()
                current = user_params.get(pname)
                with st.expander(f"{pretty}  —  current value: {current}"):
                    if pdoc:
                        st.markdown(f"**What it is.** {pdoc['what']}")
                        st.markdown(f"**Turn it up ▲** {pdoc['increase']}")
                        st.markdown(f"**Turn it down ▼** {pdoc['decrease']}")
                        st.markdown(f"**Scenario.** {pdoc['scenario']}")
                    else:
                        # Fall back to the slider tooltip from the schema.
                        st.markdown(schema_now[pname][4])

            st.divider()

            # ── Compare models ─────────────────────────────────────────────
            st.subheader("⚖️ Compare models")
            st.caption(
                "Run several models on the same data (with their default "
                "parameters) and compare today's forward forecast side by side. "
                "Big disagreement between models = high genuine uncertainty."
            )
            compare_labels = [lbl for lbl, k in MODEL_ZOO.items() if k not in UNAVAILABLE]
            default_compare = [
                lbl for lbl, k in MODEL_ZOO.items()
                if k in {"gbm", "monte_carlo", "ou", "jump_diffusion", "garch"}
                and k not in UNAVAILABLE
            ]
            chosen_compare = st.multiselect(
                "Models to compare",
                compare_labels,
                default=default_compare,
                help="Heavy ML models (XGBoost/LSTM/Prophet) are slower — add them only if you want to wait.",
            )
            cmp_key = (ticker, asset_class,
                       tuple(sorted(MODEL_ZOO[l] for l in chosen_compare)))
            if st.button("⚖️ Run comparison", key="run_compare") and chosen_compare:
                rows = []
                prog = st.progress(0.0)
                for i, lbl in enumerate(chosen_compare):
                    k = MODEL_ZOO[lbl]
                    try:
                        cls = REGISTRY[k]
                        defaults = {p: spec[0] for p, spec in cls.param_schema().items()}
                        m = cls(**defaults)
                        m.fit(prep)
                        r = m.predict_forward()
                        p5 = float(r.percentiles[5][-1])
                        p50 = float(r.percentiles[50][-1])
                        p95 = float(r.percentiles[95][-1])
                        exp = float(r.expected_price)
                        if all(np.isfinite(v) for v in (p5, p50, p95, exp)):
                            rows.append({
                                "Model": lbl,
                                "Expected": exp, "P5": p5, "P50": p50, "P95": p95,
                                "Implied return %": (p50 / r.S0 - 1) * 100,
                                "Band width %": (p95 - p5) / r.S0 * 100,
                            })
                    except Exception:
                        pass
                    prog.progress((i + 1) / len(chosen_compare))
                prog.empty()
                st.session_state.compare_result = (cmp_key, rows)

            cached_cmp = st.session_state.get("compare_result")
            if cached_cmp and cached_cmp[0] == cmp_key and cached_cmp[1]:
                rows = cached_cmp[1]
                entry = result_forward.S0
                cmp_df = pd.DataFrame(rows)
                fmt = {
                    "Expected": "{:,.2f}", "P5": "{:,.2f}", "P50": "{:,.2f}",
                    "P95": "{:,.2f}", "Implied return %": "{:+.1f}%",
                    "Band width %": "{:.1f}%",
                }
                st.dataframe(
                    cmp_df.style.format(fmt),
                    use_container_width=True, hide_index=True,
                )
                import plotly.graph_objects as go

                fig_cmp = go.Figure()
                for row in rows:
                    fig_cmp.add_trace(go.Scatter(
                        x=[row["P5"], row["P95"]], y=[row["Model"], row["Model"]],
                        mode="lines", line=dict(width=6),
                        showlegend=False,
                        hovertemplate="P5–P95: %{x:.2f}<extra></extra>",
                    ))
                    fig_cmp.add_trace(go.Scatter(
                        x=[row["P50"]], y=[row["Model"]], mode="markers",
                        marker=dict(size=12, symbol="diamond"),
                        showlegend=False,
                        hovertemplate="P50: %{x:.2f}<extra></extra>",
                    ))
                fig_cmp.add_vline(
                    x=entry, line_dash="dash",
                    annotation_text=f"Entry ${entry:,.2f}",
                )
                fig_cmp.update_layout(
                    template="plotly_dark",
                    height=90 + 42 * len(rows),
                    margin=dict(l=10, r=10, t=30, b=10),
                    xaxis_title="Terminal price  (bar = P5–P95, ◆ = median P50)",
                )
                st.plotly_chart(fig_cmp, use_container_width=True)
                st.caption(
                    "Each bar spans the model's P5–P95 terminal range; the diamond "
                    "is the median. Dashed line = today's entry price. Single-path "
                    "models (e.g. GBM) show as a point."
                )
            elif chosen_compare:
                st.caption("Press **Run comparison** to fetch each model's forecast.")

            st.divider()

            # ── Tailwinds / headwinds / behavioural read ───────────────────
            st.subheader("🌬️ Tailwinds, headwinds & behavioural read")
            signals = compute_signals(raw_df)
            st.caption(f"Current regime: **{signals.regime}**")

            cwind1, cwind2 = st.columns(2)
            with cwind1:
                st.markdown("**🟢 Tailwinds (supportive)**")
                if signals.tailwinds:
                    for t in signals.tailwinds:
                        st.markdown(f"- {t}")
                else:
                    st.caption("None detected from current price action.")
            with cwind2:
                st.markdown("**🔴 Headwinds (adverse)**")
                if signals.headwinds:
                    for h in signals.headwinds:
                        st.markdown(f"- {h}")
                else:
                    st.caption("None detected from current price action.")

            with st.expander("Behavioural metrics"):
                m = signals.metrics

                def _pct(x):
                    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x*100:.1f}%"

                def _num(x):
                    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.2f}"

                beh_rows = {
                    "Last close": _num(m.get("last")),
                    "1M / 3M / 6M return": f"{_pct(m.get('ret_1m'))} / {_pct(m.get('ret_3m'))} / {_pct(m.get('ret_6m'))}",
                    "RSI (14)": _num(m.get("rsi14")),
                    "Realised vol (30d / full)": f"{_pct(m.get('real_vol_30'))} / {_pct(m.get('real_vol_full'))}",
                    "Drawdown from high": _pct(m.get("drawdown_from_high")),
                    "Dist. to 52w high / low": f"{_pct(m.get('dist_52w_high'))} / {_pct(m.get('dist_52w_low'))}",
                }
                st.table(pd.DataFrame(beh_rows.items(), columns=["Metric", "Value"]))

            # ── Daily news / trends (AI) ───────────────────────────────────
            st.divider()
            st.subheader("📰 Daily news & trends")
            if not llm.is_available():
                st.caption(
                    "AI brief is off. Add `GEMINI_API_KEY` to your Streamlit "
                    "secrets (free tier at aistudio.google.com/apikey) to enable "
                    "live, Google-searched news and trend context. To use Claude "
                    "instead, set `AI_PROVIDER = \"anthropic\"` and add "
                    "`ANTHROPIC_API_KEY`."
                )
            else:
                brief_key = f"{ticker}|{asset_class}"
                if st.button("🔎 Generate AI market brief", key="ai_brief"):
                    with st.spinner("Searching the web and writing the brief…"):
                        try:
                            brief = llm.generate_market_brief(
                                ticker=ticker,
                                asset_class=asset_class,
                                display_name=ticker,
                                regime=signals.regime,
                                tailwinds=signals.tailwinds,
                                headwinds=signals.headwinds,
                                driver_categories=get_drivers(ticker, asset_class),
                            )
                            st.session_state.ai_brief_result = (brief_key, brief)
                            st.session_state.ai_brief_error = None
                        except Exception as exc:
                            st.session_state.ai_brief_result = (brief_key, None)
                            st.session_state.ai_brief_error = str(exc)

                cached_brief = st.session_state.get("ai_brief_result")
                if cached_brief and cached_brief[0] == brief_key:
                    if st.session_state.get("ai_brief_error"):
                        st.error(f"AI brief failed: {st.session_state.ai_brief_error}")
                    elif cached_brief[1]:
                        st.markdown(cached_brief[1])
                        st.caption(
                            "AI-generated from live web search — verify before "
                            "acting. Not investment advice."
                        )
                else:
                    st.caption(
                        f"Press **Generate AI market brief** for live news and "
                        f"trends on {ticker}."
                    )

# ── Validation tab ────────────────────────────────────────────────────────────

    with tab_validate:
        # ── Fundamental analysis (instrument-level, shown regardless of run) ──
        st.subheader(f"🏛️ Fundamental Analysis — {ticker}")

        # Structured price drivers for non-corporate assets (commodities, crypto,
        # rates, macro). Free/rule-based, so always rendered for those classes.
        _drivers = get_drivers(ticker, asset_class)
        if _drivers:
            st.markdown("**What moves this instrument**")
            dcols = st.columns(min(len(_drivers), 3))
            for i, (cat, factors) in enumerate(_drivers.items()):
                with dcols[i % len(dcols)]:
                    st.markdown(f"*{cat}*")
                    for f in factors:
                        st.markdown(f"- {f}")
            st.divider()

        @st.cache_data(ttl=3600, show_spinner=False)
        def _load_fundamentals(tk: str, ac: str):
            return get_fundamentals(tk, ac)

        # Fetch on demand only. yfinance's .info is a slow web-scrape; Streamlit
        # executes the body of *every* tab on each rerun, so calling it eagerly
        # here would stall app startup (perpetual loading spinner). Gate it
        # behind a button so the network call only fires when the user asks.
        fund_key = (ticker, asset_class)
        if st.button("📥 Load / refresh fundamentals", key="load_fund"):
            with st.spinner("Loading fundamentals…"):
                try:
                    st.session_state.fundamentals = (fund_key, _load_fundamentals(ticker, asset_class))
                    st.session_state.fund_error = None
                except Exception as exc:
                    st.session_state.fundamentals = (fund_key, None)
                    st.session_state.fund_error = str(exc)

        cached_fund = st.session_state.get("fundamentals")
        fund = cached_fund[1] if cached_fund and cached_fund[0] == fund_key else None

        if (
            fund is None
            and st.session_state.get("fund_error")
            and cached_fund and cached_fund[0] == fund_key
        ):
            st.caption(f"Fundamentals unavailable: {st.session_state.fund_error}")

        if fund is None:
            st.caption(
                f"Press **Load / refresh fundamentals** to pull valuation, "
                f"profitability, growth and analyst data for {ticker}."
            )
        else:
            if not fund.applicable:
                st.info(fund.note)
            else:
                if fund.sector or fund.industry:
                    st.caption(f"**{fund.name}** · {fund.sector} · {fund.industry}")
                if fund.note:
                    st.caption(fund.note)

                group_items = list(fund.groups.items())
                if group_items:
                    fcols = st.columns(min(len(group_items), 2))
                    for i, (group, kv) in enumerate(group_items):
                        with fcols[i % len(fcols)]:
                            st.markdown(f"**{group}**")
                            st.table(
                                pd.DataFrame(kv.items(), columns=["Metric", "Value"])
                            )

                if fund.analyst:
                    st.markdown("**Analyst view**")
                    st.table(
                        pd.DataFrame(fund.analyst.items(), columns=["Metric", "Value"])
                    )

                if fund.summary:
                    with st.expander("Business summary"):
                        st.write(fund.summary)

        st.divider()
        st.subheader("🎯 Validation & Lessons")

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
                    doc = MODEL_DOCS.get(model_key)
                    if doc:
                        st.markdown(f"### {doc['title']}")
                        st.markdown(f"**What it does.** {doc['what']}")
                        st.markdown(doc["how"])
                        st.markdown(f"**Why / when to use it.** {doc['why']}")

# ── Model History tab ─────────────────────────────────────────────────────────

    with tab_history:
        st.subheader("Past Model Runs")
        st.caption(
            "Every time you press ▶ Run Model, the prediction is saved here. "
            "Once the forecast horizon has elapsed, the actual price is fetched "
            "and the error is calculated automatically."
        )

        # Persistence status — make it obvious whether runs survive a redeploy,
        # and *why* if they won't.
        status = get_status()
        if status["connected"]:
            st.success(
                "🟢 Connected to managed Postgres — history persists across "
                "restarts and redeploys."
            )
        elif status["configured"]:
            # A DATABASE_URL was found but the connection failed → show the reason.
            st.error(
                "🔴 `DATABASE_URL` is set but the database connection **failed**, "
                "so the app fell back to local SQLite (runs will not persist in the "
                f"cloud). Reason:\n\n```\n{status['error']}\n```\n\n"
                "Common causes: wrong password, using the **direct** connection "
                "(port 5432, IPv6-only) instead of the **Transaction pooler** "
                "(port 6543, host `…pooler.supabase.com`), or an un-encoded special "
                "character in the password."
            )
        else:
            st.warning(
                "🟡 No `DATABASE_URL` detected — using local SQLite. History persists "
                "on this machine only; a cloud deployment will **lose runs on every "
                "redeploy**. Add `DATABASE_URL` (your Supabase pooler string) to your "
                "Streamlit **Settings → Secrets** to persist history in the cloud."
            )

        if st.session_state.get("log_error"):
            st.error(
                f"⚠️ The last run could **not** be saved to history: "
                f"{st.session_state.log_error}"
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
