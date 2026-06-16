"""
Parameter documentation & scenario analysis.

For every tunable parameter the sidebar exposes, explain what it means, what
happens when you turn it up vs down, and a concrete "+X" scenario — so the
workbench doubles as a study tool. Keyed by the exact param names returned by
each model's `param_schema()`.
"""
from __future__ import annotations

# What data the models actually consume — shown once at the top of the section.
DATA_USAGE = """
**What data is used.** Every model is fit on the **historical window** you pick
in the sidebar (e.g. 1 year of daily closing prices), minus the last
*validation window* days which are held out as out-of-sample ground truth.
From the training slice the app estimates the annualised **drift (μ)** and
**volatility (σ)** you see on the Historical tab. The forecast then projects
*forward from today* using those estimates. Longer windows give more stable
μ/σ but blend in older regimes; shorter windows react faster but are noisier.
"""

PARAM_DOCS: dict[str, dict[str, str]] = {
    "horizon_days": {
        "what": "How many trading days ahead to forecast (≈21 = 1 month, ≈252 = 1 year).",
        "increase": "A longer horizon **widens the uncertainty cone** — risk compounds, so the "
                    "gap between the P5 and P95 paths grows roughly with the square root of time. "
                    "Expected drift also accumulates, pushing the median further from today's price.",
        "decrease": "A shorter horizon tightens the cone and keeps the forecast close to the entry price.",
        "scenario": "**+10 days (e.g. 30 → 40):** the cone widens by roughly √(40/30) ≈ 1.15×, so "
                    "your P5–P95 band gets ~15% wider, and the expected terminal moves ~33% further "
                    "along the drift. Use this to see how confidence decays with time.",
    },
    "num_paths": {
        "what": "How many independent Monte Carlo price paths to simulate.",
        "increase": "More paths give a **smoother, more stable estimate of the percentile bands** "
                    "(P5/P50/P95) — they reduce *Monte Carlo sampling noise*. They do **not** change "
                    "the true underlying distribution or make the model more correct about the real "
                    "world; they just measure the model's own distribution more precisely. Accuracy "
                    "improves ~1/√N, so returns diminish fast.",
        "decrease": "Fewer paths run faster but the bands jitter from run to run.",
        "scenario": "**+10 paths (e.g. 150 → 160):** essentially no visible change — sampling error "
                    "only falls ~3%. To meaningfully tighten the estimate you need a *multiple* "
                    "(150 → 600 halves the noise). Lesson: num_paths controls **estimation precision, "
                    "not forecast skill**.",
    },
    "seed": {
        "what": "The random-number seed. Fixes which specific random draws the simulation uses.",
        "increase": "Changing the seed draws a **different random sample of paths from the same "
                    "distribution** — the median/P5/P95 wobble slightly but are statistically "
                    "equivalent. A fixed seed makes runs **reproducible** (same inputs → same chart).",
        "decrease": "Same effect — any change is just a different sample. (0 often means 'random each run'.)",
        "scenario": "**+10 to the seed:** a completely different set of simulated paths, but with "
                    "enough num_paths the summary stats barely move. If your P5/P95 swing a lot when "
                    "you nudge the seed, that's a signal you need **more paths**, not a 'better' seed.",
    },
    "p": {
        "what": "AR order — how many past values feed the prediction (ARIMA) / ARCH lags (GARCH).",
        "increase": "Captures longer memory / more autocorrelation, but risks **overfitting** noise "
                    "and can make forecasts unstable.",
        "decrease": "Simpler, more robust; may miss real structure.",
        "scenario": "**+1 (e.g. 2 → 3):** adds one more lag term. If out-of-sample error doesn't "
                    "improve on the Validation tab, the extra lag is just fitting noise — keep it low.",
    },
    "d": {
        "what": "Differencing order. d=1 models *returns* (the standard for prices); d=0 models the level.",
        "increase": "Higher differencing removes more trend but can over-difference and inject noise.",
        "decrease": "d=0 assumes the price level itself is stationary — rarely true for assets.",
        "scenario": "**d=1 vs d=2:** d=1 is almost always right for prices. d=2 differences the "
                    "returns again — only for series with a strong changing trend.",
    },
    "q": {
        "what": "MA order — how many past forecast *errors* feed the next prediction.",
        "increase": "Smooths reaction to recent shocks; too high overfits.",
        "decrease": "More reactive to the latest error.",
        "scenario": "**+1 (e.g. 2 → 3):** check the Validation tab — if MAPE/RMSE don't drop, "
                    "revert. Parsimony usually wins in time-series models.",
    },
    "jump_threshold": {
        "what": "How many daily σ a return must exceed to be classified as a 'jump' (Merton model).",
        "increase": "A **higher** threshold flags **fewer** jumps — the model behaves more like plain "
                    "GBM with thinner tails.",
        "decrease": "A **lower** threshold flags **more** jumps — fatter tails, wider crash/spike "
                    "scenarios in the cone.",
        "scenario": "**2σ → 3σ:** far fewer historical moves count as jumps, so the jump intensity (λ) "
                    "drops and the downside P5 path gets less extreme. Lower it to stress-test fat tails.",
    },
    "kappa": {
        "what": "Heston variance mean-reversion speed — how fast volatility snaps back to its long-run level.",
        "increase": "Volatility shocks die out **faster**; the vol process is more tightly anchored.",
        "decrease": "Vol shocks **persist** longer — clustered, regime-like volatility.",
        "scenario": "**+1 (e.g. 2 → 3):** a vol spike fades quicker, so the forecast cone stabilises "
                    "sooner. Pair with the Feller condition (2κθ > ξ²) so variance stays positive.",
    },
    "xi": {
        "what": "Vol-of-vol (%) — how much the variance itself fluctuates (Heston).",
        "increase": "**Fatter tails and a more skewed terminal distribution** — variance swings more, "
                    "so extreme outcomes become more likely.",
        "decrease": "Approaches constant-volatility GBM behaviour.",
        "scenario": "**+10% (e.g. 50 → 60):** the tails of the terminal-price histogram get heavier — "
                    "watch the P95/P5 spread widen even with the same drift.",
    },
    "rho": {
        "what": "Price-vol correlation (%). Negative (typical for equities) = vol rises when price falls.",
        "increase": "Toward 0 / positive → the leverage effect weakens; downside and vol decouple.",
        "decrease": "More negative → stronger leverage effect, **heavier left tail** (downside risk).",
        "scenario": "**−70% → −90%:** crashes and vol spikes become more tightly linked, skewing the "
                    "distribution further to the downside — a more conservative risk picture.",
    },
    "lookback": {
        "what": "How many past days are used as lag features (ML models: Linear Regression, XGBoost).",
        "increase": "More context per prediction; can capture longer patterns but adds features and "
                    "**overfitting risk**, and shrinks the usable training set.",
        "decrease": "Leaner, faster, more reactive to recent days.",
        "scenario": "**+10 (e.g. 20 → 30):** the model sees a longer recent window. If Validation "
                    "error rises, you've added noise features — shorten it back.",
    },
    "n_estimators": {
        "what": "Number of boosting trees in XGBoost.",
        "increase": "More trees can fit subtler patterns but **slow training and overfit** if unregularised.",
        "decrease": "Faster, more robust, less prone to memorising noise.",
        "scenario": "**+10 (e.g. 100 → 110):** marginal. Big jumps (100 → 400) matter more — but watch "
                    "the Validation tab: if train improves while out-of-sample doesn't, you're overfitting.",
    },
    "max_depth": {
        "what": "Maximum depth of each XGBoost tree — model complexity per tree.",
        "increase": "Deeper trees capture more interactions but **overfit quickly**; keep ≤ 5.",
        "decrease": "Shallow trees generalise better (the usual safe choice).",
        "scenario": "**+1 (e.g. 3 → 4):** each tree can model one more level of interaction. Past ~5, "
                    "out-of-sample accuracy usually *falls* even as in-sample improves.",
    },
}
