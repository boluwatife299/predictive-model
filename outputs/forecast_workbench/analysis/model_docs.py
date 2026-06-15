"""
Structured model documentation.

Each entry explains a model along three axes the Forecast tab renders:
  • what   — what the model does, in one or two plain sentences
  • how    — how it is modelled (the maths / mechanism, LaTeX where useful)
  • why    — why/when you'd reach for it, plus its blind spots

Keeping this here (rather than inline in app.py) lets both the UI and the
Phase-5 AI narrative pull from one source of truth.
"""
from __future__ import annotations

MODEL_DOCS: dict[str, dict[str, str]] = {
    "gbm": {
        "title": "Geometric Brownian Motion (GBM)",
        "what": "Projects a single smooth price path that drifts upward at the asset's "
                "historical average return while being nudged by random noise.",
        "how": r"""
$$dS = \mu S \, dt + \sigma S \, dW_t$$

- **μ** (drift): annualised expected return from historical log returns.
- **σ** (vol): annualised std of log returns.
- **dWₜ**: Wiener process — the random shock.
""",
        "why": "The foundation of Black-Scholes; tractable and guarantees positive prices. "
               "**Blind spots:** constant μ and σ, no fat tails, no jumps, no mean reversion.",
    },
    "monte_carlo": {
        "title": "Monte Carlo Simulation",
        "what": "Runs hundreds of independent GBM paths and reads the distribution of "
                "outcomes — giving you a probability cone rather than a single guess.",
        "how": r"""
$$S_i(t+\Delta t) = S_i(t) \cdot \exp\!\left[\left(\mu - \tfrac{\sigma^2}{2}\right)\Delta t + \sigma\sqrt{\Delta t}\,Z\right],\quad Z \sim \mathcal{N}(0,1)$$

- **P5 path** = downside scenario (VaR proxy).
- **P95 path** = upside scenario.
- **Fan width** = uncertainty; wider = higher σ.
""",
        "why": "Best when you care about the *range* of outcomes and tail risk, not a point "
               "estimate. **Blind spots:** same as GBM — constant vol, no jumps, no regime changes.",
    },
    "ou": {
        "title": "Ornstein-Uhlenbeck (Mean Reversion)",
        "what": "Assumes the price is tethered to an equilibrium level and gets pulled back "
                "whenever it strays — the opposite worldview to a trending model.",
        "how": r"""
$$dX = \theta(\mu - X)\,dt + \sigma\,dW$$

- **θ** (reversion speed): how fast price snaps back to equilibrium.
- **μ** (long-run mean): the equilibrium price level.
- **Half-life** = ln(2)/θ days — how long a shock takes to halve.
""",
        "why": "Ideal for VIX, rates, commodity spreads and pairs-trading spreads. "
               "**Blind spots:** poor for trending assets; assumes prices always revert.",
    },
    "jump_diffusion": {
        "title": "Merton Jump Diffusion",
        "what": "GBM plus sudden discontinuous jumps — it bakes in the fat-tailed, gap-prone "
                "behaviour real markets show around shocks and earnings.",
        "how": r"""
$$dS = (\mu - \lambda\bar{k})S\,dt + \sigma S\,dW + S\,dJ$$

- **λ** (intensity): expected jumps per year.
- **μⱼ, σⱼ**: mean and std of log-jump sizes.
- **k̄** = E[e^Y − 1]: compensator keeping drift unbiased.
""",
        "why": "Best for crypto, single stocks around earnings, anything with fat tails. "
               "**Blind spots:** still constant vol between jumps; jump timing is random, not event-driven.",
    },
    "heston": {
        "title": "Heston Stochastic Volatility",
        "what": "Lets volatility itself wander and mean-revert, capturing the real-world fact "
                "that calm and turbulent regimes cluster and that vol spikes when prices fall.",
        "how": r"""
$$dS = \mu S\,dt + \sqrt{V}\,S\,dW_1$$
$$dV = \kappa(\theta - V)\,dt + \xi\sqrt{V}\,dW_2,\quad \text{Corr}(dW_1,dW_2)=\rho$$

- **κ**: variance mean-reversion speed.
- **θ**: long-run variance (√θ = long-run vol).
- **ξ** (xi): vol-of-vol — how much variance fluctuates.
- **ρ**: typically negative for equities (price falls → vol spikes).
- **Feller condition:** 2κθ > ξ² for variance to stay non-negative.
""",
        "why": "Best for options pricing and assets where the vol smile matters. "
               "**Blind spots:** more parameters to calibrate; can be unstable on short samples.",
    },
    "arima": {
        "title": "ARIMA(p, d, q)",
        "what": "A classical time-series model that predicts the next value from a weighted "
                "blend of recent values and recent forecast errors.",
        "how": r"""
$$y_t = c + \sum_{i=1}^p \phi_i y_{t-i} + \sum_{j=1}^q \theta_j \varepsilon_{t-j} + \varepsilon_t$$

- **p** AR lags: how many past values predict today.
- **d** differences: 1 = model returns (standard for prices).
- **q** MA lags: how many past forecast errors predict today.
""",
        "why": "Best for assets with autocorrelated returns or detectable short-term patterns. "
               "**Blind spots:** linear only, no volatility clustering, forecasts revert to mean quickly.",
    },
    "garch": {
        "title": "GARCH(p, q)",
        "what": "Fixes GBM's biggest flaw by letting volatility cluster — quiet days follow "
                "quiet days, shocks follow shocks — so the forecast cone breathes with the market.",
        "how": r"""
$$r_t = \sigma_t \varepsilon_t, \quad \sigma_t^2 = \omega + \sum_{i=1}^p \alpha_i r_{t-i}^2 + \sum_{j=1}^q \beta_j \sigma_{t-j}^2$$

- **α** (ARCH): weight on recent squared shock — how fast vol reacts.
- **β** (GARCH): weight on past variance — how long vol persists.
- **Persistence** = α + β. Close to 1 = long-memory vol.
""",
        "why": "Best whenever volatility regimes matter (most risk assets). Paths use the "
               "time-varying σₜ forecast. **Blind spots:** still assumes a fixed mean return.",
    },
    "linear_regression": {
        "title": "Linear Regression (Ridge OLS Baseline)",
        "what": "A transparent benchmark that fits a straight-line relationship between lagged "
                "prices/features and tomorrow's price.",
        "how": r"""
$$\hat{P}_{t+1} = \beta_0 + \beta_1 P_t + \beta_2 P_{t-1} + \ldots + \beta_k \bar{P} + \varepsilon$$

Features: lag₁…lag_k prices, rolling mean, rolling std, time index.
Recursive multi-step forecast — each prediction feeds the next as a new lag.
""",
        "why": "Use this as your benchmark. If LSTM or XGBoost can't beat it, they're overfit. "
               "**Blind spots:** linear only; recursive error accumulates.",
    },
    "xgboost": {
        "title": "XGBoost (Gradient Boosted Trees)",
        "what": "An ensemble of decision trees where each tree corrects the previous one's "
                "mistakes — captures non-linearities and feature interactions a line can't.",
        "how": r"""
$$\hat{y} = \sum_{k=1}^K f_k(x),\quad f_k \in \mathcal{F}$$

Features: lagged prices, rolling mean/std, log-returns. Recursive T-step forecast.

- **n_estimators**: number of trees (more = richer model).
- **max_depth**: tree complexity. Keep ≤ 5 to avoid overfitting.
""",
        "why": "Strong on non-linearity, interactions and regime-like behaviour. "
               "**Blind spots:** not natively sequential; recursive errors compound.",
    },
    "prophet": {
        "title": "Prophet (Meta / Facebook)",
        "what": "Decomposes a series into trend + seasonality + holidays — built for data with "
                "strong calendar cycles.",
        "how": r"""
$$y(t) = \text{trend}(t) + \text{seasonality}(t) + \text{holidays}(t) + \varepsilon_t$$

- **Trend**: piecewise linear with automatic changepoint detection.
- **Seasonality**: Fourier series for weekly and yearly cycles.
- **changepoint_prior_scale**: higher = more flexible trend.
""",
        "why": "Best for assets with strong seasonality (gold, agricultural commodities, BTC cycles). "
               "**Blind spots:** designed for business metrics, not stochastic price processes.",
    },
    "lstm": {
        "title": "LSTM Neural Network",
        "what": "A recurrent neural network with memory gates that learns complex non-linear "
                "temporal patterns from a rolling window of recent prices.",
        "how": r"""
$$h_t = \text{LSTM}(x_t, h_{t-1}, c_{t-1})$$

Gates control memory: **forget** (discard history), **input** (add new info),
**output** (produce hidden state). Architecture: Input(lookback,1) → LSTM(units)
→ Dropout(0.1) → Dense(1), with MinMax scaling.
""",
        "why": "Best for assets with complex non-linear temporal structure and plenty of data. "
               "**Blind spots:** data-hungry, slow to train, black-box.",
    },
}
