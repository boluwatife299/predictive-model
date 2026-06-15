"""
Fundamental analysis via yfinance.

For equities/ETFs this surfaces valuation, profitability, growth, balance-sheet
health, analyst targets and a business description. Crypto and commodity
futures have no corporate fundamentals, so we return whatever market context is
available and flag the rest as not applicable — the UI handles this gracefully.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import yfinance as yf


@dataclass
class FundamentalsReport:
    applicable: bool = True
    name: str = ""
    sector: str = ""
    industry: str = ""
    summary: str = ""
    groups: dict[str, dict[str, str]] = field(default_factory=dict)  # group -> {label: value}
    analyst: dict[str, str] = field(default_factory=dict)
    note: str = ""


def _fmt_money(v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(v) >= div:
            return f"${v/div:,.2f}{unit}"
    return f"${v:,.2f}"


def _fmt_pct(v) -> str:
    try:
        return f"{float(v)*100:.2f}%"
    except (TypeError, ValueError):
        return "—"


def _fmt_num(v, nd: int = 2) -> str:
    try:
        return f"{float(v):,.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def get_fundamentals(ticker: str, asset_class: str) -> FundamentalsReport:
    """Build a fundamentals report for the given instrument."""
    rep = FundamentalsReport(name=ticker)

    if asset_class in ("Commodities",):
        rep.applicable = False
        rep.note = (
            "Commodity futures have no corporate fundamentals. Drivers are supply/demand, "
            "inventories, weather, the USD and real rates — see the AI news & trends panel."
        )
        return rep

    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as exc:  # network / parse failure
        rep.applicable = False
        rep.note = f"Could not load fundamentals: {exc}"
        return rep

    if not info or info.get("quoteType") in ("CRYPTOCURRENCY",) or asset_class == "Crypto":
        # Crypto: limited fundamentals — surface market context only.
        rep.applicable = bool(info)
        rep.name = info.get("shortName") or ticker
        rep.note = (
            "Crypto assets have no earnings/balance-sheet fundamentals. Shown below is "
            "available market context; behavioural and on-chain/flow drivers matter more."
        )
        rep.groups["Market context"] = {
            "Market cap": _fmt_money(info.get("marketCap")),
            "Circulating supply": _fmt_num(info.get("circulatingSupply"), 0),
            "Volume (24h)": _fmt_money(info.get("volume24Hr") or info.get("volume")),
            "52w high / low": f"{_fmt_num(info.get('fiftyTwoWeekHigh'))} / {_fmt_num(info.get('fiftyTwoWeekLow'))}",
        }
        return rep

    rep.name = info.get("longName") or info.get("shortName") or ticker
    rep.sector = info.get("sector", "") or ""
    rep.industry = info.get("industry", "") or ""
    rep.summary = info.get("longBusinessSummary", "") or ""

    rep.groups["Valuation"] = {
        "Market cap": _fmt_money(info.get("marketCap")),
        "Trailing P/E": _fmt_num(info.get("trailingPE")),
        "Forward P/E": _fmt_num(info.get("forwardPE")),
        "Price / Book": _fmt_num(info.get("priceToBook")),
        "Price / Sales": _fmt_num(info.get("priceToSalesTrailing12Months")),
        "EV / EBITDA": _fmt_num(info.get("enterpriseToEbitda")),
    }
    rep.groups["Profitability"] = {
        "Gross margin": _fmt_pct(info.get("grossMargins")),
        "Operating margin": _fmt_pct(info.get("operatingMargins")),
        "Profit margin": _fmt_pct(info.get("profitMargins")),
        "Return on equity": _fmt_pct(info.get("returnOnEquity")),
        "Return on assets": _fmt_pct(info.get("returnOnAssets")),
    }
    rep.groups["Growth & income"] = {
        "Revenue growth (yoy)": _fmt_pct(info.get("revenueGrowth")),
        "Earnings growth (yoy)": _fmt_pct(info.get("earningsGrowth")),
        "Dividend yield": _fmt_pct(info.get("dividendYield")),
        "Payout ratio": _fmt_pct(info.get("payoutRatio")),
    }
    rep.groups["Balance sheet & risk"] = {
        "Debt / Equity": _fmt_num(info.get("debtToEquity")),
        "Current ratio": _fmt_num(info.get("currentRatio")),
        "Free cash flow": _fmt_money(info.get("freeCashflow")),
        "Total cash": _fmt_money(info.get("totalCash")),
        "Beta": _fmt_num(info.get("beta")),
    }

    target = info.get("targetMeanPrice")
    current = info.get("currentPrice") or info.get("regularMarketPrice")
    upside = "—"
    if target and current:
        try:
            upside = f"{(float(target)/float(current)-1)*100:+.1f}%"
        except (TypeError, ValueError, ZeroDivisionError):
            upside = "—"
    rep.analyst = {
        "Recommendation": str(info.get("recommendationKey", "—")).replace("_", " ").title(),
        "Analysts": _fmt_num(info.get("numberOfAnalystOpinions"), 0),
        "Mean target": _fmt_num(target),
        "Implied upside": upside,
        "Target range": f"{_fmt_num(info.get('targetLowPrice'))} – {_fmt_num(info.get('targetHighPrice'))}",
    }
    return rep
