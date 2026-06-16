"""
Structured price drivers for non-corporate assets.

Equities have fundamentals (P/E, margins, growth). Commodities, crypto, FX and
rates don't — their prices are moved by supply/demand, macro and flow factors.
This module gives those instruments a structured, rule-based "what moves this"
breakdown so the Validation tab shows something useful instead of just
"no corporate fundamentals". The AI layer can expand these with live news.
"""
from __future__ import annotations

from config.settings import COMMODITY_CATALOG

# Build symbol -> group ("Precious Metals", "Agriculture — …") lookup.
_SYMBOL_GROUP: dict[str, str] = {
    sym: group
    for group, items in COMMODITY_CATALOG.items()
    for _name, sym in items
}

_PRECIOUS_METALS = {
    "Demand": [
        "Investment / ETF flows and futures positioning",
        "Central-bank buying (especially gold)",
        "Industrial use (silver, platinum, palladium, copper)",
        "Jewellery and physical demand (India, China)",
    ],
    "Supply": [
        "Mine production and ore grades",
        "Recycling / scrap supply",
        "Above-ground inventories (COMEX/LBMA warehouse stocks)",
    ],
    "Macro": [
        "Real interest rates (inverse driver for gold)",
        "US dollar strength (inverse)",
        "Inflation expectations and safe-haven demand",
        "Geopolitical risk and financial-stress episodes",
    ],
}

_AGRICULTURE = {
    "Supply": [
        "Weather (drought, frost, excess rain) in key growing regions",
        "Planting intentions, acreage and yields",
        "USDA WASDE / crop-progress reports and global stocks-to-use",
    ],
    "Demand": [
        "Food and animal-feed consumption",
        "Biofuel demand (ethanol for corn, biodiesel for soybean/veg oils)",
        "Export demand and competing-origin pricing (e.g. Brazil, Black Sea)",
    ],
    "Macro / Cost": [
        "US dollar strength (inverse for exports)",
        "Energy and fertiliser costs",
        "Freight rates and trade policy / tariffs / export bans",
    ],
}

_CRYPTO = {
    "Network / adoption": [
        "On-chain activity, active addresses, fees",
        "Hash rate / staking participation and security",
        "Developer activity and protocol upgrades",
    ],
    "Flows": [
        "Spot-ETF net flows and institutional allocation",
        "Exchange reserves and stablecoin supply (liquidity)",
        "Derivatives funding rates and open interest",
    ],
    "Macro / cycle": [
        "Global liquidity, real rates and risk appetite",
        "Regulation and policy headlines",
        "Halving cycle (BTC) and token unlocks/emissions",
    ],
}

_FIXED_INCOME = {
    "Rates & policy": [
        "Central-bank policy path and forward guidance",
        "Inflation prints (CPI/PCE) and inflation expectations",
        "Growth and labour-market data",
    ],
    "Supply / technical": [
        "Government issuance and auction demand",
        "Term premium and curve shape",
        "Credit spreads (for HYG/LQD) and default expectations",
    ],
    "Macro": [
        "Risk sentiment / flight-to-quality flows",
        "US dollar and global rate differentials",
    ],
}

_MACRO = {
    "Drivers": [
        "Monetary policy and real interest rates",
        "US dollar trend",
        "Risk-on / risk-off sentiment (VIX, USO, GLD as proxies)",
        "Growth, inflation and geopolitical shocks",
    ],
}


def get_drivers(ticker: str, asset_class: str) -> dict[str, list[str]] | None:
    """
    Return a structured {category: [factors]} map of what moves this instrument,
    or None when corporate fundamentals are the better lens (equities).
    """
    if asset_class == "Commodities":
        group = _SYMBOL_GROUP.get(ticker, "")
        if group.startswith("Precious Metals"):
            return _PRECIOUS_METALS
        if group.startswith("Agriculture"):
            return _AGRICULTURE
        # Fallback for any commodity not in a known group.
        return _PRECIOUS_METALS
    if asset_class == "Crypto":
        return _CRYPTO
    if asset_class == "Fixed Income":
        return _FIXED_INCOME
    if asset_class == "Macro":
        return _MACRO
    return None  # Equities -> use fundamentals instead
