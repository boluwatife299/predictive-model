"""
AI narrative layer — provider-agnostic, defaults to Claude (Anthropic).

Generates a live "daily news & trends" market brief for the selected instrument
using Claude with the server-side web-search tool, weaving in the rule-based
signals (regime, tailwinds, headwinds, drivers) the rest of the app computed.

Design notes
------------
* Provider-agnostic by intent: `generate_market_brief` is the only entry point
  the UI calls. Swapping to Gemini later means adding a branch here, not
  touching app.py.
* Graceful degradation: if the SDK isn't installed or no API key is configured,
  `is_available()` returns False and the UI hides the feature — nothing crashes.
* Cost control: this makes a billed API call (with web search), so the UI only
  ever calls it behind an explicit button, and results are cached per ticker.

Configuration (Streamlit secrets or env):
    ANTHROPIC_API_KEY   required to enable the feature
    AI_MODEL            optional model override (default: claude-opus-4-8;
                        set to "claude-sonnet-4-6" for ~5x cheaper calls)
"""
from __future__ import annotations

import os

# Default to the most capable model; overridable for cost via AI_MODEL.
_DEFAULT_MODEL = "claude-opus-4-8"


def _secret(name: str) -> str | None:
    """Read a config value from env first, then Streamlit secrets."""
    val = os.getenv(name)
    if val:
        return val
    try:
        import streamlit as st

        return st.secrets.get(name)  # type: ignore[return-value]
    except Exception:
        return None


def _api_key() -> str | None:
    return _secret("ANTHROPIC_API_KEY")


def _model() -> str:
    return _secret("AI_MODEL") or _DEFAULT_MODEL


def is_available() -> bool:
    """True when the Anthropic SDK is importable and an API key is configured."""
    if not _api_key():
        return False
    try:
        import anthropic  # noqa: F401
    except Exception:
        return False
    return True


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {x}" for x in items) if items else "- (none detected)"


def generate_market_brief(
    *,
    ticker: str,
    asset_class: str,
    display_name: str,
    regime: str,
    tailwinds: list[str],
    headwinds: list[str],
    driver_categories: dict[str, list[str]] | None = None,
) -> str:
    """
    Return a markdown market brief for the instrument, grounded in live web
    search. Raises if the SDK/key are missing (call is_available() first).
    """
    import anthropic

    client = anthropic.Anthropic(api_key=_api_key())

    drivers_block = ""
    if driver_categories:
        lines = []
        for cat, factors in driver_categories.items():
            lines.append(f"**{cat}:** " + "; ".join(factors))
        drivers_block = "\n".join(lines)

    system = (
        "You are a markets analyst writing a concise, balanced daily brief for a "
        "quantitative forecasting workbench. Use the web_search tool to find the "
        "most recent (last ~2 weeks) news, catalysts and price-relevant "
        "developments for the instrument. Be specific and cite what you find. Do "
        "not give investment advice or price targets — describe drivers and "
        "context. Output GitHub-flavoured markdown with these sections:\n"
        "### 📰 Recent news & catalysts\n"
        "### 🟢 Tailwinds\n"
        "### 🔴 Headwinds\n"
        "### 👀 What to watch next\n"
        "Keep it tight — a few bullets per section."
    )

    user = (
        f"Instrument: {display_name} ({ticker}) — asset class: {asset_class}.\n\n"
        f"The workbench already computed this technical read:\n"
        f"- Regime: {regime}\n"
        f"- Tailwinds (price action):\n{_bullets(tailwinds)}\n"
        f"- Headwinds (price action):\n{_bullets(headwinds)}\n"
    )
    if drivers_block:
        user += f"\nKnown structural drivers for this asset:\n{drivers_block}\n"
    user += (
        "\nSearch the web for current news and write the brief. Connect the "
        "headlines to the technical read where relevant."
    )

    messages = [{"role": "user", "content": user}]
    tools = [{"type": "web_search_20260209", "name": "web_search"}]

    # Server-side web search runs its own loop; it can return pause_turn when it
    # hits the per-turn iteration cap. Re-send to let it resume.
    response = None
    for _ in range(6):
        response = client.messages.create(
            model=_model(),
            max_tokens=2000,
            system=system,
            messages=messages,
            tools=tools,
        )
        if response.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": response.content})
            continue
        break

    if response is None:
        return "_No response from the AI service._"

    text = "\n".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    ).strip()
    return text or "_The AI returned no text for this instrument._"
