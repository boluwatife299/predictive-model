"""
AI narrative layer — provider-agnostic. Supports Google Gemini and Anthropic
Claude; selected via the AI_PROVIDER secret (default: "gemini").

Generates a live "daily news & trends" market brief for the selected instrument
using the provider's web/search-grounding, weaving in the rule-based signals
(regime, tailwinds, headwinds, drivers) the rest of the app computed.

Why Gemini by default: its native Google Search grounding is a strong fit for
daily-news synthesis and its free tier is generous (gemini-2.5-flash).

Configuration (Streamlit secrets or env):
    AI_PROVIDER        "gemini" (default) or "anthropic"
    # Gemini
    GEMINI_API_KEY     required when AI_PROVIDER=gemini
    GEMINI_MODEL       optional (default: gemini-2.5-flash)
    # Anthropic
    ANTHROPIC_API_KEY  required when AI_PROVIDER=anthropic
    AI_MODEL           optional (default: claude-opus-4-8)

Design:
* `generate_market_brief` is the only entry point the UI calls.
* Graceful degradation: `is_available()` checks the configured provider's key
  and SDK; if missing, the UI hides the feature — nothing crashes.
* Cost control: this is a billed/quota'd call, so the UI only runs it behind an
  explicit button, and results are cached per ticker.
"""
from __future__ import annotations

import os

_DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
_DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-8"


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


def _provider() -> str:
    return (_secret("AI_PROVIDER") or "gemini").strip().lower()


def is_available() -> bool:
    """True when the configured provider's SDK is importable and key is set."""
    provider = _provider()
    if provider == "anthropic":
        if not _secret("ANTHROPIC_API_KEY"):
            return False
        try:
            import anthropic  # noqa: F401
        except Exception:
            return False
        return True
    # default: gemini
    if not _secret("GEMINI_API_KEY"):
        return False
    try:
        from google import genai  # noqa: F401
    except Exception:
        return False
    return True


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {x}" for x in items) if items else "- (none detected)"


def _build_prompt(
    *,
    ticker: str,
    asset_class: str,
    display_name: str,
    regime: str,
    tailwinds: list[str],
    headwinds: list[str],
    driver_categories: dict[str, list[str]] | None,
) -> tuple[str, str]:
    """Return (system_instruction, user_prompt) shared across providers."""
    system = (
        "You are a markets analyst writing a concise, balanced daily brief for a "
        "quantitative forecasting workbench. Search the web for the most recent "
        "(last ~2 weeks) news, catalysts and price-relevant developments for the "
        "instrument. Be specific and cite what you find. Do not give investment "
        "advice or price targets — describe drivers and context. Output "
        "GitHub-flavoured markdown with these sections:\n"
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
    if driver_categories:
        lines = [f"**{cat}:** " + "; ".join(f) for cat, f in driver_categories.items()]
        user += "\nKnown structural drivers for this asset:\n" + "\n".join(lines) + "\n"
    user += (
        "\nSearch the web for current news and write the brief. Connect the "
        "headlines to the technical read where relevant."
    )
    return system, user


def _gemini_brief(system: str, user: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=_secret("GEMINI_API_KEY"))
    config = types.GenerateContentConfig(
        system_instruction=system,
        tools=[types.Tool(google_search=types.GoogleSearch())],
        max_output_tokens=2000,
    )
    response = client.models.generate_content(
        model=_secret("GEMINI_MODEL") or _DEFAULT_GEMINI_MODEL,
        contents=user,
        config=config,
    )
    return (response.text or "").strip() or "_The AI returned no text for this instrument._"


def _anthropic_brief(system: str, user: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=_secret("ANTHROPIC_API_KEY"))
    model = _secret("AI_MODEL") or _DEFAULT_ANTHROPIC_MODEL
    messages = [{"role": "user", "content": user}]
    tools = [{"type": "web_search_20260209", "name": "web_search"}]

    # Server-side web search runs its own loop; pause_turn means "resume".
    response = None
    for _ in range(6):
        response = client.messages.create(
            model=model, max_tokens=2000, system=system, messages=messages, tools=tools
        )
        if response.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": response.content})
            continue
        break
    if response is None:
        return "_No response from the AI service._"
    text = "\n".join(
        b.text for b in response.content if getattr(b, "type", "") == "text"
    ).strip()
    return text or "_The AI returned no text for this instrument._"


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
    search. Raises if the configured provider's SDK/key are missing
    (call is_available() first).
    """
    system, user = _build_prompt(
        ticker=ticker,
        asset_class=asset_class,
        display_name=display_name,
        regime=regime,
        tailwinds=tailwinds,
        headwinds=headwinds,
        driver_categories=driver_categories,
    )
    if _provider() == "anthropic":
        return _anthropic_brief(system, user)
    return _gemini_brief(system, user)
