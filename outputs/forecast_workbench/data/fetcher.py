"""
Data Fetcher — modular ingestion layer.

Equities / ETFs / Macro : yfinance
Crypto                  : yfinance (e.g. BTC-USD, ETH-USD)
                          Supports BTC/USDT style input — auto-converted.

Returns a standardised DataFrame with columns:
    Date | Open | High | Low | Close | Volume
"""
from __future__ import annotations

import warnings

import pandas as pd
import yfinance as yf

from config.settings import DEFAULT_INTERVAL, DEFAULT_PERIOD

warnings.filterwarnings("ignore", category=FutureWarning)


def _normalise_crypto_ticker(symbol: str) -> str:
    """
    Convert exchange-style tickers to Yahoo Finance format.
    Examples:
        BTC/USDT  -> BTC-USD
        ETH/USDT  -> ETH-USD
        SOL/USD   -> SOL-USD
        BTC-USD   -> BTC-USD  (no-op)
    """
    symbol = symbol.upper().strip()
    if "/" in symbol:
        base, quote = symbol.split("/", 1)
        quote = "USD" if quote in ("USDT", "USDC", "BUSD") else quote
        return f"{base}-{quote}"
    return symbol


class DataFetcher:
    """Unified entry point for all market data."""

    def fetch(
        self,
        ticker: str,
        asset_class: str,
        period: str = DEFAULT_PERIOD,
        interval: str = DEFAULT_INTERVAL,
    ) -> pd.DataFrame:
        """
        Route to yfinance for all asset classes.
        Crypto tickers are normalised from exchange format (BTC/USDT) to
        Yahoo Finance format (BTC-USD) automatically.

        Returns a clean OHLCV DataFrame indexed by Date.
        Raises ValueError if the data cannot be fetched.
        """
        if asset_class == "Crypto":
            ticker = _normalise_crypto_ticker(ticker)
        return self._fetch_yfinance(ticker, period, interval)

    def _fetch_yfinance(
        self,
        ticker: str,
        period: str,
        interval: str,
    ) -> pd.DataFrame:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            raise ValueError(
                f"No data returned for '{ticker}'. "
                "Check the ticker symbol and try again."
            )

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"
        return df[["Open", "High", "Low", "Close", "Volume"]].copy()
