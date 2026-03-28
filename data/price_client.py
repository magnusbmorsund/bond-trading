"""
Fetch and cache ETF price data from Yahoo Finance.
"""
import os
import pandas as pd
import yfinance as yf

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import ETF_UNIVERSE, DATA_DIR


def _cache_path() -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, "etf_prices.csv")


def fetch_prices(tickers: list = None, start: str = "2000-01-01", force: bool = False) -> pd.DataFrame:
    """
    Return adjusted close prices for all ETFs as a DataFrame.
    Uses local cache; refreshes if stale (>1 day old).
    """
    if tickers is None:
        tickers = ETF_UNIVERSE

    path = _cache_path()

    if not force and os.path.exists(path):
        cached = pd.read_csv(path, index_col=0, parse_dates=True)
        if (pd.Timestamp.today() - cached.index[-1]).days <= 1:
            return cached

    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    prices = raw["Close"]
    prices.to_csv(path)
    print(f"  Fetched prices for {tickers} ({len(prices)} trading days)")
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns."""
    return prices.apply(lambda col: col.dropna().pct_change()).reindex(prices.index)
