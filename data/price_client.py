"""
Fetch and cache ETF price data from Yahoo Finance.
"""
import os
import logging
import pandas as pd
import numpy as np
import yfinance as yf

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import ETF_UNIVERSE, DATA_DIR, PRICE_SPIKE_THRESHOLD

logger = logging.getLogger(__name__)


def _price_is_fresh(last_date: pd.Timestamp) -> bool:
    """Cache is fresh if it covers the most recent completed trading day."""
    today = pd.Timestamp.today().normalize()
    last_bday = pd.Timestamp(np.busday_offset(today.date(), 0, roll="backward"))
    return last_date.normalize() >= last_bday


def _cache_path() -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, "etf_prices.csv")


def _sanity_check_prices(prices: pd.DataFrame) -> None:
    """Log warnings for suspicious price data."""
    # Check for zero or negative prices
    non_pos = (prices <= 0).any()
    for etf in non_pos[non_pos].index:
        n = (prices[etf] <= 0).sum()
        logger.warning("ETF %s has %d zero/negative price row(s) — data may be corrupt", etf, n)

    # Check for extreme single-day moves
    daily_ret = prices.pct_change()
    spikes = (daily_ret.abs() > PRICE_SPIKE_THRESHOLD)
    for etf in spikes.columns:
        spike_dates = spikes[etf][spikes[etf]].index
        if len(spike_dates):
            for dt in spike_dates:
                move = daily_ret.loc[dt, etf]
                logger.warning(
                    "Price spike: %s on %s moved %+.1f%% — check for split/error",
                    etf, dt.date(), move * 100,
                )

    # Check for stale tails (last price same for >5 consecutive days — possible feed issue)
    for etf in prices.columns:
        tail = prices[etf].dropna().tail(10)
        if len(tail) >= 5 and tail.nunique() == 1:
            logger.warning(
                "ETF %s: last 10 prices are all identical (%.2f) — possible stale feed",
                etf, tail.iloc[-1],
            )


def fetch_prices(start: str = "2000-01-01", force: bool = False) -> pd.DataFrame:
    """
    Return adjusted close prices for all ETFs as a DataFrame.
    Uses local cache; refreshes if stale (>1 trading day old).
    """
    tickers = ETF_UNIVERSE
    path = _cache_path()

    if not force and os.path.exists(path):
        cached = pd.read_csv(path, index_col=0, parse_dates=True)
        if _price_is_fresh(cached.index[-1]):
            return cached

    raw    = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    prices = raw["Close"]
    _sanity_check_prices(prices)
    prices.to_csv(path)
    logger.info("Fetched prices for %d tickers  (%d trading days)", len(tickers), len(prices))
    return prices


def fetch_vix(start: str = "2000-01-01", force: bool = False) -> pd.Series:
    """Fetch VIX index (^VIX) as a daily Series, with local cache."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "vix.csv")

    if not force and os.path.exists(path):
        cached = pd.read_csv(path, index_col=0, parse_dates=True).squeeze("columns")
        if _price_is_fresh(cached.index[-1]):
            return cached

    raw = yf.download("^VIX", start=start, auto_adjust=True, progress=False)
    vix = raw["Close"].squeeze()
    vix.name = "vix"
    vix.to_csv(path, header=True)
    logger.info("Fetched VIX  (%d trading days)", len(vix))
    return vix
