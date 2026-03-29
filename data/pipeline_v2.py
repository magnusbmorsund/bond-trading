"""
v2 data pipeline — loads the expanded v2 ETF universe and FRED series.

Key differences from pipeline.py:
  - ETF prices cached to etf_prices_v2.csv (separate from v1 cache)
    to avoid contaminating v1's cache with the larger v2 universe.
  - FRED series fetched per-series via fred_client.fetch_series (same per-series
    cache files, safe to share with v1 — each FRED series has its own file).
  - New v2 series (DTWEXBGS, IPMAN) fetched automatically.
"""
import logging
import os
import pandas as pd
import numpy as np
import yfinance as yf

import config_v2 as config
from data.fred_client import fetch_series, FRED_API_KEY

logger = logging.getLogger(__name__)

_V2_PRICE_CACHE = os.path.join(config.DATA_DIR, "etf_prices_v2.csv")


def _price_is_fresh(last_date: pd.Timestamp) -> bool:
    today     = pd.Timestamp.today().normalize()
    last_bday = pd.Timestamp(np.busday_offset(today.date(), 0, roll="backward"))
    return last_date.normalize() >= last_bday


def _fetch_prices_v2(start: str, force: bool = False) -> pd.DataFrame:
    """
    Fetch adjusted close prices for the full v2 universe.
    Cached separately from v1 (etf_prices_v2.csv).
    """
    os.makedirs(config.DATA_DIR, exist_ok=True)

    if not force and os.path.exists(_V2_PRICE_CACHE):
        cached = pd.read_csv(_V2_PRICE_CACHE, index_col=0, parse_dates=True)
        # Check all v2 tickers are present and cache is fresh
        missing = set(config.ETF_UNIVERSE) - set(cached.columns)
        if not missing and _price_is_fresh(cached.index[-1]):
            return cached

    logger.info("Downloading v2 ETF prices for %d tickers...", len(config.ETF_UNIVERSE))
    raw    = yf.download(
        config.ETF_UNIVERSE, start=start,
        auto_adjust=True, progress=False,
    )
    prices = raw["Close"]

    # Sanity check: warn on missing tickers
    missing = set(config.ETF_UNIVERSE) - set(prices.columns)
    if missing:
        logger.warning("Missing v2 ETFs in downloaded data: %s", sorted(missing))

    prices.to_csv(_V2_PRICE_CACHE)
    logger.info("v2 prices cached → %s  (%d rows, %d tickers)", _V2_PRICE_CACHE, len(prices), len(prices.columns))
    return prices


def _fetch_vix(start: str, force: bool = False) -> pd.Series:
    """Shared VIX fetch — reuses v1 vix.csv cache (same data)."""
    vix_path = os.path.join(config.DATA_DIR, "vix.csv")
    os.makedirs(config.DATA_DIR, exist_ok=True)

    if not force and os.path.exists(vix_path):
        cached = pd.read_csv(vix_path, index_col=0, parse_dates=True).squeeze("columns")
        if _price_is_fresh(cached.index[-1]):
            return cached

    raw = yf.download("^VIX", start=start, auto_adjust=True, progress=False)
    vix = raw["Close"].squeeze()
    vix.name = "vix"
    vix.to_csv(vix_path, header=True)
    logger.info("Fetched VIX  (%d trading days)", len(vix))
    return vix


def _fetch_all_fred_v2(start: str, force: bool = False) -> pd.DataFrame:
    """
    Fetch all v2 FRED series using fred_client.fetch_series (per-series cache).
    Safe to call alongside v1 — each series has its own cache file.
    """
    frames = {}
    for label, series_id in config.FRED_SERIES.items():
        try:
            s = fetch_series(series_id, start=start, force=force)
            frames[label] = s
        except Exception as exc:
            logger.error("Skipping FRED:%s (%s) — %s", series_id, label, exc)

    if not frames:
        raise RuntimeError("All v2 FRED series failed. Check FRED_API_KEY.")

    missing = set(config.FRED_SERIES.keys()) - set(frames.keys())
    if missing:
        logger.warning("Missing v2 FRED series: %s", sorted(missing))

    df = pd.DataFrame(frames).ffill()
    return df


def load_all(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        macro  : DataFrame of FRED macro signals + VIX (daily, ffilled)
        prices : DataFrame of v2 ETF adjusted closes (daily)
    """
    logger.info("Loading v2 FRED macro data...")
    macro = _fetch_all_fred_v2(start=config.BACKTEST_START, force=force)

    missing_labels = set(config.FRED_SERIES.keys()) - set(macro.columns)
    if missing_labels:
        logger.warning("Missing FRED labels: %s", sorted(missing_labels))

    logger.info("Loading v2 ETF prices...")
    prices = _fetch_prices_v2(start=config.BACKTEST_START, force=force)

    missing_etfs = set(config.ETF_UNIVERSE) - set(prices.columns)
    if missing_etfs:
        logger.warning("Missing v2 ETFs in price data: %s", sorted(missing_etfs))

    logger.info("Loading VIX...")
    vix = _fetch_vix(start=config.BACKTEST_START, force=force)

    macro = macro.copy()
    macro["vix"] = vix.reindex(macro.index).ffill()

    common = macro.index.intersection(prices.index)
    macro  = macro.loc[common]
    prices = prices.loc[common]

    # Report data age
    for label, df in [("Macro v2", macro), ("Prices v2", prices)]:
        last = df.index[-1]
        age  = (pd.Timestamp.today() - last).days
        if age > 3:
            logger.warning("%s last date is %s (%d days ago)", label, last.date(), age)
        else:
            logger.info("%s last date: %s (%d day(s) ago)", label, last.date(), age)

    logger.info(
        "v2 data loaded — macro=%s  prices=%s  range=%s → %s",
        macro.shape, prices.shape,
        macro.index[0].date(), macro.index[-1].date(),
    )
    return macro, prices
