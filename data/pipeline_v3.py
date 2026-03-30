"""
v3 data pipeline — loads the expanded v3 ETF universe and FRED series.

Key differences from pipeline_v2.py:
  - ETF prices cached to etf_prices_v3.csv (separate from v1/v2 caches)
    to avoid contaminating earlier caches with the larger v3 universe.
  - FRED series fetched per-series via fred_client.fetch_series (same per-series
    cache files, safe to share with v1/v2 — each FRED series has its own file).
  - Fetches ^VIX3M (CBOE 3-Month VIX) and exposes it as macro["vix3m"].
  - New v3 series (EDV, JPST, DBMF, MTUM) fetched automatically.
  - Log label: "v3"
"""
import logging
import os
import pandas as pd
import numpy as np
import yfinance as yf

import config_v3 as config
from data.fred_client import fetch_series, FRED_API_KEY

logger = logging.getLogger(__name__)

_V3_PRICE_CACHE = os.path.join(config.DATA_DIR, "etf_prices_v3.csv")


def _price_is_fresh(last_date: pd.Timestamp) -> bool:
    today     = pd.Timestamp.today().normalize()
    last_bday = pd.Timestamp(np.busday_offset(today.date(), 0, roll="backward"))
    return last_date.normalize() >= last_bday


def _fetch_prices_v3(start: str, force: bool = False) -> pd.DataFrame:
    """
    Fetch adjusted close prices for the full v3 universe.
    Cached separately from v1/v2 (etf_prices_v3.csv).
    """
    os.makedirs(config.DATA_DIR, exist_ok=True)

    if not force and os.path.exists(_V3_PRICE_CACHE):
        cached = pd.read_csv(_V3_PRICE_CACHE, index_col=0, parse_dates=True)
        # Check all v3 tickers are present and cache is fresh
        missing = set(config.ETF_UNIVERSE) - set(cached.columns)
        if not missing and _price_is_fresh(cached.index[-1]):
            return cached

    logger.info("Downloading v3 ETF prices for %d tickers...", len(config.ETF_UNIVERSE))
    raw    = yf.download(
        config.ETF_UNIVERSE, start=start,
        auto_adjust=True, progress=False,
    )
    prices = raw["Close"]

    # Ensure all universe ETFs are present (NaN-fill pre-launch / unavailable tickers)
    for ticker in config.ETF_UNIVERSE:
        if ticker not in prices.columns:
            prices[ticker] = np.nan
            logger.warning("ETF %s unavailable from yfinance — column set to NaN", ticker)

    # Sanity check: warn on missing tickers
    missing = set(config.ETF_UNIVERSE) - set(prices.columns)
    if missing:
        logger.warning("Missing v3 ETFs in downloaded data: %s", sorted(missing))

    prices.to_csv(_V3_PRICE_CACHE)
    logger.info("v3 prices cached → %s  (%d rows, %d tickers)", _V3_PRICE_CACHE, len(prices), len(prices.columns))
    return prices


def _fetch_vix(start: str, force: bool = False) -> pd.Series:
    """Shared VIX fetch — reuses vix.csv cache (same data)."""
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


def _fetch_vix3m(start: str, force: bool = False) -> pd.Series:
    """
    Fetch CBOE 3-Month VIX (^VIX3M) — used for VIX term structure signal.
    Normal term structure: VIX3M > VIX (inverted spot/futures = contango).
    Inverted term structure (VIX > VIX3M): acute stress regime.
    Cached to vix3m.csv.
    """
    vix3m_path = os.path.join(config.DATA_DIR, "vix3m.csv")
    os.makedirs(config.DATA_DIR, exist_ok=True)

    if not force and os.path.exists(vix3m_path):
        cached = pd.read_csv(vix3m_path, index_col=0, parse_dates=True).squeeze("columns")
        if _price_is_fresh(cached.index[-1]):
            return cached

    try:
        raw = yf.download("^VIX3M", start=start, auto_adjust=True, progress=False)
        vix3m = raw["Close"].squeeze()
        vix3m.name = "vix3m"
        vix3m.to_csv(vix3m_path, header=True)
        logger.info("Fetched VIX3M  (%d trading days)", len(vix3m))
        return vix3m
    except Exception as exc:
        logger.warning("VIX3M fetch failed (%s) — vix_term_structure signal will be zeroed", exc)
        return pd.Series(dtype=float, name="vix3m")


def _fetch_all_fred_v3(start: str, force: bool = False) -> pd.DataFrame:
    """
    Fetch all v3 FRED series using fred_client.fetch_series (per-series cache).
    Safe to call alongside v1/v2 — each series has its own cache file.
    """
    frames = {}
    for label, series_id in config.FRED_SERIES.items():
        try:
            s = fetch_series(series_id, start=start, force=force)
            frames[label] = s
        except Exception as exc:
            logger.error("Skipping FRED:%s (%s) — %s", series_id, label, exc)

    if not frames:
        raise RuntimeError("All v3 FRED series failed. Check FRED_API_KEY.")

    missing = set(config.FRED_SERIES.keys()) - set(frames.keys())
    if missing:
        logger.warning("Missing v3 FRED series: %s", sorted(missing))

    df = pd.DataFrame(frames).ffill()
    return df


def load_all(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        macro  : DataFrame of FRED macro signals + VIX + VIX3M (daily, ffilled)
        prices : DataFrame of v3 ETF adjusted closes (daily)
    """
    logger.info("Loading v3 FRED macro data...")
    macro = _fetch_all_fred_v3(start=config.BACKTEST_START, force=force)

    missing_labels = set(config.FRED_SERIES.keys()) - set(macro.columns)
    if missing_labels:
        logger.warning("Missing FRED labels: %s", sorted(missing_labels))

    logger.info("Loading v3 ETF prices...")
    prices = _fetch_prices_v3(start=config.BACKTEST_START, force=force)

    missing_etfs = set(config.ETF_UNIVERSE) - set(prices.columns)
    if missing_etfs:
        logger.warning("Missing v3 ETFs in price data: %s", sorted(missing_etfs))

    logger.info("Loading VIX...")
    vix = _fetch_vix(start=config.BACKTEST_START, force=force)

    logger.info("Loading VIX3M...")
    vix3m = _fetch_vix3m(start=config.BACKTEST_START, force=force)

    macro = macro.copy()
    macro["vix"] = vix.reindex(macro.index).ffill()
    macro["vix3m"] = vix3m.reindex(macro.index).ffill()

    # Report VIX3M coverage so user knows if term structure signal is active
    vix3m_col = macro["vix3m"]
    if vix3m_col.isna().all():
        logger.warning("VIX3M column is all NaN — vix_term_structure signal disabled for entire backtest")
    else:
        first_valid = vix3m_col.first_valid_index()
        nan_count   = int(vix3m_col.isna().sum())
        if nan_count > 0:
            logger.info("VIX3M data starts %s — vix_term_structure inactive before that date", first_valid.date())

    common = macro.index.intersection(prices.index)
    macro  = macro.loc[common]
    prices = prices.loc[common]

    # Report data age
    for label, df in [("Macro v3", macro), ("Prices v3", prices)]:
        last = df.index[-1]
        age  = (pd.Timestamp.today() - last).days
        if age > 3:
            logger.warning("%s last date is %s (%d days ago)", label, last.date(), age)
        else:
            logger.info("%s last date: %s (%d day(s) ago)", label, last.date(), age)

    logger.info(
        "v3 data loaded — macro=%s  prices=%s  range=%s → %s",
        macro.shape, prices.shape,
        macro.index[0].date(), macro.index[-1].date(),
    )
    return macro, prices
