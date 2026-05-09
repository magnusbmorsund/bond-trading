"""
Shared download/cache logic for all sector rotation pipelines.

Each pipeline module calls load_all(config, cache_filename, label) — a 1-liner wrapper
is all that's needed per-version.
"""
import os
import logging
import pandas as pd
import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)


def _price_is_fresh(last_date: pd.Timestamp) -> bool:
    today     = pd.Timestamp.today().normalize()
    last_bday = pd.Timestamp(np.busday_offset(today.date(), 0, roll="backward"))
    return last_date.normalize() >= last_bday


def _fetch(config, path: str, label: str) -> pd.DataFrame:
    tickers = config.ALL_TICKERS
    raw     = yf.download(tickers, start=config.BACKTEST_START, auto_adjust=True, progress=False)
    prices  = raw["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    daily_ret = prices.pct_change()
    spikes    = daily_ret.abs() > config.PRICE_SPIKE_THRESHOLD
    for etf in spikes.columns:
        for dt in spikes[etf][spikes[etf]].index:
            logger.warning("Price spike: %s on %s moved %+.1f%%",
                           etf, dt.date(), daily_ret.loc[dt, etf] * 100)

    prices.to_csv(path)
    logger.info("Fetched %s prices: %d tickers × %d days", label, len(tickers), len(prices))
    return prices


def load_all(config, cache_filename: str, label: str, force: bool = False) -> pd.DataFrame:
    """Load (or force-refresh) sector prices; emit staleness warnings."""
    os.makedirs(config.DATA_DIR, exist_ok=True)
    path    = os.path.join(config.DATA_DIR, cache_filename)
    tickers = config.ALL_TICKERS

    if not force and os.path.exists(path):
        cached  = pd.read_csv(path, index_col=0, parse_dates=True)
        missing = set(tickers) - set(cached.columns)
        if not missing and _price_is_fresh(cached.index[-1]):
            prices = cached
        else:
            prices = _fetch(config, path, label)
    else:
        prices = _fetch(config, path, label)

    missing = set(tickers) - set(prices.columns)
    if missing:
        logger.warning("Missing tickers in price data: %s", sorted(missing))

    age = (pd.Timestamp.today() - prices.index[-1]).days
    if age > 3:
        logger.warning("%s price data last date is %s (%d days ago)",
                       label, prices.index[-1].date(), age)
    else:
        logger.info("%s price data: %s → %s  (%d tickers)",
                    label, prices.index[0].date(), prices.index[-1].date(), len(prices.columns))
    return prices
