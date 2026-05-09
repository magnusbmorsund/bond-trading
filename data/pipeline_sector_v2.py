"""
Data pipeline for the V2 sector rotation strategy.
No FRED required — pure price-based signals.
"""
import os
import logging
import pandas as pd
import numpy as np
import yfinance as yf

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config_sector_v2 as config

logger = logging.getLogger(__name__)


def _price_is_fresh(last_date: pd.Timestamp) -> bool:
    today     = pd.Timestamp.today().normalize()
    last_bday = pd.Timestamp(np.busday_offset(today.date(), 0, roll="backward"))
    return last_date.normalize() >= last_bday


def _cache_path() -> str:
    os.makedirs(config.DATA_DIR, exist_ok=True)
    return os.path.join(config.DATA_DIR, "sector_v2_prices.csv")


def _sanity_check(prices: pd.DataFrame) -> None:
    daily_ret = prices.pct_change()
    spikes    = daily_ret.abs() > config.PRICE_SPIKE_THRESHOLD
    for etf in spikes.columns:
        spike_dates = spikes[etf][spikes[etf]].index
        for dt in spike_dates:
            logger.warning("Price spike: %s on %s moved %+.1f%%",
                           etf, dt.date(), daily_ret.loc[dt, etf] * 100)


def fetch_prices(start: str = None, force: bool = False) -> pd.DataFrame:
    start   = start or config.BACKTEST_START
    tickers = config.ALL_TICKERS
    path    = _cache_path()

    if not force and os.path.exists(path):
        cached  = pd.read_csv(path, index_col=0, parse_dates=True)
        missing = set(tickers) - set(cached.columns)
        if not missing and _price_is_fresh(cached.index[-1]):
            return cached

    raw    = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    prices = raw["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    _sanity_check(prices)
    prices.to_csv(path)
    logger.info("Fetched V2 sector prices: %d tickers × %d days", len(tickers), len(prices))
    return prices


def load_all(force: bool = False) -> pd.DataFrame:
    prices  = fetch_prices(force=force)
    missing = set(config.ALL_TICKERS) - set(prices.columns)
    if missing:
        logger.warning("Missing tickers in price data: %s", sorted(missing))

    today = pd.Timestamp.today()
    age   = (today - prices.index[-1]).days
    if age > 3:
        logger.warning("V2 sector price data last date is %s (%d days ago)",
                       prices.index[-1].date(), age)
    else:
        logger.info("V2 sector price data: %s → %s  (%d tickers)",
                    prices.index[0].date(), prices.index[-1].date(), len(prices.columns))
    return prices


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s  %(message)s")
    prices = load_all(force=True)
    print(f"\nPrice data: {prices.shape}  ({prices.index[0].date()} → {prices.index[-1].date()})")
    print("Columns:", list(prices.columns))
    print("\nLatest prices:\n", prices.tail(2).T)
