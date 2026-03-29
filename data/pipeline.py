"""
Orchestrates fetching of all data needed for the strategy.
Run directly:  python -m data.pipeline
"""
import logging
import pandas as pd

from data.fred_client  import fetch_all
from data.price_client import fetch_prices, fetch_vix
from config import BACKTEST_START, FRED_SERIES, ETF_UNIVERSE

logger = logging.getLogger(__name__)

# Columns downstream code requires from the macro DataFrame
_REQUIRED_MACRO_COLS = {"duration_z", "credit_z", "inflation_z", "vix_raw"}
# Signals that feed into composites — warn if any are absent
_EXPECTED_FRED_LABELS = set(FRED_SERIES.keys())


def _report_data_age(df: pd.DataFrame, label: str) -> None:
    last = df.index[-1]
    age  = (pd.Timestamp.today() - last).days
    if age > 3:
        logger.warning("%s last date is %s (%d days ago)", label, last.date(), age)
    else:
        logger.info("%s last date: %s (%d day(s) ago)", label, last.date(), age)


def load_all(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        macro  : DataFrame of FRED macro signals + VIX (daily, ffilled)
        prices : DataFrame of ETF adjusted closes (daily)
    """
    logger.info("Loading FRED macro data...")
    macro = fetch_all(start=BACKTEST_START, force=force)

    # Warn about any missing FRED labels that signals.py will silently skip
    missing_labels = _EXPECTED_FRED_LABELS - set(macro.columns)
    if missing_labels:
        logger.warning("Missing FRED labels in macro DataFrame: %s", sorted(missing_labels))

    logger.info("Loading ETF prices...")
    prices = fetch_prices(start=BACKTEST_START, force=force)

    # Warn about any missing ETFs
    missing_etfs = set(ETF_UNIVERSE) - set(prices.columns)
    if missing_etfs:
        logger.warning("Missing ETFs in price DataFrame: %s", sorted(missing_etfs))

    logger.info("Loading VIX...")
    vix = fetch_vix(start=BACKTEST_START, force=force)

    # Merge VIX into macro
    macro = macro.copy()
    macro["vix"] = vix.reindex(macro.index).ffill()

    # Align on common dates
    common = macro.index.intersection(prices.index)
    macro  = macro.loc[common]
    prices = prices.loc[common]

    _report_data_age(macro,  "Macro")
    _report_data_age(prices, "Prices")

    logger.info(
        "Data loaded — macro=%s  prices=%s  range=%s → %s",
        macro.shape, prices.shape,
        macro.index[0].date(), macro.index[-1].date(),
    )
    return macro, prices


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s  %(message)s")
    macro, prices = load_all(force=True)
    print(f"\nMacro data:  {macro.shape}  ({macro.index[0].date()} → {macro.index[-1].date()})")
    print(f"Price data:  {prices.shape}  ({prices.index[0].date()} → {prices.index[-1].date()})")
    print("\nMacro columns:", list(macro.columns))
    print("ETF columns:  ", list(prices.columns))
    print("\nLatest macro snapshot:")
    print(macro.tail(1).T)
