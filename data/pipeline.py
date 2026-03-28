"""
Orchestrates fetching of all data needed for the strategy.
Run directly:  python -m data.pipeline
"""
import pandas as pd

from data.fred_client import fetch_all
from data.price_client import fetch_prices
from config import BACKTEST_START


def load_all(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        macro  : DataFrame of FRED macro signals (daily, ffilled)
        prices : DataFrame of ETF adjusted closes (daily)
    """
    print("Loading FRED macro data...")
    macro = fetch_all(start=BACKTEST_START, force=force)

    print("Loading ETF prices...")
    prices = fetch_prices(start=BACKTEST_START, force=force)

    # Align on common dates
    common = macro.index.intersection(prices.index)
    macro  = macro.loc[common]
    prices = prices.loc[common]

    return macro, prices


if __name__ == "__main__":
    macro, prices = load_all(force=True)
    print(f"\nMacro data:  {macro.shape}  ({macro.index[0].date()} → {macro.index[-1].date()})")
    print(f"Price data:  {prices.shape}  ({prices.index[0].date()} → {prices.index[-1].date()})")
    print("\nMacro columns:", list(macro.columns))
    print("ETF columns:  ", list(prices.columns))
    print("\nLatest macro snapshot:")
    print(macro.tail(1).T)
