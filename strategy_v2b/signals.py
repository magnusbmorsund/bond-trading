"""v2b signals — re-exports everything from v2, adds momentum_short."""
from strategy_v2.signals import *  # noqa: F401, F403
import pandas as pd
import config_v2b as config


def momentum_short(prices: pd.DataFrame) -> pd.DataFrame:
    """3-month price return (no skip) — used as the duration momentum gate."""
    longer = prices.shift(config.DURATION_MOM_LOOKBACK)
    return prices / longer.replace(0, float("nan")) - 1
