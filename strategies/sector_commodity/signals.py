"""Commodity-supercycle signals — thin wrapper around sector_shared.signals.

Reuses the V2e multi-timescale momentum composite (with the 24m/36m supercycle
lookbacks active via W_SLOW_24M / W_SLOW_36M). The ONE override is spy_regime:
commodities are an equity diversifier, so the SPY-trend cash gate is neutralised
— exposure is governed by momentum + the macro overlay, not by the equity trend.
"""
import pandas as pd

import configs.sector_commodity as config
from strategies.sector_shared.signals import (
    composite_score     as _cs,
    mom_12m,
    rolling_vol         as _rv,
    rolling_corr_at_dates,
    resample_to_period_end,
    resample_to_month_end,
)


def composite_score(prices):
    return _cs(prices, config)


def rolling_vol(prices):
    return _rv(prices, config)


def spy_regime(prices):
    """Always-invested regime — do NOT gate commodity exposure on the SPY trend."""
    return pd.Series(1.0, index=prices.index, name="spy_regime")
