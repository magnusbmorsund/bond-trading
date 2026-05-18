"""V2e signals — thin wrapper around strategies.sector_shared.signals.

V2e's config has W_SLOW_24M and W_SLOW_36M, which the shared composite_score
automatically detects to activate the NaN-safe supercycle blend.
"""
import configs.sector_v2e as config
from strategies.sector_shared.signals import (
    composite_score     as _cs,
    mom_12m,
    rolling_vol         as _rv,
    spy_regime          as _sr,
    rolling_corr_at_dates,
    resample_to_period_end,
    resample_to_month_end,
)


def composite_score(prices):
    return _cs(prices, config)


def rolling_vol(prices):
    return _rv(prices, config)


def spy_regime(prices):
    return _sr(prices, config)
