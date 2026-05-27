"""V2f signals — thin wrapper around strategies.sector_shared.signals.

V2f shares all signal logic with V2e (24m/36m supercycle blend); only the
universe differs (XBI and IGV removed — no UCITS equivalents on Nordnet).
"""
import configs.sector_v2f as config
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
