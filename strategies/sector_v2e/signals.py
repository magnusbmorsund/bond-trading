"""
Multi-timescale price signals for the V2e sector rotation strategy.

V2e extends V2d by adding 24m and 36m momentum components to the slow blend,
targeting multi-year economic supercycles (commodity, tech, rate cycles).
The _weighted_blend helper handles warmup gracefully: cells where a long
lookback is not yet available (NaN) are excluded from that date's blend
rather than propagating NaN to the whole composite.
"""
import pandas as pd
import numpy as np
import configs.sector_v2e as config

_LOOKBACKS = {
    "1m":  21,
    "3m":  63,
    "6m":  126,
    "12m": 252,
    "18m": 378,
    "24m": 504,
    "36m": 756,
}


def _total_return(prices: pd.DataFrame, days: int) -> pd.DataFrame:
    shifted = prices.shift(days)
    return prices / shifted.replace(0, float("nan")) - 1


def _weighted_blend(pairs: list) -> pd.DataFrame:
    """Weighted average of (weight, DataFrame) pairs, ignoring NaN per cell.

    During the warmup period when long lookbacks aren't yet available,
    shorter lookbacks carry the full weight rather than poisoning the composite.
    """
    num, den = None, None
    for w, m in pairs:
        if w <= 0:
            continue
        valid = m.notna().astype(float)
        if num is None:
            num = (w * m).fillna(0.0)
            den = w * valid
        else:
            num = num + (w * m).fillna(0.0)
            den = den + w * valid
    if num is None:
        return pd.DataFrame()
    return num.div(den).where(den > 0)


def composite_score(prices: pd.DataFrame) -> pd.DataFrame:
    mom_1m  = _total_return(prices, _LOOKBACKS["1m"])
    mom_3m  = _total_return(prices, _LOOKBACKS["3m"])
    mom_6m  = _total_return(prices, _LOOKBACKS["6m"])
    mom_12m = _total_return(prices, _LOOKBACKS["12m"])
    mom_18m = _total_return(prices, _LOOKBACKS["18m"])
    mom_24m = _total_return(prices, _LOOKBACKS["24m"])
    mom_36m = _total_return(prices, _LOOKBACKS["36m"])

    slow = _weighted_blend([
        (config.W_SLOW_12M, mom_12m),
        (config.W_SLOW_18M, mom_18m),
        (config.W_SLOW_24M, mom_24m),
        (config.W_SLOW_36M, mom_36m),
    ])
    fast = _weighted_blend([
        (config.W_FAST_1M, mom_1m),
        (config.W_FAST_3M, mom_3m),
    ])
    accel = (mom_3m - mom_6m).clip(lower=0)

    return (
        config.ALPHA_SLOW * slow
        + (1.0 - config.ALPHA_SLOW) * fast
        + config.BETA_ACCEL * accel
    )


def mom_12m(prices: pd.DataFrame) -> pd.DataFrame:
    return _total_return(prices, _LOOKBACKS["12m"])


def rolling_vol(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().rolling(config.VOL_WINDOW).std() * np.sqrt(252)


def spy_regime(prices: pd.DataFrame) -> pd.Series:
    if config.SPY_TICKER not in prices.columns:
        return pd.Series(1.0, index=prices.index, name="spy_regime")

    spy = prices[config.SPY_TICKER]
    ma  = spy.rolling(config.SPY_MA_WINDOW, min_periods=config.SPY_MA_WINDOW // 2).mean()
    raw = (spy >= ma).astype(float)

    if config.SPY_MA_CONFIRM <= 1:
        return raw.rename("spy_regime")

    confirmed    = raw.copy()
    below_streak = 0
    for i in range(len(raw)):
        if raw.iloc[i] == 0:
            below_streak += 1
        else:
            below_streak = 0
        if below_streak < config.SPY_MA_CONFIRM:
            confirmed.iloc[i] = 1.0

    return confirmed.rename("spy_regime")


def rolling_corr_at_dates(prices: pd.DataFrame, dates, window: int) -> dict:
    """Return {date: correlation_matrix} sampled at each rebalance date."""
    daily_ret = prices.pct_change()
    idx = prices.index
    result = {}
    for d in dates:
        loc = idx.get_loc(d)
        start = max(0, loc - window + 1)
        result[d] = daily_ret.iloc[start : loc + 1].corr()
    return result


def resample_to_period_end(df: pd.DataFrame, freq: str = "ME") -> pd.DataFrame:
    resampled = df.resample(freq).last()
    if freq.startswith("W"):
        actual_dates = (
            pd.Series(df.index, index=df.index)
            .resample(freq)
            .last()
            .dropna()
        )
        common = resampled.index.intersection(actual_dates.index)
        resampled = resampled.loc[common]
        resampled.index = pd.DatetimeIndex(actual_dates.loc[common].values)
    return resampled


def resample_to_month_end(df: pd.DataFrame) -> pd.DataFrame:
    return resample_to_period_end(df, "ME")
