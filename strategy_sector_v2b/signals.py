"""
Multi-timescale price signals for the V2b sector rotation strategy.
Identical logic to V2 — uses config_sector_v2b for parameters.
"""
import pandas as pd
import numpy as np
import config_sector_v2b as config

_LOOKBACKS = {
    "1m":  21,
    "3m":  63,
    "6m":  126,
    "12m": 252,
    "18m": 378,
}


def _total_return(prices: pd.DataFrame, days: int) -> pd.DataFrame:
    shifted = prices.shift(days)
    return prices / shifted.replace(0, float("nan")) - 1


def composite_score(prices: pd.DataFrame) -> pd.DataFrame:
    mom_1m  = _total_return(prices, _LOOKBACKS["1m"])
    mom_3m  = _total_return(prices, _LOOKBACKS["3m"])
    mom_6m  = _total_return(prices, _LOOKBACKS["6m"])
    mom_12m = _total_return(prices, _LOOKBACKS["12m"])
    mom_18m = _total_return(prices, _LOOKBACKS["18m"])

    w_slow = config.W_SLOW_12M + config.W_SLOW_18M
    w_fast = config.W_FAST_1M  + config.W_FAST_3M

    slow  = (config.W_SLOW_12M * mom_12m + config.W_SLOW_18M * mom_18m) / w_slow
    fast  = (config.W_FAST_1M  * mom_1m  + config.W_FAST_3M  * mom_3m ) / w_fast
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


def resample_to_period_end(df: pd.DataFrame, freq: str = "ME") -> pd.DataFrame:
    """
    Resample to period-end.

    For calendar-anchored freqs like "ME", pandas uses calendar dates (e.g.,
    Jan 31, Feb 28) as the index — some of these fall on weekends and won't
    match the trading-day price index, but ffill in the backtest covers the
    gap.  For week-anchored "W" the anchor is always Sunday, which is NEVER
    a trading day, so we must re-index to the actual last trading day instead.
    """
    resampled = df.resample(freq).last()
    if freq.startswith("W"):
        # Replace Sunday-anchor index with actual last trading day of each week.
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
