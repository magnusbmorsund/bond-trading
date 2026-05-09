"""
Multi-timescale price signals for the V2 sector rotation strategy.

composite_score() blends five lookbacks (1M, 3M, 6M, 12M, 18M) into a
single ranking score per ETF:

  slow  = normalised weighted avg of 12M + 18M momentum  (supercycle detection)
  fast  = normalised weighted avg of 1M + 3M momentum    (tactical entry)
  accel = (mom_3m - mom_6m).clip(lower=0)               (positive acceleration)
  score = ALPHA_SLOW * slow + (1-ALPHA_SLOW) * fast + BETA_ACCEL * accel

NaN-safe: ETFs with insufficient price history simply receive NaN scores and
are excluded from selection naturally.
"""
import pandas as pd
import numpy as np
import configs.sector_v2 as config

# Trading-day approximations for each lookback
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
    """
    Multi-timescale composite momentum score — same index as prices.
    Higher = stronger momentum across all timescales.
    """
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
    """12M total return, used for adaptive trailing-stop classification."""
    return _total_return(prices, _LOOKBACKS["12m"])


def rolling_vol(prices: pd.DataFrame) -> pd.DataFrame:
    """Annualised rolling realised vol over VOL_WINDOW days."""
    return prices.pct_change().rolling(config.VOL_WINDOW).std() * np.sqrt(252)


def spy_regime(prices: pd.DataFrame) -> pd.Series:
    """
    Binary regime: 1 = risk-on (SPY above its MA), 0 = defensive.
    SPY_MA_CONFIRM consecutive days below MA required before flipping to 0.
    """
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


def resample_to_month_end(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("ME").last()
