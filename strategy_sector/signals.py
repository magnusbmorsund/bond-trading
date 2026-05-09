"""
Price-based signals for the sector rotation strategy.

No FRED data required — all signals derive from ETF price history.

Signals:
  momentum()    — N-day total return, rank-based sector selection
  rolling_vol() — annualised realised vol for inverse-vol weighting
  spy_regime()  — 1 when SPY is above its MA (risk-on), 0 otherwise
"""
import pandas as pd
import numpy as np
import config_sector as config


def momentum(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Total return over MOMENTUM_LOOKBACK days (no skip).
    Returns NaN for rows with insufficient history.
    """
    longer = prices.shift(config.MOMENTUM_LOOKBACK)
    return prices / longer.replace(0, float("nan")) - 1


def rolling_vol(prices: pd.DataFrame) -> pd.DataFrame:
    """Annualised rolling realised vol over VOL_WINDOW days."""
    return prices.pct_change().rolling(config.VOL_WINDOW).std() * np.sqrt(252)


def spy_regime(prices: pd.DataFrame) -> pd.Series:
    """
    Binary regime filter: 1 = risk-on (SPY above MA), 0 = defensive.
    SPY_MA_CONFIRM consecutive days below the MA required before flipping to 0.
    This prevents whipsaw during brief dips below the MA.
    """
    if config.SPY_TICKER not in prices.columns:
        return pd.Series(1.0, index=prices.index, name="spy_regime")

    spy = prices[config.SPY_TICKER]
    ma  = spy.rolling(config.SPY_MA_WINDOW, min_periods=config.SPY_MA_WINDOW // 2).mean()
    raw = (spy >= ma).astype(float)

    if config.SPY_MA_CONFIRM <= 1:
        return raw.rename("spy_regime")

    # Require SPY_MA_CONFIRM consecutive below-MA days before switching to 0
    # (above-MA flips back to 1 immediately — conservative on exits, fast on re-entries)
    confirmed = raw.copy()
    below_streak = 0
    for i in range(len(raw)):
        if raw.iloc[i] == 0:
            below_streak += 1
        else:
            below_streak = 0
        if below_streak < config.SPY_MA_CONFIRM:
            confirmed.iloc[i] = 1.0   # not yet confirmed defensive

    return confirmed.rename("spy_regime")


def resample_to_month_end(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("ME").last()
