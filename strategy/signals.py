"""
Compute monthly signals from macro data and ETF prices.

Three macro signals (from FRED):
  1. yield_curve  — z-score of 2s10s spread → duration preference
  2. credit       — inverted z-score of HY OAS → risk-on/risk-off
  3. inflation    — rate of change of 10Y breakeven → nominal vs TIPS

One price signal:
  4. momentum     — 12-1 month return per ETF → inclusion filter

All lookback parameters are read from `config` at call time so that
optimize.py can monkey-patch them between Optuna trials.
"""
import pandas as pd
import numpy as np
import config


# ---------------------------------------------------------------------------
# Macro signals
# ---------------------------------------------------------------------------

def yield_curve_signal(macro: pd.DataFrame) -> pd.Series:
    s  = macro["spread_2s10s"].dropna()
    mu = s.rolling(config.LOOKBACK_SIGNAL).mean()
    sd = s.rolling(config.LOOKBACK_SIGNAL).std()
    return ((s - mu) / sd).rename("yield_curve")


def credit_signal(macro: pd.DataFrame) -> pd.Series:
    s  = macro["hy_oas"].dropna()
    mu = s.rolling(config.LOOKBACK_SIGNAL).mean()
    sd = s.rolling(config.LOOKBACK_SIGNAL).std()
    return (-(s - mu) / sd).rename("credit")


def inflation_signal(macro: pd.DataFrame) -> pd.Series:
    s = macro["breakeven_10y"].dropna()
    return s.pct_change(63).rename("inflation")


def compute_all_macro(macro: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with all three macro signals, daily."""
    yc  = yield_curve_signal(macro)
    cr  = credit_signal(macro)
    inf = inflation_signal(macro)
    return pd.concat([yc, cr, inf], axis=1).ffill()


# ---------------------------------------------------------------------------
# Price-based signals
# ---------------------------------------------------------------------------

def momentum(prices: pd.DataFrame) -> pd.DataFrame:
    past   = prices.shift(config.MOMENTUM_SKIP)
    longer = prices.shift(config.MOMENTUM_WINDOW)
    return (past / longer - 1).rename(columns=lambda c: c)


def rolling_vol(prices: pd.DataFrame) -> pd.DataFrame:
    daily_ret = prices.pct_change()
    return daily_ret.rolling(config.LOOKBACK_VOL).std() * np.sqrt(252)


# ---------------------------------------------------------------------------
# Resample to month-end
# ---------------------------------------------------------------------------

def resample_to_month_end(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("ME").last()
