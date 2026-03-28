"""
Compute monthly signals from macro data and ETF prices.

Three macro signals (from FRED):
  1. yield_curve  — z-score of 2s10s spread → duration preference
  2. credit       — inverted z-score of HY OAS → risk-on/risk-off
  3. inflation    — rate of change of 10Y breakeven → nominal vs TIPS

One price signal:
  4. momentum     — 12-1 month return per ETF → inclusion filter
"""
import pandas as pd
import numpy as np

from config import LOOKBACK_SIGNAL, LOOKBACK_VOL, MOMENTUM_WINDOW, MOMENTUM_SKIP


# ---------------------------------------------------------------------------
# Macro signals
# ---------------------------------------------------------------------------

def yield_curve_signal(macro: pd.DataFrame) -> pd.Series:
    """
    Rolling z-score of 2s10s spread (T10Y2Y).
    High  → steep curve → favor long duration (TLT/IEF)
    Low   → flat/inverted → favor short duration (SHY)
    """
    s = macro["spread_2s10s"].dropna()
    mu = s.rolling(LOOKBACK_SIGNAL).mean()
    sd = s.rolling(LOOKBACK_SIGNAL).std()
    return ((s - mu) / sd).rename("yield_curve")


def credit_signal(macro: pd.DataFrame) -> pd.Series:
    """
    Inverted rolling z-score of HY OAS (BAMLH0A0HYM2).
    Positive → tight spreads → risk-on (include LQD/HYG)
    Negative → wide spreads  → risk-off (go to Treasuries)
    """
    s = macro["hy_oas"].dropna()
    mu = s.rolling(LOOKBACK_SIGNAL).mean()
    sd = s.rolling(LOOKBACK_SIGNAL).std()
    return (-(s - mu) / sd).rename("credit")


def inflation_signal(macro: pd.DataFrame) -> pd.Series:
    """
    3-month rate of change of 10Y breakeven inflation.
    Positive → rising inflation expectations → favor TIP
    Negative → falling expectations → favor nominal Treasuries
    """
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
    """
    12-1 month price momentum for each ETF.
    Positive → include in portfolio.
    Negative → exclude (cash stays in remaining ETFs).
    """
    past   = prices.shift(MOMENTUM_SKIP)
    longer = prices.shift(MOMENTUM_WINDOW)
    return (past / longer - 1).rename(columns=lambda c: c)


def rolling_vol(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Annualised rolling daily return volatility (3-month window).
    Used for inverse-vol weighting.
    """
    daily_ret = prices.pct_change()
    return daily_ret.rolling(LOOKBACK_VOL).std() * np.sqrt(252)


# ---------------------------------------------------------------------------
# Resample to month-end
# ---------------------------------------------------------------------------

def resample_to_month_end(df: pd.DataFrame) -> pd.DataFrame:
    """Take last available observation each calendar month."""
    return df.resample("ME").last()
