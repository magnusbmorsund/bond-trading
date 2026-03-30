"""
Shared backtest building blocks used by strategy/backtest.py, strategy_v2/backtest.py,
and strategy_v3/backtest.py. Extracted to eliminate ~95% code duplication across versions.

All functions are pure (no global config imports) — callers pass config values explicitly.
"""
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def vol_scale(
    raw_returns: pd.Series,
    vol_target: float,
    vol_lookback: int,
    max_leverage: float,
) -> pd.Series:
    """Scale raw_returns so realised vol tracks vol_target. Leverage capped at max_leverage."""
    realised_vol = raw_returns.rolling(vol_lookback).std() * np.sqrt(252)
    scale = (vol_target / realised_vol.shift(1)).clip(upper=max_leverage)
    return raw_returns * scale


def drawdown_overlay(
    ret: pd.Series,
    cash_rate: pd.Series,
    dd_threshold: float,
    dd_scale: float,
) -> pd.Series:
    """
    Reduce exposure to dd_scale when portfolio is in drawdown beyond dd_threshold
    AND prior day return was negative. Re-enter fully as soon as prior day turns positive.
    Undeployed fraction earns daily fed funds rate.
    """
    nav  = (1 + ret.fillna(0)).cumprod()
    peak = nav.cummax()
    dd   = (nav - peak) / peak

    dd_lag  = dd.shift(1).fillna(0)
    ret_lag = ret.shift(1).fillna(0)

    in_distress = (dd_lag <= dd_threshold) & (ret_lag < 0)
    scale = pd.Series(np.where(in_distress, dd_scale, 1.0), index=ret.index)

    transitions = in_distress.astype(int).diff().fillna(0)
    for dt in transitions[transitions == 1].index:
        logger.warning(
            "DD overlay ENTER: %s  dd=%.1f%%  (threshold=%.1f%%)",
            dt.date(), dd_lag.loc[dt] * 100, dd_threshold * 100,
        )
    for dt in transitions[transitions == -1].index:
        logger.info("DD overlay EXIT:  %s  (resuming full exposure)", dt.date())

    daily_cash = cash_rate.reindex(ret.index).ffill().fillna(0) / 100 / 252
    return ret * scale + daily_cash * (1.0 - scale)


def apply_trailing_stops(
    daily_w: pd.DataFrame,
    prices: pd.DataFrame,
    stop_etfs: list,
    stop_pct: float,
    stop_window: int,
) -> pd.DataFrame:
    """
    For each ETF in stop_etfs: if today's price is more than stop_pct below its
    stop_window-day rolling peak, zero that position and redirect freed weight to SHY.
    Applied daily — exits far faster than the monthly momentum rebalance.
    """
    stop_cols = [e for e in stop_etfs if e in prices.columns]
    if not stop_cols:
        return daily_w

    stop_prices  = prices[stop_cols].reindex(daily_w.index).ffill()
    rolling_peak = stop_prices.rolling(stop_window, min_periods=1).max().shift(1)
    stop_trigger = stop_prices < rolling_peak * (1.0 - stop_pct)

    w = daily_w.copy()
    for etf in stop_cols:
        if etf not in w.columns:
            continue
        triggered = stop_trigger[etf].reindex(w.index).fillna(False)
        freed      = w[etf].where(triggered, 0.0)
        w[etf]     = w[etf].where(~triggered, 0.0)
        if "SHY" in w.columns:
            w["SHY"] = w["SHY"] + freed

        trigger_dates = triggered[triggered].index
        if len(trigger_dates):
            logger.info(
                "Trailing stop: %s triggered on %d days  (first=%s, last=%s)",
                etf, len(trigger_dates), trigger_dates[0].date(), trigger_dates[-1].date(),
            )

    return w


def effective_weights_core(
    signal_weights: pd.Series,
    recent_prices: pd.DataFrame,
    stop_etfs: list,
    stop_pct: float,
    stop_window: int,
) -> pd.Series:
    """
    Apply trailing stops to a single set of signal weights against recent prices.
    Use for live/today's actionable positions, not historical backtest.
    Freed weight from any stopped-out ETF goes to SHY.
    """
    w = signal_weights.copy()

    for etf in stop_etfs:
        if etf not in w.index or w[etf] <= 0:
            continue
        if etf not in recent_prices.columns:
            continue
        prices_etf = recent_prices[etf].dropna()
        if len(prices_etf) < 2:
            continue
        peak  = prices_etf.iloc[-stop_window:].max()
        today = prices_etf.iloc[-1]
        if today < peak * (1.0 - stop_pct):
            pct_below = (peak - today) / peak
            logger.warning(
                "Trailing stop active: %s is %.1f%% below %d-day peak "
                "(today=%.2f  peak=%.2f) — position zeroed, weight moved to SHY",
                etf, pct_below * 100, stop_window, today, peak,
            )
            freed = w[etf]
            w[etf] = 0.0
            if "SHY" in w.index:
                w["SHY"] = w["SHY"] + freed

    return w
