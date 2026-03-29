"""
Vectorised backtest engine with volatility targeting.

Weights are set at each month-end close, effective the next trading day.
Volatility targeting scales daily exposure so that realised portfolio vol
tracks config.VOL_TARGET — this is the main lever for hitting a return target.
"""
import logging
import pandas as pd
import numpy as np
import config

from strategy.signals   import (
    compute_all_macro, momentum, rolling_vol, resample_to_month_end
)
from strategy.portfolio import build_weight_series

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Volatility targeting
# ---------------------------------------------------------------------------

def _vol_scale(raw_returns: pd.Series) -> pd.Series:
    """
    Scale raw_returns so that realised vol tracks config.VOL_TARGET.
    Uses a LOOKBACK_VOL-day trailing vol estimate with a 1-day lag.
    Leverage is capped at config.MAX_LEVERAGE.
    """
    realised_vol = raw_returns.rolling(config.VOL_LOOKBACK).std() * np.sqrt(252)
    scale = (config.VOL_TARGET / realised_vol.shift(1)).clip(upper=config.MAX_LEVERAGE)
    return raw_returns * scale


# ---------------------------------------------------------------------------
# Drawdown overlay — ride upswings, step aside in downturns
# ---------------------------------------------------------------------------

def _drawdown_overlay(ret: pd.Series, cash_rate: pd.Series) -> pd.Series:
    """
    Asymmetric exposure scaling:
      - REDUCE to DD_SCALE when portfolio is in drawdown beyond DD_THRESHOLD
        AND the previous day's return was negative (momentum confirms downturn).
      - RE-ENTER FULLY as soon as the previous day's return turns positive
        (catch upswings immediately).

    Undeployed fraction earns the daily fed funds rate.
    """
    threshold = config.DD_THRESHOLD
    dd_scale  = config.DD_SCALE

    nav  = (1 + ret.fillna(0)).cumprod()
    peak = nav.cummax()
    dd   = (nav - peak) / peak

    dd_lag  = dd.shift(1).fillna(0)
    ret_lag = ret.shift(1).fillna(0)

    in_distress = (dd_lag <= threshold) & (ret_lag < 0)
    scale = pd.Series(np.where(in_distress, dd_scale, 1.0), index=ret.index)

    # Log drawdown overlay transitions (enter / exit)
    transitions = in_distress.astype(int).diff().fillna(0)
    entries = transitions[transitions == 1].index
    exits   = transitions[transitions == -1].index
    for dt in entries:
        logger.warning(
            "DD overlay ENTER: %s  dd=%.1f%%  (threshold=%.1f%%)",
            dt.date(), dd_lag.loc[dt] * 100, threshold * 100,
        )
    for dt in exits:
        logger.info("DD overlay EXIT:  %s  (resuming full exposure)", dt.date())

    daily_cash = cash_rate.reindex(ret.index).ffill().fillna(0) / 100 / 252
    overlay    = ret * scale + daily_cash * (1.0 - scale)
    return overlay


# ---------------------------------------------------------------------------
# Per-position trailing stops (commodity bucket)
# ---------------------------------------------------------------------------

def _apply_trailing_stops(daily_w: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    For each commodity ETF (HEDGE_ETFS), if today's price is more than
    TRAILING_STOP_PCT below its TRAILING_STOP_WINDOW-day rolling peak,
    zero that position and redirect the freed weight to SHY.

    Applied daily — exits far faster than the monthly momentum rebalance,
    cutting intra-month drawdowns in trending-down commodity regimes.
    """
    stop_pct = config.TRAILING_STOP_PCT
    window   = config.TRAILING_STOP_WINDOW

    hedge_cols = [e for e in config.HEDGE_ETFS if e in prices.columns]
    if not hedge_cols:
        return daily_w

    hedge_prices = prices[hedge_cols].reindex(daily_w.index).ffill()
    rolling_peak = hedge_prices.rolling(window, min_periods=1).max().shift(1)
    stop_trigger = hedge_prices < rolling_peak * (1.0 - stop_pct)

    w = daily_w.copy()
    for etf in hedge_cols:
        if etf not in w.columns:
            continue
        triggered = stop_trigger[etf].reindex(w.index).fillna(False)
        freed      = w[etf].where(triggered, 0.0)
        w[etf]     = w[etf].where(~triggered, 0.0)
        if "SHY" in w.columns:
            w["SHY"] = w["SHY"] + freed

        # Log first trigger date per ETF (avoid log flood for multi-year stop periods)
        trigger_dates = triggered[triggered].index
        if len(trigger_dates):
            first = trigger_dates[0]
            last  = trigger_dates[-1]
            n     = len(trigger_dates)
            logger.info(
                "Trailing stop: %s triggered on %d days  (first=%s, last=%s)",
                etf, n, first.date(), last.date(),
            )

    return w


# ---------------------------------------------------------------------------
# Live effective weights (for production use)
# ---------------------------------------------------------------------------

def effective_weights(signal_weights: pd.Series, recent_prices: pd.DataFrame) -> pd.Series:
    """
    Apply trailing stops to a single set of signal weights against recent prices.

    Use this when you want today's actionable positions, not historical backtest weights.
    Any commodity ETF that has dropped > TRAILING_STOP_PCT below its
    TRAILING_STOP_WINDOW-day rolling peak is zeroed out; freed weight goes to SHY.

    Parameters
    ----------
    signal_weights : Series indexed by ETF ticker (latest monthly model weights)
    recent_prices  : DataFrame of recent ETF prices (needs at least TRAILING_STOP_WINDOW rows)

    Returns
    -------
    Series of effective weights (same index as signal_weights)
    """
    w        = signal_weights.copy()
    stop_pct = config.TRAILING_STOP_PCT
    window   = config.TRAILING_STOP_WINDOW

    for etf in config.HEDGE_ETFS:
        if etf not in w.index or w[etf] <= 0:
            continue
        if etf not in recent_prices.columns:
            continue
        prices_etf = recent_prices[etf].dropna()
        if len(prices_etf) < 2:
            continue
        peak  = prices_etf.iloc[-window:].max()
        today = prices_etf.iloc[-1]
        if today < peak * (1.0 - stop_pct):
            pct_below = (peak - today) / peak
            logger.warning(
                "Trailing stop active: %s is %.1f%% below %d-day peak "
                "(today=%.2f  peak=%.2f) — position zeroed, weight moved to SHY",
                etf, pct_below * 100, window, today, peak,
            )
            freed = w[etf]
            w[etf] = 0.0
            if "SHY" in w.index:
                w["SHY"] = w["SHY"] + freed

    return w


# ---------------------------------------------------------------------------
# Main backtest
# ---------------------------------------------------------------------------

def run(macro: pd.DataFrame, prices: pd.DataFrame) -> dict:
    """
    Run the full backtest.

    Parameters
    ----------
    macro  : daily FRED macro + VIX DataFrame (from data.pipeline)
    prices : daily ETF adjusted-close DataFrame

    Returns
    -------
    dict with keys: weights, daily_returns, daily_returns_bm,
                    nav, nav_bm, turnover, diagnostics
    """
    # ── 1. Compute daily signals ───────────────────────────────────────────
    macro_signals = compute_all_macro(macro)
    mom_daily     = momentum(prices[config.ETF_UNIVERSE])
    vol_daily     = rolling_vol(prices[config.ETF_UNIVERSE])

    # ── 2. Resample to month-end ───────────────────────────────────────────
    macro_m = resample_to_month_end(macro_signals)
    mom_m   = resample_to_month_end(mom_daily)
    vol_m   = resample_to_month_end(vol_daily)

    common  = macro_m.index.intersection(mom_m.index).intersection(vol_m.index)
    macro_m = macro_m.loc[common]
    mom_m   = mom_m.loc[common]
    vol_m   = vol_m.loc[common]

    # ── 3. Build monthly target weights ────────────────────────────────────
    weights = build_weight_series(macro_m, mom_m, vol_m)

    # ── 4. Daily strategy returns (pre vol-target) ─────────────────────────
    daily_ret = prices[config.ETF_UNIVERSE].pct_change()
    # Fix: .reindex().ffill() instead of deprecated .reindex(method="ffill")
    daily_w   = weights.reindex(daily_ret.index).ffill().shift(1)

    # ── 4b. Per-position trailing stops (commodity ETFs only) ──────────────
    daily_w   = _apply_trailing_stops(daily_w, prices[config.ETF_UNIVERSE])

    raw_daily = (daily_w * daily_ret).sum(axis=1)
    raw_daily.name = "strategy_raw"

    # ── 5. Apply volatility targeting ─────────────────────────────────────
    vol_scaled           = _vol_scale(raw_daily)

    # ── 5b. Apply drawdown overlay (ride upswings, step aside in downturns)
    cash_rate            = macro["fedfunds"] if "fedfunds" in macro.columns else pd.Series(0.0, index=macro.index)
    strategy_daily       = _drawdown_overlay(vol_scaled, cash_rate)
    strategy_daily.name  = "strategy"

    # ── 6. Equal-weight benchmark (no vol targeting) ───────────────────────
    bm_w            = pd.Series(1 / len(config.ETF_UNIVERSE), index=config.ETF_UNIVERSE)
    benchmark_daily = (daily_ret * bm_w).sum(axis=1)
    benchmark_daily.name = "benchmark_ew"

    # ── 7. NAV ────────────────────────────────────────────────────────────
    nav    = (1 + strategy_daily.fillna(0)).cumprod()
    nav_bm = (1 + benchmark_daily.fillna(0)).cumprod()

    # ── 8. Turnover ────────────────────────────────────────────────────────
    prev_w   = weights.shift(1).fillna(0)
    turnover = (weights - prev_w).abs().sum(axis=1) / 2

    # ── 9. Diagnostics ─────────────────────────────────────────────────────
    diagnostics = pd.concat([macro_m, weights], axis=1)

    return {
        "weights":          weights,
        "daily_returns":    strategy_daily,
        "daily_returns_bm": benchmark_daily,
        "nav":              nav,
        "nav_bm":           nav_bm,
        "turnover":         turnover,
        "diagnostics":      diagnostics,
    }
