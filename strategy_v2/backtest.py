"""
v2 backtest engine — identical logic to strategy/backtest.py but wired to
config_v2 and strategy_v2 modules. No algorithmic differences; the v2
signals, portfolio weights, and ETF universe flow through automatically.
"""
import logging
import pandas as pd
import numpy as np
import config_v2 as config

from strategy_v2.signals   import (
    compute_all_macro, momentum, rolling_vol, resample_to_month_end
)
from strategy_v2.portfolio import build_weight_series

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Volatility targeting
# ---------------------------------------------------------------------------

def _vol_scale(raw_returns: pd.Series) -> pd.Series:
    realised_vol = raw_returns.rolling(config.VOL_LOOKBACK).std() * np.sqrt(252)
    scale = (config.VOL_TARGET / realised_vol.shift(1)).clip(upper=config.MAX_LEVERAGE)
    return raw_returns * scale


# ---------------------------------------------------------------------------
# Drawdown overlay
# ---------------------------------------------------------------------------

def _drawdown_overlay(ret: pd.Series, cash_rate: pd.Series) -> pd.Series:
    threshold = config.DD_THRESHOLD
    dd_scale  = config.DD_SCALE

    nav  = (1 + ret.fillna(0)).cumprod()
    peak = nav.cummax()
    dd   = (nav - peak) / peak

    dd_lag  = dd.shift(1).fillna(0)
    ret_lag = ret.shift(1).fillna(0)

    in_distress = (dd_lag <= threshold) & (ret_lag < 0)
    scale = pd.Series(np.where(in_distress, dd_scale, 1.0), index=ret.index)

    transitions = in_distress.astype(int).diff().fillna(0)
    for dt in transitions[transitions == 1].index:
        logger.warning(
            "DD overlay ENTER: %s  dd=%.1f%%  (threshold=%.1f%%)",
            dt.date(), dd_lag.loc[dt] * 100, threshold * 100,
        )
    for dt in transitions[transitions == -1].index:
        logger.info("DD overlay EXIT:  %s  (resuming full exposure)", dt.date())

    daily_cash = cash_rate.reindex(ret.index).ffill().fillna(0) / 100 / 252
    return ret * scale + daily_cash * (1.0 - scale)


# ---------------------------------------------------------------------------
# Per-position trailing stops (commodity bucket — now includes SLV)
# ---------------------------------------------------------------------------

def _apply_trailing_stops(daily_w: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
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

        trigger_dates = triggered[triggered].index
        if len(trigger_dates):
            logger.info(
                "Trailing stop: %s triggered on %d days  (first=%s, last=%s)",
                etf, len(trigger_dates), trigger_dates[0].date(), trigger_dates[-1].date(),
            )

    return w


# ---------------------------------------------------------------------------
# Live effective weights
# ---------------------------------------------------------------------------

def effective_weights(signal_weights: pd.Series, recent_prices: pd.DataFrame) -> pd.Series:
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
    daily_w   = weights.reindex(daily_ret.index).ffill().shift(1)

    # ── 4b. Per-position trailing stops ────────────────────────────────────
    daily_w   = _apply_trailing_stops(daily_w, prices[config.ETF_UNIVERSE])

    raw_daily = (daily_w * daily_ret).sum(axis=1)
    raw_daily.name = "strategy_raw"

    # ── 5. Volatility targeting ────────────────────────────────────────────
    vol_scaled = _vol_scale(raw_daily)

    # ── 5b. Drawdown overlay ───────────────────────────────────────────────
    cash_rate            = macro["fedfunds"] if "fedfunds" in macro.columns else pd.Series(0.0, index=macro.index)
    strategy_daily       = _drawdown_overlay(vol_scaled, cash_rate)
    strategy_daily.name  = "strategy"

    # ── 6. Equal-weight benchmark ──────────────────────────────────────────
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
