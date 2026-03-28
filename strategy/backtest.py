"""
Vectorised backtest engine with volatility targeting.

Weights are set at each month-end close, effective the next trading day.
Volatility targeting scales daily exposure so that realised portfolio vol
tracks config.VOL_TARGET — this is the main lever for hitting a return target.
"""
import pandas as pd
import numpy as np
import config

from strategy.signals   import (
    compute_all_macro, momentum, rolling_vol, resample_to_month_end
)
from strategy.portfolio import build_weight_series


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
    # Shift by 1: yesterday's vol → today's scaling
    scale = (config.VOL_TARGET / realised_vol.shift(1)).clip(upper=config.MAX_LEVERAGE)
    return raw_returns * scale


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
    daily_w   = weights.reindex(daily_ret.index, method="ffill").shift(1)
    raw_daily = (daily_w * daily_ret).sum(axis=1)
    raw_daily.name = "strategy_raw"

    # ── 5. Apply volatility targeting ─────────────────────────────────────
    strategy_daily       = _vol_scale(raw_daily)
    strategy_daily.name  = "strategy"

    # ── 6. Equal-weight benchmark (no vol targeting) ───────────────────────
    bm_w           = pd.Series(1 / len(config.ETF_UNIVERSE), index=config.ETF_UNIVERSE)
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
