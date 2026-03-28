"""
Vectorised backtest engine.

Weights are set at each month-end close and held through the next month.
Returns are computed from the following month's daily prices so there is
no look-ahead bias.

Usage:
    from strategy.backtest import run
    results = run(macro, prices)
"""
import pandas as pd
import numpy as np

from strategy.signals  import (
    compute_all_macro, momentum, rolling_vol, resample_to_month_end
)
from strategy.portfolio import build_weight_series
from config import ETF_UNIVERSE


def run(macro: pd.DataFrame, prices: pd.DataFrame) -> dict:
    """
    Run the full backtest.

    Parameters
    ----------
    macro  : daily FRED macro DataFrame (from data.pipeline)
    prices : daily ETF adjusted-close DataFrame

    Returns
    -------
    dict with keys:
        weights        : pd.DataFrame — monthly target weights
        daily_returns  : pd.Series    — strategy daily returns
        nav            : pd.Series    — cumulative NAV (starts at 1.0)
        turnover       : pd.Series    — monthly one-way turnover
        diagnostics    : pd.DataFrame — signals at each rebalance date
    """
    # ── 1. Compute daily signals ───────────────────────────────────────────
    macro_signals = compute_all_macro(macro)
    mom_daily     = momentum(prices)
    vol_daily     = rolling_vol(prices)

    # ── 2. Resample everything to month-end ────────────────────────────────
    macro_m = resample_to_month_end(macro_signals)
    mom_m   = resample_to_month_end(mom_daily)
    vol_m   = resample_to_month_end(vol_daily)

    # Align on dates present in all three
    common = macro_m.index.intersection(mom_m.index).intersection(vol_m.index)
    macro_m = macro_m.loc[common]
    mom_m   = mom_m.loc[common]
    vol_m   = vol_m.loc[common]

    # ── 3. Build monthly target weights ───────────────────────────────────
    weights = build_weight_series(macro_m, mom_m, vol_m)

    # ── 4. Compute daily strategy returns ─────────────────────────────────
    daily_ret = prices[ETF_UNIVERSE].pct_change()

    # For each day, find the most recent month-end weight
    # Weights are set at month-end close → effective from next trading day
    daily_w = weights.reindex(daily_ret.index, method="ffill").shift(1)

    strategy_daily = (daily_w * daily_ret).sum(axis=1)
    strategy_daily.name = "strategy"

    # ── 5. Benchmark: equal-weight buy-and-hold ────────────────────────────
    bm_w = pd.Series(1 / len(ETF_UNIVERSE), index=ETF_UNIVERSE)
    benchmark_daily = (daily_ret[ETF_UNIVERSE] * bm_w).sum(axis=1)
    benchmark_daily.name = "benchmark_ew"

    # ── 6. NAV series ──────────────────────────────────────────────────────
    nav       = (1 + strategy_daily).cumprod()
    nav_bm    = (1 + benchmark_daily).cumprod()

    # ── 7. Monthly turnover ────────────────────────────────────────────────
    prev_w = weights.shift(1).fillna(0)
    turnover = (weights - prev_w).abs().sum(axis=1) / 2  # one-way

    # ── 8. Diagnostics ────────────────────────────────────────────────────
    diagnostics = pd.concat([macro_m, weights], axis=1)

    return {
        "weights":        weights,
        "daily_returns":  strategy_daily,
        "daily_returns_bm": benchmark_daily,
        "nav":            nav,
        "nav_bm":         nav_bm,
        "turnover":       turnover,
        "diagnostics":    diagnostics,
    }
