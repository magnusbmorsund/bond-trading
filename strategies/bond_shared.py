"""
Shared backtest run logic for bond strategy variants (V1 / V2 / V3).

Each variant's backtest.py calls run_bond() passing its own config module and
signal/portfolio modules. This eliminates the ~80-line run() body that was
previously duplicated verbatim across all three versions.
"""
import logging
import pandas as pd

from strategies.backtest_core import (
    vol_scale, drawdown_overlay, apply_trailing_stops, apply_transaction_costs,
)

logger = logging.getLogger(__name__)


def run_bond(macro: pd.DataFrame, prices: pd.DataFrame, config, signals, portfolio,
             stop_etfs=None) -> dict:
    """Run the full bond backtest pipeline.

    Args:
        macro:     FRED macro DataFrame (daily, ffilled)
        prices:    ETF price DataFrame (daily close)
        config:    the version's config module (read at call time — supports Optuna patching)
        signals:   the version's signals module (compute_all_macro, momentum, rolling_vol,
                   resample_to_month_end)
        portfolio: the version's portfolio module (build_weight_series)
        stop_etfs: ETF list to apply trailing stops to; defaults to config.HEDGE_ETFS
    """
    if stop_etfs is None:
        stop_etfs = config.HEDGE_ETFS

    # ── 1. Compute daily signals ───────────────────────────────────────────
    macro_signals = signals.compute_all_macro(macro)
    mom_daily     = signals.momentum(prices[config.ETF_UNIVERSE])
    vol_daily     = signals.rolling_vol(prices[config.ETF_UNIVERSE])

    # ── 2. Resample to month-end ───────────────────────────────────────────
    macro_m = signals.resample_to_month_end(macro_signals)
    mom_m   = signals.resample_to_month_end(mom_daily)
    vol_m   = signals.resample_to_month_end(vol_daily)

    common  = macro_m.index.intersection(mom_m.index).intersection(vol_m.index)
    macro_m, mom_m, vol_m = macro_m.loc[common], mom_m.loc[common], vol_m.loc[common]

    # ── 3. Build monthly target weights ────────────────────────────────────
    weights = portfolio.build_weight_series(macro_m, mom_m, vol_m)

    # ── 4. Daily returns + trailing stops ─────────────────────────────────
    daily_ret = prices[config.ETF_UNIVERSE].pct_change()
    daily_w   = apply_trailing_stops(
        weights.reindex(daily_ret.index).ffill().shift(1),
        prices[config.ETF_UNIVERSE],
        stop_etfs=stop_etfs,
        stop_pct=config.TRAILING_STOP_PCT,
        stop_window=config.TRAILING_STOP_WINDOW,
    )
    raw_daily      = apply_transaction_costs((daily_w * daily_ret).sum(axis=1), daily_w)
    raw_daily.name = "strategy_raw"

    # ── 5. Vol targeting + drawdown overlay ───────────────────────────────
    cash_rate = macro["fedfunds"] if "fedfunds" in macro.columns else pd.Series(0.0, index=macro.index)
    strategy_daily = drawdown_overlay(
        vol_scale(raw_daily, config.VOL_TARGET, config.VOL_LOOKBACK, config.MAX_LEVERAGE),
        cash_rate, config.DD_THRESHOLD, config.DD_SCALE,
    )
    strategy_daily.name = "strategy"

    # ── 6. Equal-weight benchmark ──────────────────────────────────────────
    bm_w            = pd.Series(1 / len(config.ETF_UNIVERSE), index=config.ETF_UNIVERSE)
    benchmark_daily = (daily_ret * bm_w).sum(axis=1)
    benchmark_daily.name = "benchmark_ew"

    # ── 7. NAV ─────────────────────────────────────────────────────────────
    nav    = (1 + strategy_daily.fillna(0)).cumprod()
    nav_bm = (1 + benchmark_daily.fillna(0)).cumprod()

    # ── 8. Turnover ────────────────────────────────────────────────────────
    turnover = (weights - weights.shift(1).fillna(0)).abs().sum(axis=1) / 2

    return {
        "weights":          weights,
        "daily_returns":    strategy_daily,
        "daily_returns_bm": benchmark_daily,
        "nav":              nav,
        "nav_bm":           nav_bm,
        "turnover":         turnover,
        "diagnostics":      pd.concat([macro_m, weights], axis=1),
    }
