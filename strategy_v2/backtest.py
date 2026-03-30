"""
v2 backtest engine — wired to config_v2 and strategy_v2 modules.
Shared logic lives in strategy/backtest_core.py.
"""
import logging
import pandas as pd
import numpy as np
import config_v2 as config

from strategy_v2.signals    import compute_all_macro, momentum, rolling_vol, resample_to_month_end
from strategy_v2.portfolio  import build_weight_series
from strategy.backtest_core import vol_scale, drawdown_overlay, apply_trailing_stops, effective_weights_core

logger = logging.getLogger(__name__)


def _apply_trailing_stops(daily_w: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    return apply_trailing_stops(
        daily_w, prices,
        stop_etfs=config.HEDGE_ETFS,
        stop_pct=config.TRAILING_STOP_PCT,
        stop_window=config.TRAILING_STOP_WINDOW,
    )


def effective_weights(signal_weights: pd.Series, recent_prices: pd.DataFrame) -> pd.Series:
    return effective_weights_core(
        signal_weights, recent_prices,
        stop_etfs=config.HEDGE_ETFS,
        stop_pct=config.TRAILING_STOP_PCT,
        stop_window=config.TRAILING_STOP_WINDOW,
    )


def run(macro: pd.DataFrame, prices: pd.DataFrame) -> dict:
    macro_signals = compute_all_macro(macro)
    mom_daily     = momentum(prices[config.ETF_UNIVERSE])
    vol_daily     = rolling_vol(prices[config.ETF_UNIVERSE])

    macro_m = resample_to_month_end(macro_signals)
    mom_m   = resample_to_month_end(mom_daily)
    vol_m   = resample_to_month_end(vol_daily)

    common  = macro_m.index.intersection(mom_m.index).intersection(vol_m.index)
    macro_m, mom_m, vol_m = macro_m.loc[common], mom_m.loc[common], vol_m.loc[common]

    weights = build_weight_series(macro_m, mom_m, vol_m)

    daily_ret = prices[config.ETF_UNIVERSE].pct_change()
    daily_w   = _apply_trailing_stops(
        weights.reindex(daily_ret.index).ffill().shift(1),
        prices[config.ETF_UNIVERSE],
    )
    raw_daily      = (daily_w * daily_ret).sum(axis=1)
    raw_daily.name = "strategy_raw"

    cash_rate = macro["fedfunds"] if "fedfunds" in macro.columns else pd.Series(0.0, index=macro.index)
    strategy_daily      = drawdown_overlay(
        vol_scale(raw_daily, config.VOL_TARGET, config.VOL_LOOKBACK, config.MAX_LEVERAGE),
        cash_rate, config.DD_THRESHOLD, config.DD_SCALE,
    )
    strategy_daily.name = "strategy"

    bm_w            = pd.Series(1 / len(config.ETF_UNIVERSE), index=config.ETF_UNIVERSE)
    benchmark_daily = (daily_ret * bm_w).sum(axis=1)
    benchmark_daily.name = "benchmark_ew"

    nav    = (1 + strategy_daily.fillna(0)).cumprod()
    nav_bm = (1 + benchmark_daily.fillna(0)).cumprod()
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
