"""
Sector rotation backtest engine.

Daily simulation steps:
  1. Compute monthly momentum + vol signals → build_weight_series()
  2. Forward-fill monthly weights to daily grid (execute at next open = shift(1))
  3. Apply daily trailing stops (all non-cash sector ETFs)
  4. Apply transaction costs
  5. Vol-scale to VOL_TARGET
  6. Drawdown overlay (full exit when DD > DD_THRESHOLD and prior day negative)

No FRED data — cash_rate defaults to 0 (conservative; SHY yield not modelled explicitly).
"""
import logging
import pandas as pd
import numpy as np
import config_sector as config

from strategy_sector.signals   import momentum, rolling_vol, spy_regime, resample_to_month_end
from strategy_sector.portfolio import build_weight_series
from strategy.backtest_core    import (
    vol_scale, drawdown_overlay, apply_trailing_stops,
    effective_weights_core, apply_transaction_costs,
)

logger = logging.getLogger(__name__)


def _apply_trailing_stops(daily_w: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    return apply_trailing_stops(
        daily_w, prices,
        stop_etfs=config.TRAILING_STOP_ETFS,
        stop_pct=config.TRAILING_STOP_PCT,
        stop_window=config.TRAILING_STOP_WINDOW,
    )


def effective_weights(signal_weights: pd.Series, recent_prices: pd.DataFrame) -> pd.Series:
    """Trailing-stop-adjusted weights for today's actionable positions."""
    return effective_weights_core(
        signal_weights, recent_prices,
        stop_etfs=config.TRAILING_STOP_ETFS,
        stop_pct=config.TRAILING_STOP_PCT,
        stop_window=config.TRAILING_STOP_WINDOW,
    )


def run(prices: pd.DataFrame) -> dict:
    """
    Run full sector rotation backtest.

    Parameters
    ----------
    prices : DataFrame of adjusted closes for ALL_TICKERS (sector ETFs + SPY)

    Returns
    -------
    dict with keys: weights, daily_returns, daily_returns_bm, nav, nav_bm, turnover
    """
    etf_prices = prices.reindex(columns=config.ETF_UNIVERSE)
    spy_prices = prices[config.SPY_TICKER] if config.SPY_TICKER in prices.columns else None

    mom_daily    = momentum(etf_prices)
    vol_daily    = rolling_vol(etf_prices)
    regime_daily = spy_regime(prices) if spy_prices is not None else pd.Series(1.0, index=prices.index)

    mom_m    = resample_to_month_end(mom_daily)
    vol_m    = resample_to_month_end(vol_daily)
    regime_m = resample_to_month_end(regime_daily.to_frame()).iloc[:, 0]

    common = mom_m.index.intersection(vol_m.index).intersection(regime_m.index)
    mom_m, vol_m, regime_m = mom_m.loc[common], vol_m.loc[common], regime_m.loc[common]

    weights = build_weight_series(mom_m, vol_m, regime_m)

    # Trim to backtest start after warm-up period
    start = pd.Timestamp(config.BACKTEST_START)
    daily_ret = etf_prices.pct_change()
    daily_ret = daily_ret.loc[daily_ret.index >= start]

    daily_w = _apply_trailing_stops(
        weights.reindex(daily_ret.index).ffill().shift(1),
        etf_prices,
    )

    raw_daily      = apply_transaction_costs((daily_w * daily_ret).sum(axis=1), daily_w)
    raw_daily.name = "strategy_raw"

    cash_rate = pd.Series(0.0, index=daily_ret.index)   # SHY yield not explicitly modelled
    strategy_daily = drawdown_overlay(
        vol_scale(raw_daily, config.VOL_TARGET, config.VOL_LOOKBACK, config.MAX_LEVERAGE),
        cash_rate, config.DD_THRESHOLD, config.DD_SCALE,
    )
    strategy_daily.name = "strategy"

    # Equal-weight sector benchmark (all core sectors, no SPY, no SHY)
    bm_etfs     = [e for e in config.SECTOR_CORE if e in daily_ret.columns]
    bm_w        = 1.0 / len(bm_etfs)
    bm_daily    = daily_ret[bm_etfs].mean(axis=1)
    bm_daily.name = "benchmark_ew"

    nav    = (1 + strategy_daily.fillna(0)).cumprod()
    nav_bm = (1 + bm_daily.fillna(0)).cumprod()
    turnover = (weights - weights.shift(1).fillna(0)).abs().sum(axis=1) / 2

    return {
        "weights":          weights,
        "daily_returns":    strategy_daily,
        "daily_returns_bm": bm_daily,
        "nav":              nav,
        "nav_bm":           nav_bm,
        "turnover":         turnover,
    }
