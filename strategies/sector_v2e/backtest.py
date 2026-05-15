"""
V2e sector rotation backtest engine.

V2e extends V2d by adding 24m and 36m momentum lookbacks to the slow blend,
giving the ranking signal memory of multi-year supercycles. All other mechanics
(cluster caps, adaptive trailing stops, vol-target, DD overlay) are identical to V2d.
"""
import logging
import pandas as pd
import numpy as np
import configs.sector_v2e as config

from strategies.sector_v2e.signals   import (
    composite_score, mom_12m, rolling_vol, spy_regime,
    resample_to_period_end, rolling_corr_at_dates,
)
from strategies.sector_v2e.portfolio import build_weight_series
from strategies.backtest_core        import (
    vol_scale, drawdown_overlay, apply_transaction_costs,
)

logger = logging.getLogger(__name__)


def _adaptive_stop_pct(m12_series: pd.Series) -> pd.Series:
    thr_sup = config.SUPERCYCLE_MOM_THRESHOLD
    thr_tac = config.TACTICAL_MOM_THRESHOLD
    t = ((m12_series - thr_tac) / (thr_sup - thr_tac)).clip(0.0, 1.0)
    return config.STOP_TACTICAL + t * (config.STOP_SUPERCYCLE - config.STOP_TACTICAL)


def _apply_adaptive_trailing_stops(
    daily_w: pd.DataFrame,
    prices: pd.DataFrame,
    mom12m_daily: pd.DataFrame,
) -> pd.DataFrame:
    """Daily rolling trailing stops — freed weight → SHY."""
    stop_etfs = [
        e for e in config.TRAILING_STOP_ETFS
        if e in prices.columns and e in daily_w.columns
    ]
    if not stop_etfs:
        return daily_w

    w = daily_w.copy()

    for etf in stop_etfs:
        prices_etf   = prices[etf].reindex(w.index).ffill()
        rolling_peak = prices_etf.rolling(
            config.TRAILING_STOP_WINDOW, min_periods=1
        ).max().shift(1)

        m12 = mom12m_daily[etf].reindex(w.index).ffill().fillna(0.0) \
              if etf in mom12m_daily.columns \
              else pd.Series(0.0, index=w.index)

        stop_pct  = _adaptive_stop_pct(m12)
        triggered = (prices_etf < rolling_peak * (1.0 - stop_pct)).fillna(False)

        freed  = w[etf].where(triggered, 0.0)
        w[etf] = w[etf].where(~triggered, 0.0)
        if "SHY" in w.columns:
            w["SHY"] = w["SHY"] + freed

        n = triggered.sum()
        if n:
            logger.info(
                "Adaptive stop: %s triggered %d days (first=%s last=%s avg_stop=%.1f%%)",
                etf, n, triggered[triggered].index[0].date(),
                triggered[triggered].index[-1].date(),
                stop_pct[triggered].mean() * 100,
            )

    return w


def effective_weights(
    signal_weights: pd.Series,
    recent_prices: pd.DataFrame,
) -> pd.Series:
    """Adaptive trailing-stop-adjusted weights for today's live positions."""
    w = signal_weights.copy()

    for etf in config.TRAILING_STOP_ETFS:
        if etf not in w.index or w[etf] <= 0:
            continue
        if etf not in recent_prices.columns:
            continue
        prices_etf = recent_prices[etf].dropna()
        if len(prices_etf) < 2:
            continue

        # Issue #3: warn when price history is shorter than the stop window
        if len(prices_etf) < config.TRAILING_STOP_WINDOW:
            logger.warning(
                "%s: only %d days of price history (need %d for full stop window) — "
                "peak computed on truncated window; stop threshold may be unreliable",
                etf, len(prices_etf), config.TRAILING_STOP_WINDOW,
            )

        lb  = min(252, len(prices_etf) - 1)
        m12 = float(prices_etf.iloc[-1] / prices_etf.iloc[-lb - 1] - 1) if lb > 0 else 0.0

        thr_sup  = config.SUPERCYCLE_MOM_THRESHOLD
        thr_tac  = config.TACTICAL_MOM_THRESHOLD
        t        = max(0.0, min(1.0, (m12 - thr_tac) / (thr_sup - thr_tac)))
        stop_pct = config.STOP_TACTICAL + t * (config.STOP_SUPERCYCLE - config.STOP_TACTICAL)

        peak  = prices_etf.iloc[-config.TRAILING_STOP_WINDOW:].max()
        today = prices_etf.iloc[-1]
        if today < peak * (1.0 - stop_pct):
            pct_below = (peak - today) / peak
            logger.warning(
                "Adaptive stop active: %s is %.1f%% below peak "
                "(m12=%.0f%% → stop_pct=%.1f%%) — zeroed, moved to SHY",
                etf, pct_below * 100, m12 * 100, stop_pct * 100,
            )
            freed = w[etf]
            w[etf] = 0.0
            if "SHY" in w.index:
                w["SHY"] = w["SHY"] + freed

    return w


def compute_stop_pcts(
    signal_weights: pd.Series,
    recent_prices: pd.DataFrame,
) -> pd.DataFrame:
    """For each held position return stop%, peak price, stop price, margin."""
    rows = []
    for etf in signal_weights[signal_weights > 0].index:
        if etf == config.CASH_ETF or etf not in recent_prices.columns:
            continue
        prices_etf = recent_prices[etf].dropna()
        if len(prices_etf) < 2:
            continue

        lb       = min(252, len(prices_etf) - 1)
        m12      = float(prices_etf.iloc[-1] / prices_etf.iloc[-lb - 1] - 1) if lb > 0 else 0.0
        thr_sup  = config.SUPERCYCLE_MOM_THRESHOLD
        thr_tac  = config.TACTICAL_MOM_THRESHOLD
        t        = max(0.0, min(1.0, (m12 - thr_tac) / (thr_sup - thr_tac)))
        stop_pct = config.STOP_TACTICAL + t * (config.STOP_SUPERCYCLE - config.STOP_TACTICAL)

        win        = min(config.TRAILING_STOP_WINDOW, len(prices_etf))
        peak_price = prices_etf.iloc[-win:].max()
        stop_price = peak_price * (1.0 - stop_pct)
        today      = prices_etf.iloc[-1]

        rows.append({
            "etf":         etf,
            "m12":         m12,
            "stop_pct":    stop_pct,
            "peak_price":  peak_price,
            "today":       today,
            "stop_price":  stop_price,
            "pct_to_stop": (today - stop_price) / today,
        })

    return pd.DataFrame(rows).set_index("etf") if rows else pd.DataFrame()


def run(prices: pd.DataFrame) -> dict:
    """
    Run V2e backtest.
    Weekly target weights with cluster caps; daily adaptive trailing stops.
    Signal includes 24m and 36m momentum for supercycle capture.
    """
    rebal_freq = getattr(config, "REBALANCE_FREQ", "W")
    etf_prices = prices.reindex(columns=config.ETF_UNIVERSE)

    score_daily  = composite_score(etf_prices)
    mom12_daily  = mom_12m(etf_prices)
    vol_daily    = rolling_vol(etf_prices)
    regime_daily = spy_regime(prices) if config.SPY_TICKER in prices.columns else \
                   pd.Series(1.0, index=prices.index)

    score_r  = resample_to_period_end(score_daily,             rebal_freq)
    vol_r    = resample_to_period_end(vol_daily,               rebal_freq)
    regime_r = resample_to_period_end(regime_daily.to_frame(), rebal_freq).iloc[:, 0]

    common = score_r.index.intersection(vol_r.index).intersection(regime_r.index)
    score_r, vol_r, regime_r = score_r.loc[common], vol_r.loc[common], regime_r.loc[common]

    corr_r = rolling_corr_at_dates(etf_prices, common, config.CORR_LOOKBACK)

    weights = build_weight_series(score_r, vol_r, regime_r, corr_r)

    start     = pd.Timestamp(config.BACKTEST_START)
    daily_ret = etf_prices.pct_change()
    daily_ret = daily_ret.loc[daily_ret.index >= start]

    daily_w_base  = weights.reindex(daily_ret.index).ffill().shift(1)
    mom12_trimmed = mom12_daily.reindex(daily_ret.index)

    daily_w = _apply_adaptive_trailing_stops(daily_w_base, etf_prices, mom12_trimmed)

    raw_daily      = apply_transaction_costs((daily_w * daily_ret).sum(axis=1), daily_w)
    raw_daily.name = "strategy_raw"

    cash_rate      = pd.Series(0.0, index=daily_ret.index)
    strategy_daily = drawdown_overlay(
        vol_scale(raw_daily, config.VOL_TARGET, config.VOL_LOOKBACK, config.MAX_LEVERAGE),
        cash_rate, config.DD_THRESHOLD, config.DD_SCALE,
    )
    strategy_daily.name = "strategy"

    bm_etfs  = [e for e in config.SECTOR_CORE if e in daily_ret.columns]
    bm_daily = daily_ret[bm_etfs].mean(axis=1)
    bm_daily.name = "benchmark_ew"

    nav    = (1 + strategy_daily.fillna(0)).cumprod()
    nav_bm = (1 + bm_daily.fillna(0)).cumprod()
    turnover = (weights - weights.shift(1).fillna(0)).abs().sum(axis=1) / 2

    non_cash = [c for c in daily_w.columns if c != config.CASH_ETF]
    position_count = (daily_w[non_cash] > config.MIN_WEIGHT_THRESHOLD).sum(axis=1)

    return {
        "weights":          weights,
        "daily_w":          daily_w,
        "daily_returns":    strategy_daily,
        "daily_returns_bm": bm_daily,
        "nav":              nav,
        "nav_bm":           nav_bm,
        "turnover":         turnover,
        "position_count":   position_count,
    }
