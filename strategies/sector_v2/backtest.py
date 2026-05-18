"""
V2 sector rotation backtest engine.

Key difference from V1: adaptive per-ETF trailing stops.
Stop % scales with the ETF's 12M momentum — wide for confirmed supercycles
(e.g. URNM +1100%), tight for tactical bounces (e.g. UNG short-cycle).

Daily simulation:
  1. Compute multi-timescale composite score + 12M momentum + vol + regime
  2. Resample to month-end, build monthly target weights
  3. Forward-fill to daily, one-day execution lag (shift(1))
  4. Apply adaptive trailing stops (per-ETF stop % from 12M momentum)
  5. Apply transaction costs
  6. Vol-scale to VOL_TARGET
  7. Drawdown overlay (full exit on distress)
"""
import logging
import pandas as pd
import numpy as np
import configs.sector_v2 as config

from strategies.sector_v2.signals   import (
    composite_score, mom_12m, rolling_vol, spy_regime, resample_to_month_end,
)
from strategies.sector_v2.portfolio import build_weight_series
from strategies.backtest_core        import (
    vol_scale, drawdown_overlay,
    apply_transaction_costs,
)

logger = logging.getLogger(__name__)


def _adaptive_stop_pct(m12_series: pd.Series) -> pd.Series:
    """
    Vectorised adaptive stop percentage based on 12M momentum.
    Linearly interpolates between STOP_TACTICAL and STOP_SUPERCYCLE.
    """
    thr_sup = config.SUPERCYCLE_MOM_THRESHOLD
    thr_tac = config.TACTICAL_MOM_THRESHOLD
    t = ((m12_series - thr_tac) / (thr_sup - thr_tac)).clip(0.0, 1.0)
    return config.STOP_TACTICAL + t * (config.STOP_SUPERCYCLE - config.STOP_TACTICAL)


def _apply_adaptive_trailing_stops(
    daily_w: pd.DataFrame,
    prices: pd.DataFrame,
    mom12m_daily: pd.DataFrame,
) -> pd.DataFrame:
    """
    Per-ETF adaptive trailing stops — daily rolling peak, freed weight → SHY.
    Used for the optimisation / research backtest.
    """
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

        if etf in mom12m_daily.columns:
            m12 = mom12m_daily[etf].reindex(w.index).ffill().fillna(0.0)
        else:
            m12 = pd.Series(0.0, index=w.index)

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


def _apply_monthly_fixed_stops(
    daily_w: pd.DataFrame,
    prices: pd.DataFrame,
    mom12m_daily: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Monthly fixed stop loss — mimics the Nordnet trading process exactly:
      • Stop level set once at each month-end (71-day peak × adaptive stop%)
      • Level is FIXED for the entire following month
      • Once triggered, position stays out until next month-end rebalance
      • Freed weight goes to CASH (0%), NOT to SHY
    """
    stop_etfs = [
        e for e in config.TRAILING_STOP_ETFS
        if e in prices.columns and e in daily_w.columns
    ]
    if not stop_etfs:
        return daily_w

    w          = daily_w.copy()
    all_dates  = w.index
    reb_sorted = sorted([d for d in rebalance_dates if d in all_dates])

    for etf in stop_etfs:
        prices_etf = prices[etf].reindex(all_dates).ffill()
        stopped    = pd.Series(False, index=all_dates)

        for i, reb_date in enumerate(reb_sorted):
            next_reb = reb_sorted[i + 1] if i + 1 < len(reb_sorted) else all_dates[-1]
            month_mask  = (all_dates > reb_date) & (all_dates <= next_reb)
            month_dates = all_dates[month_mask]
            if len(month_dates) == 0:
                continue

            # Only relevant if holding this ETF going into the new month
            if reb_date not in w.index or w.loc[reb_date, etf] <= 0:
                continue

            # 71-day peak as of rebalance date
            reb_pos    = all_dates.get_loc(reb_date)
            start_pos  = max(0, reb_pos - config.TRAILING_STOP_WINDOW + 1)
            peak       = prices_etf.iloc[start_pos : reb_pos + 1].max()

            # Adaptive stop % from 12M momentum at rebalance date
            if etf in mom12m_daily.columns and reb_date in mom12m_daily.index:
                m12_val = float(mom12m_daily.loc[reb_date, etf])
            else:
                m12_val = 0.0
            if pd.isna(m12_val):
                m12_val = 0.0

            thr_sup  = config.SUPERCYCLE_MOM_THRESHOLD
            thr_tac  = config.TACTICAL_MOM_THRESHOLD
            t        = max(0.0, min(1.0, (m12_val - thr_tac) / (thr_sup - thr_tac)))
            stop_pct = config.STOP_TACTICAL + t * (config.STOP_SUPERCYCLE - config.STOP_TACTICAL)

            fixed_stop_price = peak * (1.0 - stop_pct)

            # Check daily within this month — once triggered, stay out rest of month
            prices_month = prices_etf[month_dates]
            below        = prices_month < fixed_stop_price
            if below.any():
                trigger_day = below[below].index[0]
                stopped[trigger_day:next_reb] = True
                logger.debug(
                    "Monthly fixed stop: %s triggered %s  peak=%.2f  stop=%.2f  (%.1f%%)",
                    etf, trigger_day.date(), peak, fixed_stop_price, stop_pct * 100,
                )

        # Route freed weight to CASH_STOP (earns 0%), not SHY and not nowhere.
        # This keeps portfolio weight sum = 1.0 so vol-scaling works correctly.
        if "CASH_STOP" not in w.columns:
            w["CASH_STOP"] = 0.0
        freed  = w[etf].where(stopped, 0.0)
        w[etf] = w[etf].where(~stopped, 0.0)
        w["CASH_STOP"] = w["CASH_STOP"] + freed

        n = stopped.sum()
        if n:
            logger.info(
                "Monthly fixed stop: %s stopped on %d days  (first=%s last=%s)",
                etf, n, stopped[stopped].index[0].date(), stopped[stopped].index[-1].date(),
            )

    return w


def effective_weights(
    signal_weights: pd.Series,
    recent_prices: pd.DataFrame,
) -> pd.Series:
    """
    Adaptive trailing-stop-adjusted weights for today's live positions.
    Mirrors _apply_adaptive_trailing_stops but for a single observation.
    """
    w = signal_weights.copy()

    for etf in config.TRAILING_STOP_ETFS:
        if etf not in w.index or w[etf] <= 0:
            continue
        if etf not in recent_prices.columns:
            continue
        prices_etf = recent_prices[etf].dropna()
        if len(prices_etf) < 2:
            continue

        # 12M momentum
        lb = min(252, len(prices_etf) - 1)
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
) -> pd.Series:
    """
    For each held position, return the trailing stop % and the current stop level (price).
    Used by the weights command to tell the trader exactly what to set in Nordnet.
    Returns a DataFrame with columns: stop_pct, peak_price, stop_price.
    """
    rows = []
    for etf in signal_weights[signal_weights > 0].index:
        if etf == config.CASH_ETF or etf not in recent_prices.columns:
            continue
        prices_etf = recent_prices[etf].dropna()
        if len(prices_etf) < 2:
            continue

        lb      = min(252, len(prices_etf) - 1)
        m12     = float(prices_etf.iloc[-1] / prices_etf.iloc[-lb - 1] - 1) if lb > 0 else 0.0
        thr_sup = config.SUPERCYCLE_MOM_THRESHOLD
        thr_tac = config.TACTICAL_MOM_THRESHOLD
        t       = max(0.0, min(1.0, (m12 - thr_tac) / (thr_sup - thr_tac)))
        stop_pct = config.STOP_TACTICAL + t * (config.STOP_SUPERCYCLE - config.STOP_TACTICAL)

        win        = min(config.TRAILING_STOP_WINDOW, len(prices_etf))
        peak_price = prices_etf.iloc[-win:].max()
        stop_price = peak_price * (1.0 - stop_pct)
        today      = prices_etf.iloc[-1]

        rows.append({
            "etf":        etf,
            "m12":        m12,
            "stop_pct":   stop_pct,
            "peak_price": peak_price,
            "today":      today,
            "stop_price": stop_price,
            "pct_to_stop": (today - stop_price) / today,
        })

    return pd.DataFrame(rows).set_index("etf") if rows else pd.DataFrame()


def _weekly_rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each calendar week."""
    return pd.DatetimeIndex(
        pd.Series(index, index=index)
        .resample("W")
        .last()
        .dropna()
        .values
    )


def run(prices: pd.DataFrame, stop_freq: str = "daily") -> dict:
    """
    Run V2 sector rotation backtest.

    Parameters
    ----------
    prices    : DataFrame of adjusted closes for ALL_TICKERS
    stop_freq : 'daily'   — daily rolling trailing stops → SHY on stop-out (default)
                'weekly'  — stop levels reset every Friday, cash on stop-out
                'monthly' — stop levels reset every month-end, cash on stop-out
    """
    etf_prices = prices.reindex(columns=config.ETF_UNIVERSE)

    score_daily  = composite_score(etf_prices)
    mom12_daily  = mom_12m(etf_prices)
    vol_daily    = rolling_vol(etf_prices)
    regime_daily = spy_regime(prices) if config.SPY_TICKER in prices.columns else \
                   pd.Series(1.0, index=prices.index)

    score_m  = resample_to_month_end(score_daily)
    vol_m    = resample_to_month_end(vol_daily)
    regime_m = resample_to_month_end(regime_daily.to_frame()).iloc[:, 0]

    common = score_m.index.intersection(vol_m.index).intersection(regime_m.index)
    score_m, vol_m, regime_m = score_m.loc[common], vol_m.loc[common], regime_m.loc[common]

    weights = build_weight_series(score_m, vol_m, regime_m)

    start     = pd.Timestamp(config.BACKTEST_START)
    daily_ret = etf_prices.pct_change()
    daily_ret = daily_ret.loc[daily_ret.index >= start]

    daily_w_base  = weights.reindex(daily_ret.index).ffill().shift(1)
    mom12_trimmed = mom12_daily.reindex(daily_ret.index)

    if stop_freq in ("weekly", "monthly"):
        if stop_freq == "monthly":
            reb_dates = weights.index
        else:
            reb_dates = _weekly_rebalance_dates(daily_ret.index)
        daily_w = _apply_monthly_fixed_stops(
            daily_w_base, etf_prices, mom12_trimmed, reb_dates,
        )
        if "CASH_STOP" in daily_w.columns:
            daily_ret = daily_ret.copy()
            daily_ret["CASH_STOP"] = 0.0
    else:
        daily_w = _apply_adaptive_trailing_stops(daily_w_base, etf_prices, mom12_trimmed)

    daily_ret_no_cash = daily_ret.copy()
    if config.CASH_ETF in daily_ret_no_cash.columns:
        daily_ret_no_cash[config.CASH_ETF] = 0.0

    raw_daily      = apply_transaction_costs((daily_w * daily_ret_no_cash).sum(axis=1), daily_w)
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

    # Position count: ETF columns with weight > 0 (excl SHY/cash)
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
