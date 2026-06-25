"""
Shared backtest engine for sector rotation strategies (V2b–V2e).

Each versioned backtest.py is a thin wrapper that calls run(), effective_weights(),
and compute_stop_pcts() with its own config and signal/portfolio modules.
"""
import logging
import pandas as pd
import numpy as np

from strategies.backtest_core import vol_scale, drawdown_overlay, apply_transaction_costs

logger = logging.getLogger(__name__)


def _adaptive_stop_pct(m12_series: pd.Series, config) -> pd.Series:
    thr_sup = config.SUPERCYCLE_MOM_THRESHOLD
    thr_tac = config.TACTICAL_MOM_THRESHOLD
    t = ((m12_series - thr_tac) / (thr_sup - thr_tac)).clip(0.0, 1.0)
    return config.STOP_TACTICAL + t * (config.STOP_SUPERCYCLE - config.STOP_TACTICAL)


def _apply_adaptive_trailing_stops(
    daily_w: pd.DataFrame,
    prices: pd.DataFrame,
    mom12m_daily: pd.DataFrame,
    config,
    daily_ret: pd.DataFrame | None = None,
):
    """Apply exits to the daily weight matrix; free weight → SHY.

    Returns (adjusted_weights, adjusted_returns). For the legacy default
    (EXIT_MODE='pct_trail', STOP_FILL='close', STOP_PERSIST=False) the returned
    weights/returns are byte-identical to the previous behaviour — the trigger
    is lagged one day, the position is masked, and returns are unchanged.

    Config knobs (all default to the legacy behaviour):
      EXIT_MODE       'pct_trail' | 'fixed_trail' | 'ma_break'
      STOP_FILL       'close' (book the trigger day at its close) | 'stop'
                      (book it at the stop LEVEL L = peak*(1-stop_pct) — the
                      realistic resting-stop fill for a liquid ETF)
      STOP_PERSIST    False = re-enter as soon as price recovers (legacy daily
                      re-eval); True = stay in cash until the next rebalance
      STOP_FIXED_PCT  trailing distance for 'fixed_trail'
      STOP_SLIP_BPS   slippage haircut on a 'stop' fill (bps)
      GAP_FRAC/GAP_EXTRA  deterministic fat-tail gap model: every
                      round(1/GAP_FRAC)-th trigger fills GAP_EXTRA *worse* than L
    """
    stop_etfs = [
        e for e in config.TRAILING_STOP_ETFS
        if e in prices.columns and e in daily_w.columns
    ]
    if daily_ret is None:
        daily_ret = prices.pct_change().reindex(daily_w.index)
    ret_adj = daily_ret.reindex(columns=daily_w.columns).copy() \
        if set(daily_w.columns) - set(daily_ret.columns) else daily_ret.copy()
    if not stop_etfs:
        return daily_w, ret_adj

    w         = daily_w.copy()
    exit_mode = getattr(config, "EXIT_MODE", "pct_trail")
    fill      = getattr(config, "STOP_FILL", "close")
    persist   = getattr(config, "STOP_PERSIST", False)
    slip      = getattr(config, "STOP_SLIP_BPS", 0.0)
    gap_frac  = getattr(config, "GAP_FRAC", 0.0)
    gap_extra = getattr(config, "GAP_EXTRA", 0.0)
    gap_every = max(1, int(round(1.0 / gap_frac))) if gap_frac > 0 else 0
    cash      = config.CASH_ETF

    for etf in stop_etfs:
        prices_etf = prices[etf].reindex(w.index).ffill()

        if exit_mode == "ma_break":
            ref     = prices_etf.rolling(getattr(config, "MA_EXIT_WINDOW", 200),
                                         min_periods=1).mean().shift(1)
            breach  = (prices_etf < ref).fillna(False)
            level   = None
        else:
            peak = prices_etf.rolling(config.TRAILING_STOP_WINDOW,
                                      min_periods=1).max().shift(1)
            if exit_mode == "fixed_trail":
                sp = pd.Series(getattr(config, "STOP_FIXED_PCT", 0.10), index=w.index)
            else:
                m12 = mom12m_daily[etf].reindex(w.index).ffill().fillna(0.0) \
                      if etf in mom12m_daily.columns else pd.Series(0.0, index=w.index)
                sp = _adaptive_stop_pct(m12, config)
            level  = peak * (1.0 - sp)
            breach = (prices_etf < level).fillna(False)

        if not persist:
            # Legacy: lag the trigger, mask the weight, leave returns untouched.
            triggered = breach.shift(1).fillna(False).astype(bool)
            freed     = w[etf].where(triggered, 0.0)
            w[etf]    = w[etf].where(~triggered, 0.0)
            if cash in w.columns:
                w[cash] = w[cash] + freed
            continue

        # Persistent: hold INTO the breach day (no look-ahead), book the exit at
        # the chosen fill, then sit in cash until the target weight next changes.
        base   = w[etf].values
        pe_v   = prices_etf.values
        prev_v = prices_etf.shift(1).values
        lvl_v  = level.values if level is not None else None
        br_v   = breach.values
        rebal  = w[etf].ne(w[etf].shift(1)).fillna(True).values
        bw     = base.copy()
        rr     = ret_adj[etf].values.copy()
        stopped = False
        ntrig   = 0
        for t in range(len(w.index)):
            if rebal[t]:
                stopped = False
            if stopped:
                bw[t] = 0.0
            elif br_v[t] and base[t] > 0 and prev_v[t] and prev_v[t] > 0:
                if fill == "stop" and lvl_v is not None:
                    ntrig += 1
                    fillpx = lvl_v[t] * (1.0 - slip / 1e4)
                    if gap_every and (ntrig % gap_every == 0):
                        fillpx = lvl_v[t] * (1.0 - gap_extra)
                    rr[t] = fillpx / prev_v[t] - 1.0
                stopped = True
        w[etf]       = bw
        ret_adj[etf] = pd.Series(rr, index=w.index).fillna(0.0).values
        if cash in w.columns:
            w[cash] = w[cash] + (base - bw)

    return w, ret_adj


def effective_weights(
    signal_weights: pd.Series,
    recent_prices: pd.DataFrame,
    config,
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

        if len(prices_etf) < config.TRAILING_STOP_WINDOW:
            logger.warning(
                "%s: only %d days of price history (need %d for full stop window) — "
                "stop threshold may be unreliable",
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
    config,
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


def run(prices: pd.DataFrame, config, sig_mod, port_mod, macro_exposure=None) -> dict:
    """
    Run sector rotation backtest.

    sig_mod and port_mod are the versioned signal/portfolio modules (thin wrappers).
    config is the versioned config module (Optuna-patchable via setattr).

    macro_exposure : optional daily pd.Series of an exposure multiplier (e.g. the
        commodity macro overlay). Applied as the OUTERMOST scaler on strategy
        returns, lagged one day to avoid look-ahead. Default None ⇒ no change
        (byte-identical to legacy behaviour for all V2b–V2h variants).
    """
    rebal_freq = getattr(config, "REBALANCE_FREQ", "W")
    etf_prices = prices.reindex(columns=config.ETF_UNIVERSE)

    score_daily  = sig_mod.composite_score(etf_prices)
    mom12_daily  = sig_mod.mom_12m(etf_prices)
    vol_daily    = sig_mod.rolling_vol(etf_prices)
    regime_daily = sig_mod.spy_regime(prices) if config.SPY_TICKER in prices.columns else \
                   pd.Series(1.0, index=prices.index)

    score_r  = sig_mod.resample_to_period_end(score_daily,              rebal_freq)
    vol_r    = sig_mod.resample_to_period_end(vol_daily,                rebal_freq)
    regime_r = sig_mod.resample_to_period_end(regime_daily.to_frame(),  rebal_freq).iloc[:, 0]

    common = score_r.index.intersection(vol_r.index).intersection(regime_r.index)
    score_r, vol_r, regime_r = score_r.loc[common], vol_r.loc[common], regime_r.loc[common]

    corr_lookback = getattr(config, "CORR_LOOKBACK", None)
    corr_r = None
    if corr_lookback is not None:
        corr_r = sig_mod.rolling_corr_at_dates(etf_prices, common, corr_lookback)

    weights = port_mod.build_weight_series(score_r, vol_r, regime_r, corr_periodic=corr_r)

    start     = pd.Timestamp(config.BACKTEST_START)
    daily_ret = etf_prices.pct_change()
    daily_ret = daily_ret.loc[daily_ret.index >= start]

    daily_w_base  = weights.reindex(daily_ret.index).ffill().shift(1)
    mom12_trimmed = mom12_daily.reindex(daily_ret.index)

    daily_w, daily_ret_eff = _apply_adaptive_trailing_stops(
        daily_w_base, etf_prices, mom12_trimmed, config, daily_ret=daily_ret
    )

    daily_ret_no_cash = daily_ret_eff.copy()
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

    # Macro exposure overlay (commodity variant): outermost exposure scaler.
    # shift(1) so day-T return is scaled by the exposure decided from prior-day
    # (already publication-lagged) macro state. The un-deployed fraction earns 0%
    # (consistent with the SHY-earns-0% convention), so a CEIL of 1.0 means the
    # overlay can only de-risk and cannot inflate returns.
    if macro_exposure is not None:
        m = (macro_exposure.reindex(strategy_daily.index).ffill().fillna(1.0)
             .shift(1).fillna(1.0))
        strategy_daily = strategy_daily * m
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
