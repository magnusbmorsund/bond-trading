"""
v2 signals — extends v1 with two additions:

  NEW  _ism_signal      : Philly Fed Future General Activity (GAFDFSA066MSFRBPHI) — diffusion index ISM proxy.
                          Leading indicator (like ISM PMI), replaces lagging INDPRO in duration composite.
  NEW  _dollar_signal   : Trade-weighted USD momentum (DTWEXBGS).
                          Rising USD = headwind for commodities + EM bonds.
                          Exposed via usd_z column in compute_all_macro output.

All other signals are identical to strategy/signals.py.
All constants are read from config_v2 at call time.
"""
import logging
import pandas as pd
import numpy as np
import config_v2 as config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _zscore(s: pd.Series, lookback: int) -> pd.Series:
    mu = s.rolling(lookback).mean()
    sd = s.rolling(lookback).std()
    return (s - mu) / sd.clip(lower=config.MIN_ZSCORE_CLIP)


# ---------------------------------------------------------------------------
# Individual signals (unchanged from v1)
# ---------------------------------------------------------------------------

def _curve_2s10s(macro: pd.DataFrame) -> pd.Series:
    return _zscore(macro["spread_2s10s"].ffill(), config.LOOKBACK_SIGNAL).rename("curve_2s10s")


def _curve_10y3m(macro: pd.DataFrame) -> pd.Series:
    return _zscore(macro["spread_10y3m"].ffill(), config.LOOKBACK_SIGNAL).rename("curve_10y3m")


def _fed_direction(macro: pd.DataFrame) -> pd.Series:
    s = macro["fedfunds"].ffill()
    roc = s.diff(63)
    return _zscore(-roc, config.LOOKBACK_SIGNAL).rename("fed_direction")


def _hy_oas(macro: pd.DataFrame) -> pd.Series:
    return (-_zscore(macro["hy_oas"].ffill(), config.LOOKBACK_SIGNAL)).rename("hy_oas")


def _vix_regime(macro: pd.DataFrame) -> pd.Series:
    vix = macro["vix"].ffill()
    level_z = _zscore(vix, config.LOOKBACK_SIGNAL)
    mom_z   = _zscore(vix.diff(21), config.LOOKBACK_SIGNAL)
    return (-(0.6 * level_z + 0.4 * mom_z)).rename("vix_regime")


def _breakeven_roc(macro: pd.DataFrame) -> pd.Series:
    bei = macro["breakeven_10y"].ffill()
    roc = bei.pct_change(63)
    return _zscore(roc.dropna().reindex(bei.index), config.LOOKBACK_SIGNAL).rename("breakeven_roc")


def _cpi_momentum(macro: pd.DataFrame) -> pd.Series:
    cpi = macro["cpi"].ffill()
    yoy = cpi.pct_change(252)
    return _zscore(yoy.dropna().reindex(cpi.index), config.LOOKBACK_SIGNAL).rename("cpi_momentum")


def _real_yield_signal(macro: pd.DataFrame) -> pd.Series:
    ry  = macro["real_yield_10y"].ffill()
    roc = ry.diff(63)
    return _zscore(-roc, config.LOOKBACK_SIGNAL).rename("real_yield_signal")


def _labor_market_signal(macro: pd.DataFrame) -> pd.Series:
    u      = macro["unemployment"].ffill()
    avg3m  = u.rolling(63).mean()
    min12m = u.rolling(252).min()
    sahm   = avg3m - min12m
    return _zscore(sahm, config.LOOKBACK_SIGNAL).rename("labor_market")


def _ig_spread_momentum(macro: pd.DataFrame) -> pd.Series:
    ig  = macro["ig_oas"].ffill()
    roc = ig.diff(63)
    return _zscore(-roc, config.LOOKBACK_SIGNAL).rename("ig_spread_momentum")


def _fed_qt_signal(macro: pd.DataFrame) -> pd.Series:
    if "fed_assets" not in macro.columns:
        logger.warning("fed_assets not in macro — fed_qt_signal set to 0 (Fed QT/QE signal missing)")
        return pd.Series(0.0, index=macro.index, name="fed_qt_signal")
    fa  = macro["fed_assets"].ffill()
    roc = fa.pct_change(63)
    return _zscore(roc, config.LOOKBACK_SIGNAL).rename("fed_qt_signal")


def _ted_stress_signal(macro: pd.DataFrame) -> pd.Series:
    if "ted_spread" not in macro.columns:
        logger.info("ted_spread not in macro — ted_stress_signal set to 0 (TEDRATE discontinued 2023)")
        return pd.Series(0.0, index=macro.index, name="ted_stress_signal")
    ted = macro["ted_spread"].ffill()
    return (-_zscore(ted, config.LOOKBACK_SIGNAL)).rename("ted_stress_signal")


# ---------------------------------------------------------------------------
# v2 NEW signals
# ---------------------------------------------------------------------------

def _ism_signal(macro: pd.DataFrame) -> pd.Series:
    """
    Philly Fed Future General Activity (GAFDFSA066MSFRBPHI) — free ISM PMI proxy.
    Diffusion index: > 0 = expanding, < 0 = contracting. Monthly, available since 1968.
    Declining index → slowing manufacturing → bullish bonds (return positive z-score).

    Falls back to INDPRO if series unavailable.
    """
    if "ism_mfg" not in macro.columns or macro["ism_mfg"].dropna().empty:
        logger.warning("ism_mfg not in macro — falling back to indpro for ISM slot (check FRED fetch)")
        if "indpro" not in macro.columns:
            return pd.Series(0.0, index=macro.index, name="ism_signal")
        ip  = macro["indpro"].ffill()
        roc = ip.pct_change(63)
        return _zscore(-roc, config.LOOKBACK_SIGNAL).rename("ism_signal")

    ism = macro["ism_mfg"].ffill()
    # 3-month change in the ISM level (it's an index, not a rate; diff is appropriate)
    roc = ism.diff(63)
    return _zscore(-roc, config.LOOKBACK_SIGNAL).rename("ism_signal")


def _dollar_signal(macro: pd.DataFrame) -> pd.Series:
    """
    Nominal broad trade-weighted USD index (DTWEXBGS).
    Rising USD = headwind for commodities (priced in USD) and EM bonds.
    Returns positive z-score when USD is rising.
    Weekly series → forward-filled to daily.

    Returns 0 if unavailable (graceful degradation).
    """
    if "usd_index" not in macro.columns or macro["usd_index"].dropna().empty:
        logger.warning("usd_index not in macro — dollar_signal set to 0 (USD commodity damper disabled)")
        return pd.Series(0.0, index=macro.index, name="dollar_signal")
    usd = macro["usd_index"].ffill()
    roc = usd.pct_change(63)   # 3-month momentum
    return _zscore(roc, config.LOOKBACK_SIGNAL).rename("dollar_signal")


# ---------------------------------------------------------------------------
# Composite signals
# ---------------------------------------------------------------------------

def compute_all_macro(macro: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a daily DataFrame with composite signals:
      duration_z  : positive → steep curve / falling real yields → favour TLT/IEF
      credit_z    : positive → tight spreads / low VIX / QE → favour credit
      inflation_z : positive → rising inflation → favour TIP/VTIP
      vix_raw     : raw VIX level (for hard overrides in portfolio)
      usd_z       : v2 NEW — positive = rising USD (headwind for commodities)
    """
    c2s10s  = _curve_2s10s(macro)
    c10y3m  = _curve_10y3m(macro)
    fed     = _fed_direction(macro)
    realyld = _real_yield_signal(macro)
    labor   = _labor_market_signal(macro)
    ism     = _ism_signal(macro)          # v2: ISM replaces INDPRO

    hyoas   = _hy_oas(macro)
    igmom   = _ig_spread_momentum(macro)
    vix     = _vix_regime(macro)
    fedqt   = _fed_qt_signal(macro)
    ted     = _ted_stress_signal(macro)

    bei     = _breakeven_roc(macro)
    cpi     = _cpi_momentum(macro)

    usd     = _dollar_signal(macro)       # v2: new signal

    dur_z = (
        config.W_DURATION_2S10S   * c2s10s
        + config.W_DURATION_10Y3M * c10y3m
        + config.W_DURATION_FED   * fed
        + config.W_DURATION_REALYLD * realyld
        + config.W_DURATION_LABOR   * labor
        + config.W_DURATION_ISM     * ism    # v2: now IPMAN (mfg-specific) not generic INDPRO
    )

    credit_z = (
        config.W_CREDIT_HYOAS  * hyoas
        + config.W_CREDIT_IGMOM  * igmom
        + config.W_CREDIT_VIX    * vix
        + config.W_CREDIT_FEDQT  * fedqt
        + config.W_CREDIT_TED    * ted
    )

    infl_z = (
        config.W_INFLATION_BEI * bei
        + config.W_INFLATION_CPI * cpi
    )

    out = pd.DataFrame({
        "duration_z":  dur_z,
        "credit_z":    credit_z,
        "inflation_z": infl_z,
        "vix_raw":     macro["vix"].ffill(),
        "usd_z":       usd,               # v2: exposed for portfolio layer
    }).ffill()

    return out


# ---------------------------------------------------------------------------
# Price-based signals (unchanged)
# ---------------------------------------------------------------------------

def momentum(prices: pd.DataFrame) -> pd.DataFrame:
    """12-1 month cross-sectional momentum. Positive → include."""
    past        = prices.shift(config.MOMENTUM_SKIP)
    longer      = prices.shift(config.MOMENTUM_WINDOW)
    longer_safe = longer.replace(0, float("nan"))   # avoid div-by-zero for zero-price rows
    return past / longer_safe - 1


def rolling_vol(prices: pd.DataFrame) -> pd.DataFrame:
    """Annualised rolling daily return volatility."""
    return prices.pct_change().rolling(config.LOOKBACK_VOL).std() * np.sqrt(252)


def resample_to_month_end(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("ME").last()
