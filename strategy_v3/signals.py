"""
v3 signals — extends v2 with new signals and restructured composites:

  NEW  _credit_impulse      : Second derivative of HY OAS — acceleration/deceleration
                              of credit stress. Positive = spreads decelerating = good.
  NEW  _vix_term_structure  : VIX / VIX3M ratio. Normal contango (VIX3M > VIX) is bullish
                              credit. Inverted = acute stress. Falls back to 0 if VIX3M missing.
  NEW  _ism_growth          : ISM level change with POSITIVE sign — for growth composite.
                              (Contrast with _ism_signal which uses NEGATIVE sign for duration.)
  NEW  _indpro_growth       : Industrial production % change — positive = growth expanding.
  NEW  _labor_growth        : Sahm-style rule inverted — negative Sahm = labor healthy.

  COMPOSITES changed:
    duration_z: 4 components only (2s10s, 10y3m, fed, realyld) — ISM and labor REMOVED.
    credit_z:   7 components (hyoas, credit_impulse, igmom, vix_regime, vix_term_structure,
                              fedqt, ted).
    inflation_z: unchanged (bei, cpi).
    growth_z:   NEW composite (ism_growth, indpro_growth, labor_growth).
    Output also adds growth_z column alongside duration_z, credit_z, inflation_z, vix_raw, usd_z.

All constants read from config_v3 at call time.
"""
import logging
import pandas as pd
import numpy as np
import config_v3 as config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _zscore(s: pd.Series, lookback: int) -> pd.Series:
    mu = s.rolling(lookback).mean()
    sd = s.rolling(lookback).std()
    return (s - mu) / sd.clip(lower=config.MIN_ZSCORE_CLIP)


# ---------------------------------------------------------------------------
# Individual signals (carried over from v2)
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


def _ism_signal(macro: pd.DataFrame) -> pd.Series:
    """
    Philly Fed Future General Activity — ISM PMI proxy.
    NEGATIVE sign: declining ISM = bullish bonds (for duration composite).
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
# v3 NEW signals
# ---------------------------------------------------------------------------

def _credit_impulse(macro: pd.DataFrame) -> pd.Series:
    """
    Second derivative of HY OAS — acceleration/deceleration of credit stress.

    d1 = hy_oas.diff(21)   (1-month change)
    d2 = d1.diff(21)       (change-in-change — is the deterioration speeding up?)

    Positive return value = spreads NOT accelerating wider = good for credit.
    Negative = spread acceleration = credit stress building.
    """
    hy  = macro["hy_oas"].ffill()
    d1  = hy.diff(21)
    d2  = d1.diff(21)
    return _zscore(-d2, config.LOOKBACK_SIGNAL).rename("credit_impulse")


def _vix_term_structure(macro: pd.DataFrame) -> pd.Series:
    """
    VIX term structure signal — ratio of spot VIX to 3-month VIX (^VIX3M).

    ratio = vix / vix3m (clip vix3m at 0.01 to avoid div-by-zero)
    signal = _zscore(-(ratio - 1.0), LOOKBACK_SIGNAL)

    Normal contango (VIX3M > VIX): ratio < 1 → ratio-1 negative → signal positive → bullish credit.
    Inverted (VIX > VIX3M): ratio > 1 → ratio-1 positive → signal negative → acute stress.

    Falls back to pd.Series(0.0, ...) if vix3m missing or all NaN.
    """
    if "vix3m" not in macro.columns or macro["vix3m"].dropna().empty:
        logger.warning("vix3m not in macro — vix_term_structure signal set to 0")
        return pd.Series(0.0, index=macro.index, name="vix_term_structure")

    vix   = macro["vix"].ffill()
    vix3m = macro["vix3m"].ffill().clip(lower=0.01)
    ratio = vix / vix3m
    return _zscore(-(ratio - 1.0), config.LOOKBACK_SIGNAL).rename("vix_term_structure")


def _ism_growth(macro: pd.DataFrame) -> pd.Series:
    """
    Philly Fed Future General Activity — ISM PMI proxy for growth composite.

    POSITIVE sign: rising ISM = expanding manufacturing = good for growth.
    (Opposite of _ism_signal which uses negative sign for the duration composite.)

    Falls back to INDPRO if series unavailable.
    """
    if "ism_mfg" not in macro.columns or macro["ism_mfg"].dropna().empty:
        logger.warning("ism_mfg not in macro — falling back to indpro for ism_growth slot")
        if "indpro" not in macro.columns:
            return pd.Series(0.0, index=macro.index, name="ism_growth")
        ip  = macro["indpro"].ffill()
        roc = ip.pct_change(63)
        return _zscore(roc, config.LOOKBACK_SIGNAL).rename("ism_growth")

    ism = macro["ism_mfg"].ffill()
    roc = ism.diff(63)
    return _zscore(roc, config.LOOKBACK_SIGNAL).rename("ism_growth")


def _indpro_growth(macro: pd.DataFrame) -> pd.Series:
    """
    Industrial production % change — positive when production growing.

    indpro.pct_change(63) captures 3-month industrial output momentum.
    Higher → more economic activity → positive growth signal.
    """
    if "indpro" not in macro.columns or macro["indpro"].dropna().empty:
        logger.warning("indpro not in macro — indpro_growth signal set to 0")
        return pd.Series(0.0, index=macro.index, name="indpro_growth")
    ip  = macro["indpro"].ffill()
    roc = ip.pct_change(63)
    return _zscore(roc, config.LOOKBACK_SIGNAL).rename("indpro_growth")


def _labor_growth(macro: pd.DataFrame) -> pd.Series:
    """
    Labor market health signal — Sahm-rule inspired, inverted for growth composite.

    sahm = avg3m_unemployment - min12m_unemployment
    signal = _zscore(-sahm, LOOKBACK_SIGNAL)

    Negative sahm (unemployment below 12-month low) = labor market healthy = positive growth.
    Positive sahm = unemployment rising = labor deteriorating = negative growth signal.
    """
    if "unemployment" not in macro.columns or macro["unemployment"].dropna().empty:
        logger.warning("unemployment not in macro — labor_growth signal set to 0")
        return pd.Series(0.0, index=macro.index, name="labor_growth")
    u      = macro["unemployment"].ffill()
    avg3m  = u.rolling(63).mean()
    min12m = u.rolling(252).min()
    sahm   = avg3m - min12m
    return _zscore(-sahm, config.LOOKBACK_SIGNAL).rename("labor_growth")


# ---------------------------------------------------------------------------
# Composite signals
# ---------------------------------------------------------------------------

def compute_all_macro(macro: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a daily DataFrame with composite signals:
      duration_z  : positive → steep curve / falling real yields → favour EDV/TLT/IEF
      credit_z    : positive → tight spreads / low VIX / QE → favour credit
      inflation_z : positive → rising inflation → favour TIP/VTIP
      growth_z    : v3 NEW — positive → expanding economy → favour equity/managed futures
      vix_raw     : raw VIX level (for hard overrides in portfolio)
      usd_z       : positive = rising USD (headwind for commodities)

    v3 changes vs v2:
      - duration_z: 4 components only (ISM and labor REMOVED — moved to growth_z)
      - credit_z: 7 components (added credit_impulse, vix_term_structure)
      - growth_z: NEW composite from ism_growth, indpro_growth, labor_growth
    """
    # Duration composite (4 components — no ISM, no labor in v3)
    c2s10s  = _curve_2s10s(macro)
    c10y3m  = _curve_10y3m(macro)
    fed     = _fed_direction(macro)
    realyld = _real_yield_signal(macro)

    # Credit composite (7 components)
    hyoas      = _hy_oas(macro)
    c_impulse  = _credit_impulse(macro)
    igmom      = _ig_spread_momentum(macro)
    vix_reg    = _vix_regime(macro)
    vix_ts     = _vix_term_structure(macro)
    fedqt      = _fed_qt_signal(macro)
    ted        = _ted_stress_signal(macro)

    # Inflation composite (unchanged)
    bei = _breakeven_roc(macro)
    cpi = _cpi_momentum(macro)

    # Growth composite (NEW in v3)
    ism_gr    = _ism_growth(macro)
    indpro_gr = _indpro_growth(macro)
    labor_gr  = _labor_growth(macro)

    # USD signal (v2, unchanged)
    usd = _dollar_signal(macro)

    dur_z = (
        config.W_DURATION_2S10S   * c2s10s
        + config.W_DURATION_10Y3M * c10y3m
        + config.W_DURATION_FED   * fed
        + config.W_DURATION_REALYLD * realyld
    )

    credit_z = (
        config.W_CREDIT_HYOAS    * hyoas
        + config.W_CREDIT_IMPULSE  * c_impulse
        + config.W_CREDIT_IGMOM    * igmom
        + config.W_CREDIT_VIX      * vix_reg
        + config.W_CREDIT_VIX_TS   * vix_ts
        + config.W_CREDIT_FEDQT    * fedqt
        + config.W_CREDIT_TED      * ted
    )

    infl_z = (
        config.W_INFLATION_BEI * bei
        + config.W_INFLATION_CPI * cpi
    )

    growth_z = (
        config.W_GROWTH_ISM    * ism_gr
        + config.W_GROWTH_INDPRO * indpro_gr
        + config.W_GROWTH_LABOR  * labor_gr
    )

    out = pd.DataFrame({
        "duration_z":  dur_z,
        "credit_z":    credit_z,
        "inflation_z": infl_z,
        "growth_z":    growth_z,
        "vix_raw":     macro["vix"].ffill(),
        "usd_z":       usd,
    }).ffill()

    return out


# ---------------------------------------------------------------------------
# Price-based signals (unchanged from v2)
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
