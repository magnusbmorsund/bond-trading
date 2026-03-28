"""
Compute monthly signals from macro data and ETF prices.

DURATION composite (→ TLT/IEF/SHY tilt):
  s1. yield_curve_2s10s  — z-score of 2s10s spread
  s2. yield_curve_10y3m  — z-score of 10Y-3M spread (better recession signal)
  s3. fed_rate_direction — 3m change in fed funds (rising = bearish duration)
  s4. real_yield_signal  — 3m change in 10Y real yield (DFII10); rising = bearish ← NEW
  s5. labor_market       — unemployment trend; rising = bullish duration (Fed will cut) ← NEW

CREDIT composite (→ LQD/HYG allocation):
  s6. hy_oas_zscore      — inverted HY OAS z-score (tight = risk-on)
  s7. ig_spread_momentum — rate of widening in IG spreads (widening fast = danger) ← NEW
  s8. vix_regime         — inverted VIX level + momentum (low/falling = risk-on)
  s9. fed_qt_signal      — Fed balance sheet momentum (shrinking = QT headwind) ← NEW

INFLATION composite (→ TIP vs nominal):
  s10. breakeven_roc     — 3m rate of change of 10Y breakeven
  s11. cpi_momentum      — 12m CPI YoY change (realised inflation)

All constants are read from `config` at call time → optimise.py can patch them.
"""
import pandas as pd
import numpy as np
import config


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _zscore(s: pd.Series, lookback: int) -> pd.Series:
    mu = s.rolling(lookback).mean()
    sd = s.rolling(lookback).std()
    return (s - mu) / sd.clip(lower=1e-6)


# ---------------------------------------------------------------------------
# Individual signals (all → z-score scale)
# ---------------------------------------------------------------------------

def _curve_2s10s(macro: pd.DataFrame) -> pd.Series:
    """Steep curve → positive → favour long duration."""
    return _zscore(macro["spread_2s10s"].ffill(), config.LOOKBACK_SIGNAL).rename("curve_2s10s")


def _curve_10y3m(macro: pd.DataFrame) -> pd.Series:
    """10Y-3M spread. Better leading recession indicator than 2s10s."""
    return _zscore(macro["spread_10y3m"].ffill(), config.LOOKBACK_SIGNAL).rename("curve_10y3m")


def _fed_direction(macro: pd.DataFrame) -> pd.Series:
    """
    3-month change in fed funds rate (ffilled monthly series).
    Rising fed funds → negative for duration (bonds lose when rates rise).
    Invert so positive = rate cuts = good for duration.
    """
    s = macro["fedfunds"].ffill()
    roc = s.diff(63)   # ~3 months of business days
    return _zscore(-roc, config.LOOKBACK_SIGNAL).rename("fed_direction")


def _hy_oas(macro: pd.DataFrame) -> pd.Series:
    """Inverted HY OAS z-score: tight spreads → positive → risk-on."""
    return (-_zscore(macro["hy_oas"].ffill(), config.LOOKBACK_SIGNAL)).rename("hy_oas")


def _vix_regime(macro: pd.DataFrame) -> pd.Series:
    """
    Inverted composite of VIX level z-score and 21-day momentum.
    Low / falling VIX → positive → risk-on → favour credit.
    """
    vix = macro["vix"].ffill()
    level_z = _zscore(vix, config.LOOKBACK_SIGNAL)
    mom_z   = _zscore(vix.diff(21), config.LOOKBACK_SIGNAL)
    composite = 0.6 * level_z + 0.4 * mom_z
    return (-composite).rename("vix_regime")


def _breakeven_roc(macro: pd.DataFrame) -> pd.Series:
    """3-month rate of change of 10Y breakeven inflation. Rising → favour TIP."""
    bei = macro["breakeven_10y"].ffill()
    roc = bei.pct_change(63)
    return _zscore(roc.dropna().reindex(bei.index), config.LOOKBACK_SIGNAL).rename("breakeven_roc")


def _cpi_momentum(macro: pd.DataFrame) -> pd.Series:
    """
    12-month YoY CPI change (realised inflation).
    Rising → positive → favour TIP.
    CPI is monthly → forward-filled to daily.
    """
    cpi = macro["cpi"].ffill()
    yoy = cpi.pct_change(252)   # ~252 business days ≈ 12 months after ffill
    return _zscore(yoy.dropna().reindex(cpi.index), config.LOOKBACK_SIGNAL).rename("cpi_momentum")


def _real_yield_signal(macro: pd.DataFrame) -> pd.Series:
    """
    10Y TIPS real yield (DFII10) direction.
    Rising real yields → bonds lose money (the 2022 driver).
    Return INVERTED 3m change z-score: negative when real yields are rising fast
    → discourages duration exposure.
    """
    ry = macro["real_yield_10y"].ffill()
    roc = ry.diff(63)    # 3-month change in real yield (pp)
    return _zscore(-roc, config.LOOKBACK_SIGNAL).rename("real_yield_signal")


def _labor_market_signal(macro: pd.DataFrame) -> pd.Series:
    """
    Unemployment rate trend (Sahm-rule-inspired).
    Rising unemployment → recession incoming → Fed will cut → bullish duration.
    Returns positive when unemployment is rising (good for long-duration bonds).
    Monthly UNRATE → forward-filled to daily.
    """
    u = macro["unemployment"].ffill()
    # Compare 3m average to 12m minimum (Sahm-rule style)
    avg3m  = u.rolling(63).mean()
    min12m = u.rolling(252).min()
    sahm   = avg3m - min12m          # positive and rising = recession signal
    return _zscore(sahm, config.LOOKBACK_SIGNAL).rename("labor_market")


def _ig_spread_momentum(macro: pd.DataFrame) -> pd.Series:
    """
    Rate of change in IG OAS spreads.
    Rapidly WIDENING IG spreads = credit stress even before levels look extreme.
    Return inverted 3m change z-score: negative when spreads widen fast →
    discourages credit exposure.
    """
    ig = macro["ig_oas"].ffill()
    roc = ig.diff(63)    # 3-month change in IG OAS (bp)
    return _zscore(-roc, config.LOOKBACK_SIGNAL).rename("ig_spread_momentum")


def _fed_qt_signal(macro: pd.DataFrame) -> pd.Series:
    """
    Fed balance sheet momentum (WALCL).
    Expanding balance sheet (QE) → bullish for bonds.
    Shrinking balance sheet (QT) → headwinds for bonds / credit.
    Returns positive when balance sheet is growing.
    Weekly series → forward-filled to daily.
    """
    if "fed_assets" not in macro.columns:
        return pd.Series(0.0, index=macro.index, name="fed_qt_signal")
    fa = macro["fed_assets"].ffill()
    roc = fa.pct_change(63)     # 3-month growth rate
    return _zscore(roc, config.LOOKBACK_SIGNAL).rename("fed_qt_signal")


def _indpro_signal(macro: pd.DataFrame) -> pd.Series:
    """
    Industrial Production Index (INDPRO) — growth proxy.
    Rising industrial production → strong economy → rates rise → bearish bonds.
    Returns positive when production is declining (bullish for bonds / gold).
    Monthly series → forward-filled to daily.
    """
    if "indpro" not in macro.columns:
        return pd.Series(0.0, index=macro.index, name="indpro_signal")
    ip = macro["indpro"].ffill()
    # 3m rate of change, inverted: declining production = positive (bullish bonds)
    roc = ip.pct_change(63)
    return _zscore(-roc, config.LOOKBACK_SIGNAL).rename("indpro_signal")


def _ted_stress_signal(macro: pd.DataFrame) -> pd.Series:
    """
    TED spread (T-bill to Eurodollar, TEDRATE).
    High TED = financial stress = reduce credit exposure.
    Returns inverted z-score: negative when TED is elevated (bearish credit).
    Weekly series → forward-filled to daily.
    """
    if "ted_spread" not in macro.columns:
        return pd.Series(0.0, index=macro.index, name="ted_stress_signal")
    ted = macro["ted_spread"].ffill()
    return (-_zscore(ted, config.LOOKBACK_SIGNAL)).rename("ted_stress_signal")


# ---------------------------------------------------------------------------
# Composite signals
# ---------------------------------------------------------------------------

def compute_all_macro(macro: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a daily DataFrame with three composite signals in [-∞, +∞]:
      duration_z  : positive → steep curve / falling real yields → favour TLT/IEF
      credit_z    : positive → tight spreads / low VIX / QE → favour LQD/HYG
      inflation_z : positive → rising inflation → favour TIP
    Plus raw vix for the hard override in portfolio.py.
    """
    c2s10s  = _curve_2s10s(macro)
    c10y3m  = _curve_10y3m(macro)
    fed     = _fed_direction(macro)
    realyld = _real_yield_signal(macro)
    labor   = _labor_market_signal(macro)
    ism     = _indpro_signal(macro)          # Industrial production deceleration

    hyoas   = _hy_oas(macro)
    igmom   = _ig_spread_momentum(macro)
    vix     = _vix_regime(macro)
    fedqt   = _fed_qt_signal(macro)
    ted     = _ted_stress_signal(macro)     # TED spread financial stress

    bei     = _breakeven_roc(macro)
    cpi     = _cpi_momentum(macro)

    # Composite duration (weighted z-score)
    dur_z = (
        config.W_DURATION_2S10S   * c2s10s
        + config.W_DURATION_10Y3M * c10y3m
        + config.W_DURATION_FED   * fed
        + config.W_DURATION_REALYLD * realyld
        + config.W_DURATION_LABOR   * labor
        + config.W_DURATION_ISM     * ism
    )

    # Composite credit
    credit_z = (
        config.W_CREDIT_HYOAS  * hyoas
        + config.W_CREDIT_IGMOM  * igmom
        + config.W_CREDIT_VIX  * vix
        + config.W_CREDIT_FEDQT * fedqt
        + config.W_CREDIT_TED   * ted
    )

    # Composite inflation
    infl_z = (
        config.W_INFLATION_BEI * bei
        + config.W_INFLATION_CPI * cpi
    )

    out = pd.DataFrame({
        "duration_z":  dur_z,
        "credit_z":    credit_z,
        "inflation_z": infl_z,
        "vix_raw":     macro["vix"].ffill(),
    }).ffill()

    return out


# ---------------------------------------------------------------------------
# Price-based signals
# ---------------------------------------------------------------------------

def momentum(prices: pd.DataFrame) -> pd.DataFrame:
    """12-1 month cross-sectional momentum. Positive → include."""
    past   = prices.shift(config.MOMENTUM_SKIP)
    longer = prices.shift(config.MOMENTUM_WINDOW)
    return (past / longer - 1).rename(columns=lambda c: c)


def rolling_vol(prices: pd.DataFrame) -> pd.DataFrame:
    """Annualised rolling daily return volatility (LOOKBACK_VOL window)."""
    return prices.pct_change().rolling(config.LOOKBACK_VOL).std() * np.sqrt(252)


def resample_to_month_end(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("ME").last()
