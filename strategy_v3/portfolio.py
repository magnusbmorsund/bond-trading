"""
v3 portfolio — six allocation buckets (always sum to 1.0):

  1. COMMODITIES (GLD, SLV, PDBC, DBA)
     USD signal dampens budget when dollar is rising.
     Momentum-gated, inverse-vol weighted within bucket.

  2. MANAGED FUTURES (DBMF)
     v3: NEW. Active when growth is declining AND trend is strong (|duration_z| > 1)
     AND DBMF momentum is positive. Carved from remaining pool after commodities.

  3. EQUITY (MTUM, SPY)
     v3: MTUM added alongside SPY. Active only in growth/risk-on regime:
     credit_z > 0 AND growth_z > 0 AND VIX below risk-on threshold.
     MTUM fraction scales with growth_z via tanh.
     Carved from remaining pool after commodities and DBMF.

  4. REAL ESTATE (VNQ)
     Small allocation when inflation and credit are both positive.
     Carved from remaining pool.

  5. CREDIT (LQD, HYG, EMB, PFF)
     Size ∝ credit_z, hard-capped by VIX.
     Inverse-vol weighted within bucket.

  6. INFLATION (TIP, VTIP)
     Split between TIP and VTIP based on duration_z (v2 logic unchanged).

  7. DURATION (EDV, TLT, IEF, JPST, SHY)
     v3: EDV and JPST added. Softmax allocation with availability-aware scoring.
     Remainder — the defensive pool.

Cash during drawdowns is handled by the drawdown overlay in backtest.py.
"""
import logging
import pandas as pd
import numpy as np
import config_v3 as config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-allocators
# ---------------------------------------------------------------------------

def _commodity_weights(
    inflation_z: float,
    duration_z: float,
    usd_z: float,
    mom: pd.Series,
    vol: pd.Series,
) -> dict:
    """
    v3 commodity basket: GLD, SLV, PDBC, DBA.

    Budget is dampened when USD is rising (commodities priced in USD —
    a strong dollar compresses dollar-denominated commodity prices).
    Dampening: budget *= (1 + W_COMMODITY_USD * tanh(-usd_z * 0.5))
    At usd_z = +2: budget reduced ~22%. At usd_z = -2: budget boosted ~22%.
    """
    comm_signal = 0.5 * inflation_z + 0.5 * duration_z

    raw_budget = config.MAX_ALT_ALLOC * (0.5 + 0.5 * np.tanh(comm_signal * config.TANH_COMM_SCALE))

    # USD drag: rising dollar reduces commodity budget
    usd_drag   = config.W_COMMODITY_USD * float(np.tanh(-usd_z * 0.5))
    budget     = float(np.clip(raw_budget * (1.0 + usd_drag), 0.0, config.MAX_ALT_ALLOC))

    if budget < config.MIN_WEIGHT_THRESHOLD:
        return {}

    available = []
    for etf in config.HEDGE_ETFS:
        mom_neg = etf in mom.index and pd.notna(mom[etf]) and mom[etf] < 0
        if not mom_neg and etf in vol.index and pd.notna(vol[etf]) and vol[etf] > 0:
            available.append(etf)

    if not available:
        return {}

    inv_vols = {e: 1.0 / max(vol[e], config.MIN_VOL_CLIP) for e in available}
    total_iv = sum(inv_vols.values())
    return {e: budget * (iv / total_iv) for e, iv in inv_vols.items()}


def _managed_futures_frac(
    growth_z: float,
    duration_z: float,
    vix_raw: float,
    mom: pd.Series,
) -> float:
    """
    v3 managed futures satellite: DBMF.

    Active only when:
      1. DBMF has positive momentum (trend filter)
      2. Growth is declining (growth_z negative) — managed futures tend to outperform
         in macro stress / trend-following environments
      3. Trend strength: |duration_z| > 1 (strong rate trend)

    Returns fraction of remaining pool to allocate to DBMF.
    """
    if (
        "DBMF" not in mom.index
        or pd.isna(mom.get("DBMF", np.nan))
        or mom["DBMF"] < 0
    ):
        return 0.0

    growth_signal = float(np.tanh(-growth_z * config.MF_SIGNAL_SCALE))  # positive when growth declining
    trend_signal  = float(np.tanh(abs(duration_z) * 0.5 - 0.5))         # positive when |dur_z| > 1
    combined      = max(growth_signal, 0.0) * 0.7 + max(trend_signal, 0.0) * 0.3
    return float(np.clip(config.MAX_MANAGED_FUTURES_ALLOC * combined, 0.0, config.MAX_MANAGED_FUTURES_ALLOC))


def _equity_weights_v3(
    credit_z: float,
    growth_z: float,
    vix_raw: float,
    mom: pd.Series,
    vol: pd.Series,
) -> tuple:
    """
    v3 equity satellite: MTUM + SPY.

    Active only when all three conditions hold:
      1. VIX < VIX_RISK_ON  (confirmed risk-on environment)
      2. growth_z > 0  (expanding economy)
      3. credit_z > 0  (tight spreads = risk appetite)

    MTUM fraction scales with growth_z via tanh — stronger growth momentum = more MTUM.
    Both MTUM and SPY individually require positive momentum to be included.

    Returns (total_eq_budget: float, weights: dict)
    """
    if vix_raw >= config.VIX_RISK_ON or growth_z <= 0 or credit_z <= 0:
        return 0.0, {}

    eq_budget = config.MAX_EQUITY_ALLOC * (0.5 + 0.5 * np.tanh(0.5 * growth_z))
    eq_budget = float(np.clip(eq_budget, 0.0, config.MAX_EQUITY_ALLOC))

    if eq_budget < config.MIN_WEIGHT_THRESHOLD:
        return 0.0, {}

    mtum_frac = float(np.clip(0.5 + 0.25 * np.tanh(growth_z), 0.0, 1.0))

    mtum_ok = (
        "MTUM" in mom.index
        and pd.notna(mom.get("MTUM", np.nan))
        and mom["MTUM"] >= 0
    )
    spy_ok = (
        "SPY" in mom.index
        and pd.notna(mom.get("SPY", np.nan))
        and mom["SPY"] >= 0
    )

    if not mtum_ok and not spy_ok:
        return 0.0, {}

    if mtum_ok and spy_ok:
        weights = {
            "MTUM": eq_budget * mtum_frac,
            "SPY":  eq_budget * (1.0 - mtum_frac),
        }
    elif mtum_ok:
        weights = {"MTUM": eq_budget}
    else:
        weights = {"SPY": eq_budget}

    return sum(weights.values()), weights


def _realestate_frac(
    inflation_z: float,
    credit_z: float,
    mom: pd.Series,
) -> float:
    """
    Real estate satellite: VNQ.

    REITs benefit when both inflation is rising (hard-asset repricing)
    and credit spreads are tight (cap-rate compression, financing cheap).
    Momentum gate prevents holding in downtrends.
    """
    re_signal = 0.5 * (inflation_z + credit_z)
    re_budget = config.MAX_REALESTATE_ALLOC * (0.5 + 0.5 * np.tanh(re_signal * 0.5))
    re_budget = float(np.clip(re_budget, 0.0, config.MAX_REALESTATE_ALLOC))

    # VNQ momentum gate
    if "VNQ" in mom.index and pd.notna(mom["VNQ"]) and mom["VNQ"] < 0:
        return 0.0

    return re_budget


def _credit_frac(credit_z: float, vix_raw: float) -> float:
    soft = config.MAX_CREDIT_ALLOC * (0.5 + 0.5 * np.tanh(credit_z))
    frac = float(np.clip(soft, 0.0, config.MAX_CREDIT_ALLOC))
    if vix_raw > config.VIX_RISK_OFF:
        frac = min(frac, 0.10)
    elif vix_raw >= config.VIX_RISK_ON:
        t    = (vix_raw - config.VIX_RISK_ON) / (config.VIX_RISK_OFF - config.VIX_RISK_ON)
        cap  = config.MAX_CREDIT_ALLOC * (1.0 - t) + 0.10 * t
        frac = min(frac, cap)
    return frac


def _credit_weights_inv_vol(c_frac: float, vol: pd.Series) -> dict:
    available = [e for e in config.CREDIT_ETFS
                 if e in vol.index and pd.notna(vol[e]) and vol[e] > 0]
    if not available:
        return {}
    inv_vol = 1.0 / vol[available].clip(lower=config.INV_VOL_CLIP)
    weights  = inv_vol / inv_vol.sum()
    return {etf: c_frac * w for etf, w in weights.items()}


def _tip_vtip_weights(tip_budget: float, duration_z: float) -> dict:
    """
    Split the TIP budget between TIP (full ~7yr duration) and VTIP (~2.5yr duration).

    When rates are rising (duration_z < 0), shift toward VTIP to reduce duration bleed:
      vtip_frac = clip(0.5 - 0.5 * tanh(duration_z * VTIP_DURATION_SCALE), 0, 1)

    Examples:
      duration_z = +2.0 → vtip_frac ≈ 0.08  (mostly TIP — falling rates, duration is fine)
      duration_z =  0.0 → vtip_frac = 0.50  (50/50 split)
      duration_z = -2.0 → vtip_frac ≈ 0.92  (mostly VTIP — rising rates, avoid duration)
    """
    if tip_budget < config.MIN_WEIGHT_THRESHOLD:
        return {}
    vtip_frac = float(np.clip(
        0.5 - 0.5 * np.tanh(duration_z * config.VTIP_DURATION_SCALE), 0.0, 1.0
    ))
    return {
        "TIP":  tip_budget * (1.0 - vtip_frac),
        "VTIP": tip_budget * vtip_frac,
    }


def _tip_frac(inflation_z: float, rates_budget: float) -> float:
    soft = config.MAX_TIP_ALLOC * (0.5 + 0.5 * np.tanh(inflation_z * 0.5))
    return float(np.clip(soft * rates_budget, 0.0, config.MAX_TIP_ALLOC))


def _duration_weights_v3(duration_z: float, vol: pd.Series) -> dict:
    """
    v3 duration bucket: EDV, TLT, IEF, JPST, SHY.

    Scores reflect duration exposure — higher duration ETFs score higher
    when duration_z is positive (bullish rates environment):
      EDV  : EDV_DURATION_SCORE * duration_z  (~24yr duration)
      TLT  : 1.5 * duration_z                (~17yr duration)
      IEF  : 0.0                              (~7yr duration — neutral)
      JPST : -0.5 * duration_z               (~0.5yr duration — rising rates hedge)
      SHY  : -1.0 * duration_z               (~2yr duration — cash-like)

    Availability-aware: only includes ETFs present in vol with valid data.
    Applies softmax over scores for smooth transitions.
    Falls back to {"SHY": 1.0} if no valid ETFs found.
    """
    scores = {
        "EDV":  config.EDV_DURATION_SCORE * duration_z,
        "TLT":  1.5  * duration_z,
        "IEF":  0.0,
        "JPST": -0.5 * duration_z,
        "SHY":  -1.0 * duration_z,
    }

    # Availability filter — only include ETFs with valid vol data
    candidates = {}
    for etf, score in scores.items():
        if (
            etf in config.DURATION_ETFS
            and etf in vol.index
            and pd.notna(vol[etf])
            and vol[etf] > 0
        ):
            candidates[etf] = score

    if not candidates:
        return {"SHY": 1.0}

    vals     = np.array(list(candidates.values()))
    exp_vals = np.exp(np.clip(vals - vals.max(), -10, 0))
    probs    = exp_vals / exp_vals.sum()
    return dict(zip(candidates.keys(), probs))


# ---------------------------------------------------------------------------
# Main weight builder
# ---------------------------------------------------------------------------

def build_weights(
    duration_z: float,
    credit_z: float,
    inflation_z: float,
    growth_z: float,
    usd_z: float,
    vix_raw: float,
    mom: pd.Series,
    vol: pd.Series,
) -> pd.Series:
    w = pd.Series(0.0, index=config.ETF_UNIVERSE)

    # ── 1. Commodity basket (GLD, SLV, PDBC, DBA) ────────────────────────
    comm = _commodity_weights(inflation_z, duration_z, usd_z, mom, vol)
    comm_total = 0.0
    for etf, alloc in comm.items():
        if etf in w.index:
            w[etf]      = alloc
            comm_total += alloc

    remaining = 1.0 - comm_total

    # ── 2. Managed futures (DBMF) — from remaining ───────────────────────
    mf_frac   = _managed_futures_frac(growth_z, duration_z, vix_raw, mom)
    mf_budget = mf_frac * remaining
    if "DBMF" in w.index and mf_budget > config.MIN_WEIGHT_THRESHOLD:
        w["DBMF"] = mf_budget

    # ── 3. Equity satellite (MTUM, SPY) — from remaining ─────────────────
    eq_total, eq_weights = _equity_weights_v3(credit_z, growth_z, vix_raw, mom, vol)
    eq_budget = eq_total  # already scaled to MAX_EQUITY_ALLOC fraction
    for etf, alloc in eq_weights.items():
        if etf in w.index:
            w[etf] = alloc

    # ── 4. Real estate (VNQ) — from remaining ─────────────────────────────
    re_frac   = _realestate_frac(inflation_z, credit_z, mom)
    re_budget = min(re_frac * remaining, remaining - mf_budget - eq_budget)
    re_budget = max(re_budget, 0.0)
    if "VNQ" in w.index and re_budget > config.MIN_WEIGHT_THRESHOLD:
        w["VNQ"] = re_budget

    bonds_remaining = remaining - mf_budget - eq_budget - re_budget

    # ── 5. Credit bucket ──────────────────────────────────────────────────
    c_frac = _credit_frac(credit_z, vix_raw) * bonds_remaining
    for etf, alloc in _credit_weights_inv_vol(c_frac, vol).items():
        w[etf] = alloc

    rates_budget = bonds_remaining - c_frac

    # ── 6. TIP/VTIP split ─────────────────────────────────────────────────
    tip_total = _tip_frac(inflation_z, rates_budget)
    for etf, alloc in _tip_vtip_weights(tip_total, duration_z).items():
        if etf in w.index:
            w[etf] = alloc

    dur_budget = rates_budget - tip_total

    # ── 7. Duration bucket (v3: EDV, TLT, IEF, JPST, SHY) ───────────────
    for etf, frac in _duration_weights_v3(duration_z, vol).items():
        w[etf] = dur_budget * frac

    # ── 8. Momentum filter — bonds, credit, and TIPS only ─────────────────
    filter_etfs = config.DURATION_ETFS + config.INFLATION_ETFS + config.CREDIT_ETFS
    for etf in filter_etfs:
        if etf in mom.index and pd.notna(mom[etf]) and mom[etf] < 0:
            w[etf] = 0.0

    # ── 9. Fallback — park in SHY ─────────────────────────────────────────
    if w.sum() < config.MIN_WEIGHT_THRESHOLD:
        w["SHY"] = 1.0
        return w

    # ── 10. Blend signal weights with inverse-vol (bonds/credit only) ─────
    alt_etfs    = config.HEDGE_ETFS + config.EQUITY_ETFS + config.REAL_ASSET_ETFS + config.MANAGED_FUTURES_ETFS
    fixed_etfs  = [e for e in alt_etfs if e in w.index and w[e] > 0]
    blend_etfs  = [e for e in config.ETF_UNIVERSE
                   if e not in alt_etfs and e in w.index and w[e] > 0]
    fixed_total = w[fixed_etfs].sum() if fixed_etfs else 0.0
    free_budget = 1.0 - fixed_total

    if blend_etfs and free_budget > config.MIN_WEIGHT_THRESHOLD:
        inv_vol     = 1.0 / vol.reindex(blend_etfs).clip(lower=config.INV_VOL_CLIP)
        iv_norm     = inv_vol / inv_vol.sum()
        signal_norm = w[blend_etfs] / w[blend_etfs].sum()
        blend       = config.SIGNAL_BLEND
        blended     = blend * signal_norm + (1.0 - blend) * iv_norm
        w[blend_etfs] = blended / blended.sum() * free_budget

    # ── 11. Ensure full investment ────────────────────────────────────────
    residual = 1.0 - w.sum()
    if residual > config.MIN_WEIGHT_THRESHOLD and "SHY" in w.index:
        w["SHY"] = w["SHY"] + residual

    return w


# ---------------------------------------------------------------------------
# Vectorised over all rebalance dates
# ---------------------------------------------------------------------------

def build_weight_series(
    macro_monthly: pd.DataFrame,
    mom_monthly: pd.DataFrame,
    vol_monthly: pd.DataFrame,
) -> pd.DataFrame:
    records = []
    for date in macro_monthly.index:
        row = macro_monthly.loc[date]
        if row[["duration_z", "credit_z", "inflation_z"]].isna().any():
            continue
        if mom_monthly.loc[date].isna().all():
            continue
        if pd.notna(row["vix_raw"]):
            vix_raw = row["vix_raw"]
        else:
            logger.warning("VIX missing on %s — using neutral default (20.0)", date.date())
            vix_raw = 20.0
        if "usd_z" in row.index and pd.notna(row["usd_z"]):
            usd_z = float(row["usd_z"])
        else:
            logger.debug("USD signal missing on %s — using neutral (0.0)", date.date())
            usd_z = 0.0
        growth_z = float(row["growth_z"]) if "growth_z" in row.index and pd.notna(row["growth_z"]) else 0.0
        w = build_weights(
            duration_z  = row["duration_z"],
            credit_z    = row["credit_z"],
            inflation_z = row["inflation_z"],
            growth_z    = growth_z,
            usd_z       = usd_z,
            vix_raw     = vix_raw,
            mom         = mom_monthly.loc[date],
            vol         = vol_monthly.loc[date],
        )
        w.name = date
        records.append(w)

    if not records:
        raise ValueError("build_weight_series produced no rows — check that macro signals have valid data.")

    return pd.DataFrame(records)
