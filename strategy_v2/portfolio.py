"""
v2 portfolio — five allocation buckets (always sum to 1.0):

  1. COMMODITIES (GLD, SLV, PDBC, DBA)
     v2: SLV added. USD signal dampens budget when dollar is rising.
     Momentum-gated, inverse-vol weighted within bucket.

  2. EQUITY satellite (SPY)
     v2: NEW. Active only in growth/risk-on regime:
     credit spreads tight + VIX below risk-on threshold + SPY momentum positive.
     Carved from remaining pool after commodities.

  3. REAL ESTATE (VNQ)
     v2: NEW. Small allocation when inflation and credit are both positive.
     Benefits from hard assets + cap rate compression.
     Carved from remaining pool.

  4. CREDIT (LQD, HYG, ANGL, SJNK, BKLN, EMB, PFF)
     Size ∝ credit_z, hard-capped by VIX.
     Inverse-vol weighted within bucket.

  5. INFLATION (TIP, VTIP)
     v2: Split between TIP (full duration) and VTIP (short duration) based on
     duration_z. Rising rates → shift toward VTIP to limit duration bleed.

  6. DURATION (TLT, IEF, SHY)
     Remainder — the defensive "cash pool".

Cash during drawdowns is handled by the drawdown overlay in backtest.py.
"""
import logging
import pandas as pd
import numpy as np
import config_v2 as config

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
    v2 commodity basket: GLD, SLV, PDBC, DBA.

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


def _equity_frac(
    credit_z: float,
    vix_raw: float,
    mom: pd.Series,
) -> float:
    """
    v2 equity satellite: SPY.

    Active only when all three conditions hold:
      1. VIX < VIX_RISK_ON  (confirmed risk-on environment)
      2. SPY momentum positive  (trend filter)
      3. credit_z > 0  (tight spreads = growth / risk appetite)

    Allocation scales with credit_z via tanh. Max = MAX_EQUITY_ALLOC of
    the remaining pool (after commodities).
    """
    if vix_raw >= config.VIX_RISK_ON:
        return 0.0

    # SPY momentum gate
    if "SPY" in mom.index and pd.notna(mom["SPY"]) and mom["SPY"] < 0:
        return 0.0

    eq_budget = config.MAX_EQUITY_ALLOC * (0.5 + 0.5 * np.tanh(credit_z * 0.5))
    return float(np.clip(eq_budget, 0.0, config.MAX_EQUITY_ALLOC))


def _realestate_frac(
    inflation_z: float,
    credit_z: float,
    mom: pd.Series,
) -> float:
    """
    v2 real estate satellite: VNQ.

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
    v2: Split the TIP budget between TIP (full ~7yr duration) and VTIP (~2.5yr duration).

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


def _duration_weights(duration_z: float) -> dict:
    scores   = {"TLT": 1.5 * duration_z, "IEF": 0.0, "SHY": -1.0 * duration_z}
    vals     = np.array(list(scores.values()))
    exp_vals = np.exp(np.clip(vals - vals.max(), -10, 0))
    probs    = exp_vals / exp_vals.sum()
    return dict(zip(scores.keys(), probs))


# ---------------------------------------------------------------------------
# Main weight builder
# ---------------------------------------------------------------------------

def build_weights(
    duration_z: float,
    credit_z: float,
    inflation_z: float,
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

    # ── 2. Equity satellite (SPY) — from remaining ────────────────────────
    eq_frac   = _equity_frac(credit_z, vix_raw, mom)
    eq_budget = eq_frac * remaining
    if "SPY" in w.index and eq_budget > config.MIN_WEIGHT_THRESHOLD:
        w["SPY"] = eq_budget

    # ── 3. Real estate (VNQ) — from remaining ─────────────────────────────
    re_frac   = _realestate_frac(inflation_z, credit_z, mom)
    re_budget = min(re_frac * remaining, remaining - eq_budget)
    re_budget = max(re_budget, 0.0)
    if "VNQ" in w.index and re_budget > config.MIN_WEIGHT_THRESHOLD:
        w["VNQ"] = re_budget

    bonds_remaining = remaining - eq_budget - re_budget

    # ── 4. Credit bucket ──────────────────────────────────────────────────
    c_frac = _credit_frac(credit_z, vix_raw) * bonds_remaining
    for etf, alloc in _credit_weights_inv_vol(c_frac, vol).items():
        w[etf] = alloc

    rates_budget = bonds_remaining - c_frac

    # ── 5. TIP/VTIP split ─────────────────────────────────────────────────
    tip_total = _tip_frac(inflation_z, rates_budget)
    for etf, alloc in _tip_vtip_weights(tip_total, duration_z).items():
        if etf in w.index:
            w[etf] = alloc

    dur_budget = rates_budget - tip_total

    # ── 6. Duration bucket ────────────────────────────────────────────────
    for etf, frac in _duration_weights(duration_z).items():
        w[etf] = dur_budget * frac

    # ── 7. Momentum filter — bonds, credit, and TIPS only ─────────────────
    filter_etfs = config.DURATION_ETFS + config.INFLATION_ETFS + config.CREDIT_ETFS
    for etf in filter_etfs:
        if etf in mom.index and pd.notna(mom[etf]) and mom[etf] < 0:
            w[etf] = 0.0

    # ── 8. Fallback — park in SHY ─────────────────────────────────────────
    if w.sum() < config.MIN_WEIGHT_THRESHOLD:
        w["SHY"] = 1.0
        return w

    # ── 9. Blend signal weights with inverse-vol (bonds/credit only) ──────
    alt_etfs    = config.HEDGE_ETFS + config.EQUITY_ETFS + config.REAL_ASSET_ETFS
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

    # ── 10. Ensure full investment ────────────────────────────────────────
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
        w = build_weights(
            duration_z  = row["duration_z"],
            credit_z    = row["credit_z"],
            inflation_z = row["inflation_z"],
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
