"""
Convert composite signals into monthly target weights.

Five allocation buckets (always sum to 1.0):

  1. COMMODITIES (GLD, PDBC, DBA, DBB)
     Primary alpha source — momentum-gated, inverse-vol weighted within bucket.
     GLD:  monetary metal — real yields falling + inflation rising
     PDBC: diversified (energy+metals+ag) — growth/inflation cycles, corr 0.27 to GLD
     DBA:  agriculture — food inflation, lowest GLD correlation (0.16), low vol
     DBB:  base metals — industrial/construction demand, corr 0.29 to GLD
     Only holds an ETF when its 12-1 month momentum is positive.
     Fixed income is the defensive "cash pool".

  2. CREDIT  (LQD, HYG, ANGL, SJNK, BKLN, EMB, PFF)
     Size ∝ credit_z, hard-capped by VIX.
     Weighted by inverse-vol within bucket.

  3. INFLATION (TIP)
     Size ∝ inflation_z.

  4. DURATION (TLT, IEF, SHY)
     Remainder — the defensive "cash pool".
     Duration tilt within bucket via softmax on duration_z.

Cash during drawdowns is handled by the drawdown overlay in backtest.py, not here.
"""
import pandas as pd
import numpy as np
import config


# ---------------------------------------------------------------------------
# Sub-allocators
# ---------------------------------------------------------------------------

def _commodity_weights(
    inflation_z: float,
    duration_z: float,
    mom: pd.Series,
    vol: pd.Series,
) -> dict:
    """
    Commodity / materials basket allocation.

    Signal: commodities are collectively bullish when real yields fall (duration_z > 0)
    AND/OR inflation rises (inflation_z > 0). Both conditions drive monetary metals,
    energy, base metals, and agriculture higher.

    Allocation rules:
      - Total budget grows smoothly from 0 to MAX_ALT_ALLOC with signal strength.
      - Only hold an ETF when its 12-1 month momentum is positive (hard gate).
      - Within the budget, weight by inverse-vol (lower-vol = larger weight).
        GLD naturally dominates (vol ~16%) vs PDBC (~18%), DBB (~20%), DBA (~13%).
    """
    # Combined commodity signal: inflation-bullish + real-yield-falling
    comm_signal = 0.5 * inflation_z + 0.5 * duration_z

    # Total commodity budget
    raw_budget = config.MAX_ALT_ALLOC * (0.5 + 0.5 * np.tanh(comm_signal * 0.8))
    budget = float(np.clip(raw_budget, 0.0, config.MAX_ALT_ALLOC))
    if budget < 1e-4:
        return {}

    # Filter to ETFs with positive momentum and available vol
    available = []
    for etf in config.HEDGE_ETFS:
        mom_neg = etf in mom.index and pd.notna(mom[etf]) and mom[etf] < 0
        if not mom_neg and etf in vol.index and pd.notna(vol[etf]) and vol[etf] > 0:
            available.append(etf)

    if not available:
        return {}

    # Inverse-vol weighting within available ETFs
    inv_vols = {e: 1.0 / max(vol[e], 0.01) for e in available}
    total_iv  = sum(inv_vols.values())
    return {e: budget * (iv / total_iv) for e, iv in inv_vols.items()}


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


def _tip_frac(inflation_z: float, rates_budget: float) -> float:
    soft = config.MAX_TIP_ALLOC * (0.5 + 0.5 * np.tanh(inflation_z * 0.5))
    return float(np.clip(soft * rates_budget, 0.0, config.MAX_TIP_ALLOC))


def _duration_weights(duration_z: float) -> dict:
    scores   = {"TLT": 1.5 * duration_z, "IEF": 0.0, "SHY": -1.0 * duration_z}
    vals     = np.array(list(scores.values()))
    exp_vals = np.exp(np.clip(vals - vals.max(), -10, 0))
    probs    = exp_vals / exp_vals.sum()
    return dict(zip(scores.keys(), probs))


def _credit_weights_inv_vol(c_frac: float, vol: pd.Series) -> dict:
    available = [e for e in config.CREDIT_ETFS
                 if e in vol.index and pd.notna(vol[e]) and vol[e] > 0]
    if not available:
        return {}
    inv_vol = 1.0 / vol[available].clip(lower=0.001)
    weights  = inv_vol / inv_vol.sum()
    return {etf: c_frac * w for etf, w in weights.items()}


# ---------------------------------------------------------------------------
# Main weight builder
# ---------------------------------------------------------------------------

def build_weights(
    duration_z: float,
    credit_z: float,
    inflation_z: float,
    vix_raw: float,
    mom: pd.Series,
    vol: pd.Series,
) -> pd.Series:
    w = pd.Series(0.0, index=config.ETF_UNIVERSE)

    # ── 1. Commodity basket (primary alpha — ride the upswings) ──────────
    comm = _commodity_weights(inflation_z, duration_z, mom, vol)
    comm_total = 0.0
    for etf, alloc in comm.items():
        if etf in w.index:
            w[etf]      = alloc
            comm_total += alloc

    remaining = 1.0 - comm_total

    # ── 2. Credit bucket ──────────────────────────────────────────────────
    c_frac = _credit_frac(credit_z, vix_raw) * remaining
    for etf, alloc in _credit_weights_inv_vol(c_frac, vol).items():
        w[etf] = alloc

    rates_budget = remaining - c_frac

    # ── 3. TIP allocation ─────────────────────────────────────────────────
    tip_frac = _tip_frac(inflation_z, rates_budget)
    w["TIP"]  = tip_frac
    dur_budget = rates_budget - tip_frac

    # ── 4. Duration bucket (defensive "cash pool") ────────────────────────
    for etf, frac in _duration_weights(duration_z).items():
        w[etf] = dur_budget * frac

    # ── 5. Momentum filter — zero out bonds/credit with negative momentum ──
    # HEDGE_ETFS already filtered in _commodity_weights; apply to rest
    for etf in config.DURATION_ETFS + [config.INFLATION_ETF] + config.CREDIT_ETFS:
        if etf in mom.index and pd.notna(mom[etf]) and mom[etf] < 0:
            w[etf] = 0.0

    # ── 6. Fallback — park in SHY (shortest duration, lowest risk) ────────
    if w.sum() < 1e-6:
        w["SHY"] = 1.0
        return w

    # ── 7. Blend signal weights with inverse-vol weights ──────────────────
    # Exclude commodity basket from blending — already inv-vol weighted
    fixed_etfs = [e for e in config.HEDGE_ETFS if e in w.index and w[e] > 0]
    blend_etfs = [e for e in config.ETF_UNIVERSE
                  if e not in config.HEDGE_ETFS and e in w.index and w[e] > 0]
    fixed_total = w[fixed_etfs].sum() if fixed_etfs else 0.0
    free_budget = 1.0 - fixed_total

    if blend_etfs and free_budget > 1e-6:
        inv_vol     = 1.0 / vol.reindex(blend_etfs).clip(lower=0.001)
        iv_norm     = inv_vol / inv_vol.sum()
        signal_norm = w[blend_etfs] / w[blend_etfs].sum()
        blend       = config.SIGNAL_BLEND
        blended     = blend * signal_norm + (1.0 - blend) * iv_norm
        w[blend_etfs] = blended / blended.sum() * free_budget

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
        w = build_weights(
            duration_z  = row["duration_z"],
            credit_z    = row["credit_z"],
            inflation_z = row["inflation_z"],
            vix_raw     = row["vix_raw"] if pd.notna(row["vix_raw"]) else 20.0,
            mom         = mom_monthly.loc[date],
            vol         = vol_monthly.loc[date],
        )
        w.name = date
        records.append(w)
    return pd.DataFrame(records)
