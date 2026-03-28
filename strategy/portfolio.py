"""
Convert composite signals into monthly target weights.

Credit bucket (LQD, HYG, ANGL, BKLN, EMB) is weighted by inverse-vol within
the bucket — this naturally favours LQD (low vol) in stress and shifts to
BKLN/ANGL/EMB (higher yield, similar vol) in calm periods.

VIX hard-cap prevents runaway credit exposure in stressed markets.
"""
import pandas as pd
import numpy as np
import config


# ---------------------------------------------------------------------------
# Sub-allocators
# ---------------------------------------------------------------------------

def _credit_frac(credit_z: float, vix_raw: float) -> float:
    """
    Soft-scale credit allocation by credit_z; hard-cap by VIX level.
    """
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
    """
    Allocate c_frac across CREDIT_ETFS using inverse-vol within the bucket.
    ETFs without vol data (not yet listed) are excluded automatically.
    """
    available = [e for e in config.CREDIT_ETFS if e in vol.index and pd.notna(vol[e]) and vol[e] > 0]
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

    # ── 1. Credit bucket — inverse-vol within bucket ───────────────────────
    c_frac = _credit_frac(credit_z, vix_raw)
    for etf, alloc in _credit_weights_inv_vol(c_frac, vol).items():
        w[etf] = alloc

    # ── 2. TIP allocation ─────────────────────────────────────────────────
    rates_budget = 1.0 - c_frac
    tip_frac     = _tip_frac(inflation_z, rates_budget)
    w["TIP"]     = tip_frac
    pure_rates   = rates_budget - tip_frac

    # ── 3. Duration bucket ────────────────────────────────────────────────
    for etf, frac in _duration_weights(duration_z).items():
        w[etf] = pure_rates * frac

    # ── 4. Momentum filter ────────────────────────────────────────────────
    for etf in config.ETF_UNIVERSE:
        if etf in mom.index and pd.notna(mom[etf]) and mom[etf] < 0:
            w[etf] = 0.0

    # ── 5. Fallback ───────────────────────────────────────────────────────
    if w.sum() < 1e-6:
        w["SHY"] = 1.0
        return w

    # ── 6. Blend signal weights with inverse-vol weights ──────────────────
    survivors    = w[w > 0].index
    inv_vol      = 1.0 / vol.reindex(survivors).clip(lower=0.001)
    iv_norm      = inv_vol / inv_vol.sum()
    signal_norm  = w[survivors] / w[survivors].sum()

    blend        = config.SIGNAL_BLEND
    blended      = blend * signal_norm + (1.0 - blend) * iv_norm
    w[survivors] = blended / blended.sum()

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
