"""
Convert composite signals into monthly target weights.

Construction logic:
  1. Compute raw credit allocation from credit_z — capped by VIX hard override
  2. Compute TIP allocation from inflation_z
  3. Allocate remaining to duration bucket (TLT/IEF/SHY) via duration_z
  4. Apply momentum filter (zero out negative-momentum ETFs)
  5. Blend signal weights with inverse-vol weights (config.SIGNAL_BLEND)
  6. Normalise to sum = 1

All config values are read at call time → optimise.py can patch them.
"""
import pandas as pd
import numpy as np
import config


# ---------------------------------------------------------------------------
# Sub-allocators
# ---------------------------------------------------------------------------

def _credit_frac(credit_z: float, vix_raw: float) -> float:
    """
    Credit allocation, soft-scaled by credit_z then hard-capped by VIX.
    VIX > VIX_RISK_OFF → never more than 10 % credit.
    VIX < VIX_RISK_ON  → full credit budget allowed.
    """
    soft = config.MAX_CREDIT_ALLOC * (0.5 + 0.5 * np.tanh(credit_z))
    frac = float(np.clip(soft, 0.0, config.MAX_CREDIT_ALLOC))

    if vix_raw > config.VIX_RISK_OFF:
        frac = min(frac, 0.10)
    elif vix_raw < config.VIX_RISK_ON:
        pass   # full credit allowed
    else:
        # Linear interpolation between 10 % and full in VIX_RISK_ON .. VIX_RISK_OFF band
        t    = (vix_raw - config.VIX_RISK_ON) / (config.VIX_RISK_OFF - config.VIX_RISK_ON)
        cap  = config.MAX_CREDIT_ALLOC * (1.0 - t) + 0.10 * t
        frac = min(frac, cap)

    return frac


def _tip_frac(inflation_z: float, rates_budget: float) -> float:
    """TIP allocation within the rates bucket, scaled by inflation_z."""
    soft = config.MAX_TIP_ALLOC * (0.5 + 0.5 * np.tanh(inflation_z * 0.5))
    return float(np.clip(soft * rates_budget, 0.0, config.MAX_TIP_ALLOC))


def _duration_weights(duration_z: float) -> dict:
    """
    Softmax allocation across TLT / IEF / SHY driven by duration_z.
    High positive → TLT heavy.  Negative → SHY heavy.
    """
    scores = {
        "TLT":  1.5 * duration_z,
        "IEF":  0.0,
        "SHY": -1.0 * duration_z,
    }
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
    vix_raw: float,
    mom: pd.Series,
    vol: pd.Series,
) -> pd.Series:
    w = pd.Series(0.0, index=config.ETF_UNIVERSE)

    # ── 1. Credit bucket (with VIX hard cap) ──────────────────────────────
    c_frac       = _credit_frac(credit_z, vix_raw)
    rates_budget = 1.0 - c_frac
    w["LQD"]     = c_frac * config.CREDIT_LQD_SPLIT
    w["HYG"]     = c_frac * (1.0 - config.CREDIT_LQD_SPLIT)

    # ── 2. TIP allocation ─────────────────────────────────────────────────
    tip_frac   = _tip_frac(inflation_z, rates_budget)
    w["TIP"]   = tip_frac
    pure_rates = rates_budget - tip_frac

    # ── 3. Duration bucket ────────────────────────────────────────────────
    for etf, frac in _duration_weights(duration_z).items():
        w[etf] = pure_rates * frac

    # ── 4. Momentum filter ────────────────────────────────────────────────
    for etf in config.ETF_UNIVERSE:
        if etf in mom.index and pd.notna(mom[etf]) and mom[etf] < 0:
            w[etf] = 0.0

    # ── 5. Fallback: hold SHY if everything filtered ───────────────────────
    if w.sum() < 1e-6:
        w["SHY"] = 1.0
        return w

    # ── 6. Blend signal weights with inverse-vol weights ──────────────────
    survivors   = w[w > 0].index
    inv_vol     = 1.0 / vol.reindex(survivors).clip(lower=0.001)
    iv_norm     = inv_vol / inv_vol.sum()
    signal_norm = w[survivors] / w[survivors].sum()

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
