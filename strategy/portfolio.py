"""
Convert signals into target portfolio weights each month.

Construction logic (applied in order):
  1. Determine credit allocation based on credit signal
  2. Determine TIP allocation based on inflation signal
  3. Allocate remaining to duration bucket (TLT/IEF/SHY) based on curve signal
  4. Filter out any ETF with negative momentum
  5. Rescale survivors to sum to 1
  6. Blend signal weights with inverse-vol weights (ratio = config.SIGNAL_BLEND)

All tunable constants are read from `config` at call time so that
optimize.py can monkey-patch them between Optuna trials.
"""
import pandas as pd
import numpy as np
import config


# ---------------------------------------------------------------------------
# Sub-allocators
# ---------------------------------------------------------------------------

def _credit_frac(credit_z: float) -> float:
    frac = config.MAX_CREDIT_ALLOC * (0.5 + 0.5 * np.tanh(credit_z))
    return float(np.clip(frac, 0.0, config.MAX_CREDIT_ALLOC))


def _tip_frac(inflation_roc: float, rates_budget: float) -> float:
    frac = config.MAX_TIP_ALLOC * (0.5 + 0.5 * np.tanh(inflation_roc * 20))
    return float(np.clip(frac * rates_budget, 0.0, config.MAX_TIP_ALLOC))


def _duration_weights(curve_z: float) -> dict:
    scores = {
        "TLT":  1.5 * curve_z,
        "IEF":  0.0,
        "SHY": -1.0 * curve_z,
    }
    vals     = np.array(list(scores.values()))
    exp_vals = np.exp(vals - vals.max())
    probs    = exp_vals / exp_vals.sum()
    return dict(zip(scores.keys(), probs))


# ---------------------------------------------------------------------------
# Main weight builder
# ---------------------------------------------------------------------------

def build_weights(
    curve_z: float,
    credit_z: float,
    inflation_roc: float,
    mom: pd.Series,
    vol: pd.Series,
) -> pd.Series:
    w = pd.Series(0.0, index=config.ETF_UNIVERSE)

    # ── 1. Credit bucket ──────────────────────────────────────────────────
    c_frac       = _credit_frac(credit_z)
    rates_budget = 1.0 - c_frac
    w["LQD"]     = c_frac * config.CREDIT_LQD_SPLIT
    w["HYG"]     = c_frac * (1.0 - config.CREDIT_LQD_SPLIT)

    # ── 2. TIP allocation ─────────────────────────────────────────────────
    tip_frac   = _tip_frac(inflation_roc, rates_budget)
    w["TIP"]   = tip_frac
    pure_rates = rates_budget - tip_frac

    # ── 3. Duration bucket ────────────────────────────────────────────────
    for etf, frac in _duration_weights(curve_z).items():
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
    survivors   = w[w > 0].index
    inv_vol     = 1.0 / vol.reindex(survivors).clip(lower=0.001)
    iv_norm     = inv_vol / inv_vol.sum()
    signal_norm = w[survivors] / w[survivors].sum()

    blend   = config.SIGNAL_BLEND
    blended = blend * signal_norm + (1.0 - blend) * iv_norm
    w[survivors] = blended / blended.sum()

    return w


# ---------------------------------------------------------------------------
# Vectorised: build weights for every rebalance date
# ---------------------------------------------------------------------------

def build_weight_series(
    macro_monthly: pd.DataFrame,
    mom_monthly: pd.DataFrame,
    vol_monthly: pd.DataFrame,
) -> pd.DataFrame:
    records = []
    for date in macro_monthly.index:
        row = macro_monthly.loc[date]
        if row.isna().any() or mom_monthly.loc[date].isna().all():
            continue
        w = build_weights(
            curve_z       = row["yield_curve"],
            credit_z      = row["credit"],
            inflation_roc = row["inflation"],
            mom           = mom_monthly.loc[date],
            vol           = vol_monthly.loc[date],
        )
        w.name = date
        records.append(w)
    return pd.DataFrame(records)
