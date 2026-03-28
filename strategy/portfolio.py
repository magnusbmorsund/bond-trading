"""
Convert composite signals into monthly target weights.

Four allocation buckets (always sum to 1.0):

  1. CREDIT  (LQD, HYG, ANGL, SJNK, BKLN, EMB, PFF)
     Size driven by credit_z + VIX hard-cap.
     Weighted by inverse-vol within bucket.

  2. HEDGE   (GLD)
     Activated when BOTH duration_z AND credit_z are simultaneously negative
     (stagflation / simultaneous rate-rise + spread widening).
     Subject to momentum filter — redirects to SHY if GLD is in downtrend.

  3. INFLATION (TIP)
     Size driven by inflation_z.

  4. DURATION (TLT, IEF, SHY)
     Remainder after credit + hedge + TIP.
     Allocation within bucket driven by duration_z softmax.
"""
import pandas as pd
import numpy as np
import config


# ---------------------------------------------------------------------------
# Sub-allocators
# ---------------------------------------------------------------------------

def _credit_frac(credit_z: float, vix_raw: float) -> float:
    soft = config.MAX_CREDIT_ALLOC * (0.5 + 0.5 * np.tanh(credit_z))
    frac = float(np.clip(soft, 0.0, config.MAX_CREDIT_ALLOC))
    if vix_raw > config.VIX_RISK_OFF:
        frac = min(frac, 0.10)
    elif vix_raw >= config.VIX_RISK_ON:
        t   = (vix_raw - config.VIX_RISK_ON) / (config.VIX_RISK_OFF - config.VIX_RISK_ON)
        cap = config.MAX_CREDIT_ALLOC * (1.0 - t) + 0.10 * t
        frac = min(frac, cap)
    return frac


def _alt_frac(duration_z: float, credit_z: float) -> float:
    """
    Gold hedge allocation.
    Activates proportionally when BOTH signals are negative simultaneously
    (the scenario where no bond ETF works — e.g. 2022 or 1994-style stagflation).
    Linear with the combined stress; zero when either signal is positive.
    """
    stress = max(0.0, -duration_z * 0.5 + -credit_z * 0.5)
    return float(np.clip(config.MAX_ALT_ALLOC * np.tanh(stress * 0.8), 0.0, config.MAX_ALT_ALLOC))


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

    # ── 1. Credit bucket ──────────────────────────────────────────────────
    c_frac = _credit_frac(credit_z, vix_raw)
    for etf, alloc in _credit_weights_inv_vol(c_frac, vol).items():
        w[etf] = alloc

    # ── 2. Gold hedge bucket ──────────────────────────────────────────────
    a_frac = _alt_frac(duration_z, credit_z)
    # If GLD has negative momentum, redirect its budget to SHY instead
    gld_mom_ok = "GLD" in mom.index and (pd.isna(mom["GLD"]) or mom["GLD"] >= 0)
    if a_frac > 0 and gld_mom_ok:
        w["GLD"] = a_frac
    else:
        a_frac = 0.0   # will go to duration bucket below

    # ── 3. TIP allocation ─────────────────────────────────────────────────
    rates_budget = 1.0 - c_frac - a_frac
    tip_frac     = _tip_frac(inflation_z, rates_budget)
    w["TIP"]     = tip_frac
    pure_rates   = rates_budget - tip_frac

    # ── 4. Duration bucket ────────────────────────────────────────────────
    for etf, frac in _duration_weights(duration_z).items():
        w[etf] = pure_rates * frac

    # ── 5. Momentum filter ────────────────────────────────────────────────
    for etf in config.ETF_UNIVERSE:
        if etf in mom.index and pd.notna(mom[etf]) and mom[etf] < 0:
            w[etf] = 0.0

    # ── 6. Fallback ───────────────────────────────────────────────────────
    if w.sum() < 1e-6:
        w["SHY"] = 1.0
        return w

    # ── 7. Blend signal weights with inverse-vol weights ──────────────────
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
