"""
Convert signals into target portfolio weights each month.

Construction logic (applied in order):
  1. Determine credit allocation based on credit signal
  2. Determine TIP allocation based on inflation signal
  3. Allocate remaining to duration bucket (TLT/IEF/SHY) based on curve signal
  4. Filter out any ETF with negative momentum
  5. Rescale survivors to sum to 1
  6. Blend 50/50 with inverse-volatility weights for risk balance
"""
import pandas as pd
import numpy as np

from config import MAX_CREDIT_ALLOC, MAX_TIP_ALLOC, ETF_UNIVERSE


# ---------------------------------------------------------------------------
# Sub-allocators
# ---------------------------------------------------------------------------

def _credit_frac(credit_z: float) -> float:
    """
    Credit allocation as fraction of portfolio.
    credit_z: z-score (positive = tight spreads = risk-on)
    Range: 0% (risk-off) → MAX_CREDIT_ALLOC (full risk-on)
    """
    frac = MAX_CREDIT_ALLOC * (0.5 + 0.5 * np.tanh(credit_z))
    return float(np.clip(frac, 0.0, MAX_CREDIT_ALLOC))


def _tip_frac(inflation_roc: float, rates_budget: float) -> float:
    """
    TIP allocation within the rates bucket.
    Scales from 0 (deflation) to MAX_TIP_ALLOC (high inflation).
    """
    frac = MAX_TIP_ALLOC * (0.5 + 0.5 * np.tanh(inflation_roc * 20))
    return float(np.clip(frac * rates_budget, 0.0, MAX_TIP_ALLOC))


def _duration_weights(curve_z: float) -> dict:
    """
    Allocate within pure-rates bucket (TLT / IEF / SHY).
    Steep curve → TLT heavy.  Flat/inverted → SHY heavy.
    """
    # Smooth continuous mapping via softmax-style blend
    scores = {
        "TLT": 1.5 * curve_z,     # benefits most from steep curve
        "IEF": 0.0,                # neutral / intermediate
        "SHY": -1.0 * curve_z,    # benefits from flat/inverted curve
    }
    # Softmax to get probabilities
    vals = np.array(list(scores.values()))
    exp_vals = np.exp(vals - vals.max())
    probs = exp_vals / exp_vals.sum()
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
    """
    Build one row of target weights for the ETF universe.

    Parameters
    ----------
    curve_z        : yield curve z-score (from signals.yield_curve_signal)
    credit_z       : credit z-score     (from signals.credit_signal)
    inflation_roc  : breakeven 3m RoC   (from signals.inflation_signal)
    mom            : momentum per ETF   (positive = include)
    vol            : annualised vol per ETF
    """
    w = pd.Series(0.0, index=ETF_UNIVERSE)

    # ── 1. Credit bucket ───────────────────────────────────────────────────
    c_frac = _credit_frac(credit_z)
    rates_budget = 1.0 - c_frac

    w["LQD"] = c_frac * 0.55   # IG gets larger share of credit budget
    w["HYG"] = c_frac * 0.45

    # ── 2. Inflation (TIP) ─────────────────────────────────────────────────
    tip_frac = _tip_frac(inflation_roc, rates_budget)
    w["TIP"] = tip_frac
    pure_rates = rates_budget - tip_frac

    # ── 3. Duration bucket ─────────────────────────────────────────────────
    dur = _duration_weights(curve_z)
    for etf, frac in dur.items():
        w[etf] = pure_rates * frac

    # ── 4. Momentum filter — zero out negative momentum ETFs ───────────────
    for etf in ETF_UNIVERSE:
        if etf in mom.index and pd.notna(mom[etf]) and mom[etf] < 0:
            w[etf] = 0.0

    # ── 5. Fallback: if everything filtered, hold SHY ──────────────────────
    if w.sum() < 1e-6:
        w["SHY"] = 1.0
        return w

    # ── 6. Blend signal weights (50%) with inverse-vol weights (50%) ───────
    survivors = w[w > 0].index
    inv_vol = 1.0 / vol.reindex(survivors).clip(lower=0.001)
    iv_norm  = inv_vol / inv_vol.sum()               # inverse-vol proportions

    signal_norm = w[survivors] / w[survivors].sum()  # signal proportions

    blended = 0.5 * signal_norm + 0.5 * iv_norm
    w[survivors] = blended / blended.sum()           # final normalise

    return w


# ---------------------------------------------------------------------------
# Vectorised: build weights for every rebalance date
# ---------------------------------------------------------------------------

def build_weight_series(
    macro_monthly: pd.DataFrame,
    mom_monthly: pd.DataFrame,
    vol_monthly: pd.DataFrame,
) -> pd.DataFrame:
    """
    Iterate over each month-end rebalance date and return a DataFrame
    of target weights with shape (dates, ETFs).
    """
    records = []

    for date in macro_monthly.index:
        row = macro_monthly.loc[date]

        # Guard against NaN signals early in the history
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
