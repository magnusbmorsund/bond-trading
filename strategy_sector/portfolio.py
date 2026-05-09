"""
Sector rotation portfolio builder.

Monthly logic (one row per rebalance date):
  1. If SPY regime = 0 (below 200d MA)  → 100% SHY
  2. Rank all sectors by momentum score
  3. Select top N with positive absolute momentum
  4. Weight by inverse realised-vol, apply min/max caps
  5. Residual (if any) → SHY

All parameters read from config_sector at call time so optimize.py
can patch them via setattr().
"""
import logging
import pandas as pd
import numpy as np
import config_sector as config

logger = logging.getLogger(__name__)


def _inv_vol_weights(etfs: list, vol_row: pd.Series) -> dict:
    """Inverse-vol weights for a list of ETFs, normalised to sum to 1."""
    inv = {}
    for e in etfs:
        v = vol_row.get(e, np.nan)
        if pd.notna(v) and v > 0:
            inv[e] = 1.0 / max(v, config.MIN_VOL_CLIP)
    if not inv:
        return {}
    total = sum(inv.values())
    return {e: iv / total for e, iv in inv.items()}


def _apply_weight_caps(weights: dict) -> dict:
    """Iteratively clip to [MIN_WEIGHT, MAX_WEIGHT] and renormalise."""
    if not weights:
        return {}
    w = dict(weights)
    for _ in range(20):
        total = sum(w.values())
        if total < config.MIN_WEIGHT_THRESHOLD:
            return {}
        w = {e: v / total for e, v in w.items()}
        capped = False
        for e in list(w):
            if w[e] > config.MAX_WEIGHT:
                w[e] = config.MAX_WEIGHT
                capped = True
        if not capped:
            break
    total = sum(w.values())
    return {e: v / total for e, v in w.items()} if total > 0 else {}


def build_weights(
    mom_row: pd.Series,
    vol_row: pd.Series,
    regime: float,
) -> pd.Series:
    """
    Build sector weights for a single rebalance date.

    Parameters
    ----------
    mom_row : momentum scores for each ETF on this date
    vol_row : realised vol for each ETF on this date
    regime  : 1.0 = risk-on, 0.0 = defensive (SPY below MA)
    """
    w = pd.Series(0.0, index=config.ETF_UNIVERSE)

    # Defensive: park everything in cash
    if regime < 0.5:
        w[config.CASH_ETF] = 1.0
        return w

    # Rank sectors by momentum
    candidates = config.SECTOR_CORE + config.SECTOR_SUB
    scores = {}
    for e in candidates:
        if e not in mom_row.index:
            continue
        m = mom_row[e]
        if pd.isna(m):
            continue
        if config.ONLY_POSITIVE_MOM and m <= 0:
            continue   # absolute momentum filter: skip sectors in downtrend
        scores[e] = m

    if not scores:
        w[config.CASH_ETF] = 1.0
        return w

    # Top N by momentum score
    ranked  = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_n   = [e for e, _ in ranked[:config.N_POSITIONS]]

    # Inverse-vol weighting
    raw_w = _inv_vol_weights(top_n, vol_row)
    if not raw_w:
        w[config.CASH_ETF] = 1.0
        return w

    # Apply weight caps
    capped = _apply_weight_caps(raw_w)
    if not capped:
        w[config.CASH_ETF] = 1.0
        return w

    for e, wt in capped.items():
        if e in w.index:
            w[e] = wt

    # Residual → SHY (can happen when max-cap binds and redistribution leaves slack)
    residual = 1.0 - w.sum()
    if residual > config.MIN_WEIGHT_THRESHOLD:
        w[config.CASH_ETF] = w.get(config.CASH_ETF, 0.0) + residual

    return w


def build_weight_series(
    mom_monthly: pd.DataFrame,
    vol_monthly: pd.DataFrame,
    regime_monthly: pd.Series,
) -> pd.DataFrame:
    records = []
    for date in mom_monthly.index:
        if mom_monthly.loc[date].isna().all():
            continue
        regime = float(regime_monthly.loc[date]) if date in regime_monthly.index else 1.0
        w = build_weights(
            mom_row = mom_monthly.loc[date],
            vol_row = vol_monthly.loc[date],
            regime  = regime,
        )
        w.name = date
        records.append(w)

    if not records:
        raise ValueError("build_weight_series produced no rows — check price history.")

    return pd.DataFrame(records).fillna(0.0)
