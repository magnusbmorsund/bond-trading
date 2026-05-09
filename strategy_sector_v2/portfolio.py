"""
V2 sector rotation portfolio builder.

Monthly logic:
  1. If SPY regime = 0  → 100% SHY
  2. Rank sectors by composite_score (multi-timescale blend)
  3. Filter: only keep ETFs with positive composite score
  4. Select top N_POSITIONS
  5. Weight by inverse realised-vol with MAX_WEIGHT cap
  6. Residual → SHY
"""
import logging
import pandas as pd
import numpy as np
import config_sector_v2 as config

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
    """Iteratively clip to MAX_WEIGHT and renormalise."""
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
    score_row: pd.Series,
    vol_row: pd.Series,
    regime: float,
) -> pd.Series:
    """
    Build ETF weights for a single rebalance date.

    Parameters
    ----------
    score_row : composite momentum scores per ETF
    vol_row   : realised vol per ETF
    regime    : 1.0 = risk-on, 0.0 = defensive
    """
    w = pd.Series(0.0, index=config.ETF_UNIVERSE)

    if regime < 0.5:
        w[config.CASH_ETF] = 1.0
        return w

    candidates = [e for e in config.ETF_UNIVERSE if e != config.CASH_ETF]
    scores = {}
    for e in candidates:
        if e not in score_row.index:
            continue
        s = score_row[e]
        if pd.isna(s):
            continue
        if config.ONLY_POSITIVE_COMPOSITE and s <= 0:
            continue
        scores[e] = s

    if not scores:
        w[config.CASH_ETF] = 1.0
        return w

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_n  = [e for e, _ in ranked[:config.N_POSITIONS]]

    raw_w = _inv_vol_weights(top_n, vol_row)
    if not raw_w:
        w[config.CASH_ETF] = 1.0
        return w

    capped = _apply_weight_caps(raw_w)
    if not capped:
        w[config.CASH_ETF] = 1.0
        return w

    for e, wt in capped.items():
        if e in w.index:
            w[e] = wt

    residual = 1.0 - w.sum()
    if residual > config.MIN_WEIGHT_THRESHOLD:
        w[config.CASH_ETF] = w.get(config.CASH_ETF, 0.0) + residual

    return w


def build_weight_series(
    score_monthly: pd.DataFrame,
    vol_monthly: pd.DataFrame,
    regime_monthly: pd.Series,
) -> pd.DataFrame:
    records = []
    for date in score_monthly.index:
        if score_monthly.loc[date].isna().all():
            continue
        regime = float(regime_monthly.loc[date]) if date in regime_monthly.index else 1.0
        w = build_weights(
            score_row = score_monthly.loc[date],
            vol_row   = vol_monthly.loc[date],
            regime    = regime,
        )
        w.name = date
        records.append(w)

    if not records:
        raise ValueError("build_weight_series produced no rows — check price history.")

    return pd.DataFrame(records).fillna(0.0)
