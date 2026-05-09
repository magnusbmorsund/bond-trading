"""
V2c sector rotation portfolio builder.

Key addition over V2b: cluster-capped top-N selection.
Within each correlated group (miners, green energy, biotech, semis, bonds,
international equity) at most CLUSTER_CAPS[group] positions are held, so
the portfolio cannot concentrate into 3-4 gold miners when they all trend.
"""
import logging
import pandas as pd
import numpy as np
import configs.sector_v2c as config

logger = logging.getLogger(__name__)


def _inv_vol_weights(etfs: list, vol_row: pd.Series) -> dict:
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


def _select_cluster_capped(
    scores: dict,
    n: int,
    corr_matrix: "pd.DataFrame | None" = None,
) -> list:
    """
    Select up to n ETFs by descending score, subject to:
      1. Cluster caps — at most CLUSTER_CAPS[cluster] picks per correlated group.
      2. Correlation filter — skip a candidate whose rolling correlation with any
         already-selected ETF exceeds config.CORR_THRESHOLD.
    """
    threshold = getattr(config, "CORR_THRESHOLD", 1.0)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    cluster_counts: dict[str, int] = {}
    selected = []
    for etf, _ in ranked:
        cluster = config.ETF_TO_CLUSTER.get(etf)
        if cluster is not None:
            cap = config.CLUSTER_CAPS.get(cluster, 999)
            if cluster_counts.get(cluster, 0) >= cap:
                continue

        if corr_matrix is not None and selected:
            try:
                corrs = [
                    abs(corr_matrix.loc[etf, held])
                    for held in selected
                    if held in corr_matrix.columns and etf in corr_matrix.index
                ]
                if corrs and max(corrs) > threshold:
                    continue
            except KeyError:
                pass

        if cluster is not None:
            cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
        selected.append(etf)
        if len(selected) >= n:
            break
    return selected


def build_weights(
    score_row: pd.Series,
    vol_row: pd.Series,
    regime: float,
    corr_matrix: "pd.DataFrame | None" = None,
) -> pd.Series:
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

    top_n = _select_cluster_capped(scores, config.N_POSITIONS, corr_matrix=corr_matrix)

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
    score_periodic: pd.DataFrame,
    vol_periodic: pd.DataFrame,
    regime_periodic: pd.Series,
    corr_periodic: "dict | None" = None,
) -> pd.DataFrame:
    records = []
    for date in score_periodic.index:
        if score_periodic.loc[date].isna().all():
            continue
        regime = float(regime_periodic.loc[date]) if date in regime_periodic.index else 1.0
        corr_matrix = corr_periodic.get(date) if corr_periodic is not None else None
        w = build_weights(
            score_row   = score_periodic.loc[date],
            vol_row     = vol_periodic.loc[date],
            regime      = regime,
            corr_matrix = corr_matrix,
        )
        w.name = date
        records.append(w)

    if not records:
        raise ValueError("build_weight_series produced no rows — check price history.")

    return pd.DataFrame(records).fillna(0.0)
