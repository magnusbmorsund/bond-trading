"""
Supabase write-client for the bond rotation strategy.

Writes two tables:
  weight_snapshots — one row per ETF per `weights` run (audit trail)
  data_freshness   — one row per data source (upserted on every load)

Credentials are read from the SUPABASE_URL / SUPABASE_KEY env vars,
or fall back to the magnus-trading .env file if present.
The module degrades gracefully: if Supabase is unreachable, a WARNING is
logged and execution continues — it never blocks the trading workflow.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import date, timezone, datetime

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy client — only instantiated on first use
# ---------------------------------------------------------------------------

_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    # Fall back to magnus-trading .env
    if not url or not key:
        env_path = os.path.expanduser("~/Desktop/magnus-trading/.env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("SUPABASE_URL="):
                        url = line.split("=", 1)[1]
                    elif line.startswith("SUPABASE_KEY="):
                        key = line.split("=", 1)[1]

    if not url or not key:
        raise EnvironmentError(
            "Supabase credentials not found. "
            "Set SUPABASE_URL and SUPABASE_KEY env vars, "
            "or add them to ~/Desktop/magnus-trading/.env"
        )

    from supabase import create_client
    _client = create_client(url, key)
    return _client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _params_hash() -> str | None:
    """SHA-256 of best_params.json so we can correlate runs to param sets."""
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "best_params.json")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_weight_snapshot(
    signal_weights: pd.Series,
    effective_weights: pd.Series,
    as_of_date: date,
) -> None:
    """
    Insert one row per ETF into weight_snapshots.

    Parameters
    ----------
    signal_weights    : Series[etf -> float]  model weights before trailing stop
    effective_weights : Series[etf -> float]  weights after trailing stop
    as_of_date        : the rebalance date the weights come from
    """
    try:
        client     = _get_client()
        phash      = _params_hash()
        run_at     = datetime.now(timezone.utc).isoformat()
        as_of_str  = as_of_date.isoformat() if hasattr(as_of_date, "isoformat") else str(as_of_date)

        rows = []
        for etf in signal_weights.index:
            sw = float(signal_weights.get(etf, 0.0))
            ew = float(effective_weights.get(etf, 0.0))
            if sw < 1e-6 and ew < 1e-6:
                continue   # skip zero-weight ETFs to keep the table lean
            rows.append({
                "run_at":              run_at,
                "as_of_date":          as_of_str,
                "etf":                 etf,
                "signal_weight":       round(sw, 6),
                "effective_weight":    round(ew, 6),
                "trailing_stop_fired": ew < sw * 0.5,
                "params_hash":         phash,
            })

        if not rows:
            logger.warning("write_weight_snapshot: no non-zero weights to write")
            return

        client.table("weight_snapshots").insert(rows).execute()
        logger.info(
            "Supabase: wrote %d weight_snapshots rows (as_of=%s)", len(rows), as_of_str
        )

    except Exception as exc:
        logger.warning("Supabase write_weight_snapshot failed (non-fatal): %s", exc)


def write_data_freshness(macro: pd.DataFrame, prices: pd.DataFrame, vix: pd.Series | None = None) -> None:
    """
    Upsert one row per data source into data_freshness.

    Call after load_all() so the table always reflects the current cache state.
    """
    try:
        client  = _get_client()
        today   = pd.Timestamp.today().normalize()
        updated = datetime.now(timezone.utc).isoformat()

        rows = []

        def _row(source: str, last_date: pd.Timestamp) -> dict:
            age = int((today - last_date.normalize()).days)
            return {
                "source":         source,
                "last_date":      last_date.date().isoformat(),
                "cache_age_days": age,
                "updated_at":     updated,
            }

        # ETF prices
        if prices is not None and len(prices):
            rows.append(_row("prices", prices.index[-1]))

        # VIX (stored in macro["vix_raw"] after pipeline merges it)
        if "vix" in macro.columns:
            vix_last = macro["vix"].dropna()
            if len(vix_last):
                rows.append(_row("vix", vix_last.index[-1]))

        # Per-FRED series — use the last non-NaN date for each column
        fred_cols = [c for c in macro.columns if c not in ("vix", "vix_raw")]
        for col in fred_cols:
            series = macro[col].dropna()
            if len(series):
                rows.append(_row(f"FRED:{col}", series.index[-1]))

        if not rows:
            return

        # Upsert on primary key (source)
        client.table("data_freshness").upsert(rows, on_conflict="source").execute()
        logger.info("Supabase: upserted %d data_freshness rows", len(rows))

    except Exception as exc:
        logger.warning("Supabase write_data_freshness failed (non-fatal): %s", exc)
