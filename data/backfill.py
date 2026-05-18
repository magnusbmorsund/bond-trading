#!/usr/bin/env python
"""
One-time historical backfill — reads all local CSV cache files and upserts
their full history into Supabase.

Run once after creating the Supabase tables (db/schema.sql) and setting:
  export SUPABASE_URL=...
  export SUPABASE_SERVICE_ROLE_KEY=...

  python data/backfill.py

Safe to re-run — upsert is idempotent (ON CONFLICT DO UPDATE).
Typical runtime: 2-5 minutes for ~400K price rows + ~90K FRED rows.
"""
import os
import sys
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
logger = logging.getLogger(__name__)

# Resolve repo root regardless of where the script is invoked from
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE = os.path.join(_ROOT, "data", "cache")


def _check_env() -> None:
    missing = [v for v in ("SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY") if not os.environ.get(v)]
    if missing:
        logger.error("Missing env vars: %s", ", ".join(missing))
        sys.exit(1)


def backfill_prices() -> None:
    """Load all price CSV files and upsert into etf_prices."""
    import data.supabase_client as sb

    price_files = [
        f for f in os.listdir(_CACHE)
        if f.endswith(".csv") and "fred_" not in f and f != "vix.csv" and f != "vix3m.csv"
    ]

    # Merge all price CSVs into a single de-duplicated wide DataFrame
    frames: list[pd.DataFrame] = []
    for fname in sorted(price_files):
        path = os.path.join(_CACHE, fname)
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            frames.append(df)
            logger.info("  Loaded %-35s  %d rows × %d cols", fname, len(df), len(df.columns))
        except Exception as exc:
            logger.warning("  Skipping %s: %s", fname, exc)

    if not frames:
        logger.warning("No price CSV files found in %s", _CACHE)
        return

    # Outer-join all price frames so every ticker is represented once
    combined = frames[0]
    for df in frames[1:]:
        new_cols = [c for c in df.columns if c not in combined.columns]
        if new_cols:
            combined = combined.join(df[new_cols], how="outer")
        # Update existing columns only where combined has NaN
        for col in df.columns:
            if col in combined.columns:
                combined[col] = combined[col].combine_first(df[col])

    combined = combined.sort_index()
    logger.info("Combined price matrix: %d rows × %d tickers", len(combined), len(combined.columns))

    # Also load VIX
    for vix_file in ("vix.csv", "vix3m.csv"):
        vix_path = os.path.join(_CACHE, vix_file)
        if os.path.exists(vix_path):
            try:
                vix = pd.read_csv(vix_path, index_col=0, parse_dates=True).squeeze("columns")
                ticker = "^VIX" if vix_file == "vix.csv" else "^VIX3M"
                combined[ticker] = vix
                logger.info("  Loaded %-35s  %d rows", vix_file, len(vix))
            except Exception as exc:
                logger.warning("  Skipping %s: %s", vix_file, exc)

    logger.info("Upserting %d price rows to Supabase...", combined.notna().sum().sum())
    written = sb.upsert_prices(combined)
    logger.info("Price backfill complete — %d rows written", written)


def backfill_fred() -> None:
    """Load all fred_*.csv files and upsert into fred_series."""
    import data.supabase_client as sb

    fred_files = [f for f in os.listdir(_CACHE) if f.startswith("fred_") and f.endswith(".csv")]
    total = 0

    for fname in sorted(fred_files):
        series_id = fname[len("fred_"):-len(".csv")]  # e.g. "CPIAUCSL"
        path = os.path.join(_CACHE, fname)
        try:
            series = pd.read_csv(path, index_col=0, parse_dates=True).squeeze("columns")
            series.name = series_id
            written = sb.upsert_fred(series_id, series)
            total += written
            logger.info("  %-35s  %d rows written", series_id, written)
        except Exception as exc:
            logger.warning("  Skipping %s: %s", fname, exc)

    logger.info("FRED backfill complete — %d rows written", total)


def main() -> None:
    _check_env()

    import data.supabase_client as sb
    if not sb.is_configured():
        logger.error("Supabase not configured")
        sys.exit(1)

    logger.info("Starting backfill from local CSV cache: %s", _CACHE)

    logger.info("--- Prices ---")
    backfill_prices()

    logger.info("--- FRED ---")
    backfill_fred()

    logger.info("Backfill finished.")


if __name__ == "__main__":
    main()
