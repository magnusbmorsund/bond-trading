#!/usr/bin/env python
"""
Daily data pipeline — fetch latest prices and FRED from external sources,
upsert into Supabase.

Called by .github/workflows/pipeline.yml after market close (22:00 UTC).

Usage:
  python scripts/data_pipeline.py            # fetch last 7 days (default)
  python scripts/data_pipeline.py --days 30  # wider lookback
  python scripts/data_pipeline.py --days 0   # fetch from Supabase latest date

Required env vars:
  SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, FRED_API_KEY
"""
import argparse
import importlib
import logging
import sys
import time
import warnings

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
logger = logging.getLogger(__name__)


# ── Universe discovery ────────────────────────────────────────────────────────

def _all_price_tickers() -> list[str]:
    """Collect every unique ticker across all strategy configs."""
    tickers: set[str] = set()
    for mod_name in [
        "configs.sector_v1", "configs.sector_v2",  "configs.sector_v2b",
        "configs.sector_v2c", "configs.sector_v2d", "configs.sector_v2e",
    ]:
        cfg = importlib.import_module(mod_name)
        tickers.update(getattr(cfg, "ALL_TICKERS", []))
    for mod_name in ["configs.bond_v1", "configs.bond_v2", "configs.bond_v3"]:
        cfg = importlib.import_module(mod_name)
        tickers.update(getattr(cfg, "ETF_UNIVERSE", []))
    tickers.add("^VIX")
    return sorted(tickers)


def _all_fred_series() -> dict[str, str]:
    """Collect every unique FRED series across all bond strategy configs.
    Returns {label: series_id} merged dict."""
    series: dict[str, str] = {}
    for mod_name in ["configs.bond_v1", "configs.bond_v2", "configs.bond_v3"]:
        cfg = importlib.import_module(mod_name)
        series.update(getattr(cfg, "FRED_SERIES", {}))
    return series


# ── Prices ────────────────────────────────────────────────────────────────────

def _fetch_and_upsert_prices(tickers: list[str], start: str) -> None:
    logger.info("Fetching prices for %d tickers from %s → today", len(tickers), start)
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False, threads=True)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]] if "Close" in raw.columns else raw

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    prices = prices.dropna(how="all")
    logger.info("Downloaded %d rows × %d tickers", len(prices), len(prices.columns))

    import data.supabase_client as sb
    written = sb.upsert_prices(prices)
    logger.info("Upserted %d price rows to Supabase", written)


# ── FRED ──────────────────────────────────────────────────────────────────────

def _fetch_and_upsert_fred(fred_series: dict[str, str], start: str) -> None:
    import fredapi
    import data.supabase_client as sb
    from configs.bond_v1 import FRED_API_KEY

    fred = fredapi.Fred(api_key=FRED_API_KEY)
    total_written = 0

    for label, series_id in fred_series.items():
        try:
            data = fred.get_series(series_id, observation_start=start)
            data.name = series_id
            written = sb.upsert_fred(series_id, data)
            total_written += written
            logger.info("  FRED %-30s  %d obs upserted", series_id, written)
        except Exception as exc:
            logger.warning("  FRED %-30s  FAILED: %s", series_id, exc)
        time.sleep(0.2)  # respect FRED rate limits (120 req/min)

    logger.info("Upserted %d FRED rows to Supabase", total_written)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch latest market data and store in Supabase")
    parser.add_argument(
        "--days", type=int, default=7,
        help="Lookback days for price/FRED fetch (0 = auto-detect from Supabase latest date)",
    )
    args = parser.parse_args()

    import data.supabase_client as sb

    if not sb.is_configured():
        logger.error("SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set — aborting")
        sys.exit(1)

    # ── Determine fetch start date ──────────────────────────────────────────
    if args.days == 0:
        latest = sb.latest_price_date()
        if latest:
            # Start 3 days before the latest stored date for overlap safety
            price_start = (latest - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
            logger.info("Auto-detected latest Supabase date: %s → fetching from %s", latest.date(), price_start)
        else:
            price_start = "2000-01-01"
            logger.info("Supabase is empty — fetching full history from %s", price_start)
    else:
        price_start = (pd.Timestamp.today() - pd.Timedelta(days=args.days)).strftime("%Y-%m-%d")
        logger.info("Fetching last %d days from %s", args.days, price_start)

    tickers = _all_price_tickers()
    fred_map = _all_fred_series()

    logger.info("Universe: %d price tickers, %d FRED series", len(tickers), len(fred_map))

    # ── Prices ──────────────────────────────────────────────────────────────
    _fetch_and_upsert_prices(tickers, price_start)

    # ── FRED ────────────────────────────────────────────────────────────────
    _fetch_and_upsert_fred(fred_map, price_start)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
