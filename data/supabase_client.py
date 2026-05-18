"""
Supabase read/write client for ETF prices and FRED macro data.

Tables (see db/schema.sql):
  etf_prices  (date, ticker, close)
  fred_series (date, series_id, value)

Environment variables required:
  SUPABASE_URL               — project URL  (e.g. https://xyz.supabase.co)
  SUPABASE_SERVICE_ROLE_KEY  — service-role key (bypasses RLS; never commit this)
"""
import os
import logging
import math
import pandas as pd

logger = logging.getLogger(__name__)

_UPSERT_BATCH = 500   # rows per Supabase upsert call
_SELECT_LIMIT = 5_000_000  # generous upper bound for full-history reads


def is_configured() -> bool:
    """True when both Supabase env vars are present."""
    return bool(
        os.environ.get("SUPABASE_URL")
        and os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    )


def _client():
    from supabase import create_client
    return create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_ROLE_KEY"],
    )


# ── Price data ────────────────────────────────────────────────────────────────

def fetch_prices(tickers: list[str], start: str = "2000-01-01") -> pd.DataFrame:
    """
    Fetch adjusted close prices from Supabase.
    Returns a wide DataFrame (DatetimeIndex × ticker columns), sorted by date.
    """
    sb = _client()
    result = (
        sb.table("etf_prices")
        .select("date,ticker,close")
        .in_("ticker", tickers)
        .gte("date", start)
        .limit(_SELECT_LIMIT)
        .execute()
    )
    if not result.data:
        raise ValueError(
            f"No price data in Supabase for tickers={tickers[:5]}... start={start}"
        )

    df = pd.DataFrame(result.data)
    df["date"] = pd.to_datetime(df["date"])
    wide = df.pivot(index="date", columns="ticker", values="close")
    wide.index.name = "Date"
    wide.columns.name = None
    return wide.sort_index()


def upsert_prices(df: pd.DataFrame) -> int:
    """
    Upsert a wide price DataFrame (DatetimeIndex × ticker columns) into etf_prices.
    Skips NaN cells. Returns total rows written.
    """
    sb = _client()
    rows = []
    for date, row in df.iterrows():
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
        for ticker, close in row.items():
            if pd.isna(close):
                continue
            rows.append({"date": date_str, "ticker": str(ticker), "close": float(close)})

    return _batch_upsert(sb, "etf_prices", rows, conflict="date,ticker")


def latest_price_date() -> pd.Timestamp | None:
    """Return the most recent date stored in etf_prices, or None if the table is empty."""
    sb = _client()
    result = (
        sb.table("etf_prices")
        .select("date")
        .order("date", desc=True)
        .limit(1)
        .execute()
    )
    if result.data:
        return pd.Timestamp(result.data[0]["date"])
    return None


# ── FRED macro data ───────────────────────────────────────────────────────────

def fetch_fred(series_ids: list[str], start: str = "2000-01-01") -> pd.DataFrame:
    """
    Fetch FRED series from Supabase.
    Returns a wide DataFrame (DatetimeIndex × series_id columns), sorted by date.
    """
    sb = _client()
    result = (
        sb.table("fred_series")
        .select("date,series_id,value")
        .in_("series_id", series_ids)
        .gte("date", start)
        .limit(_SELECT_LIMIT)
        .execute()
    )
    if not result.data:
        raise ValueError(
            f"No FRED data in Supabase for series={series_ids[:5]}... start={start}"
        )

    df = pd.DataFrame(result.data)
    df["date"] = pd.to_datetime(df["date"])
    wide = df.pivot(index="date", columns="series_id", values="value")
    wide.index.name = None
    wide.columns.name = None
    return wide.sort_index()


def upsert_fred(series_id: str, series: pd.Series) -> int:
    """
    Upsert one FRED series (pd.Series with DatetimeIndex) into fred_series.
    Returns rows written.
    """
    sb = _client()
    rows = []
    for date, value in series.items():
        if pd.isna(value):
            continue
        rows.append({
            "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
            "series_id": series_id,
            "value": float(value),
        })
    return _batch_upsert(sb, "fred_series", rows, conflict="date,series_id")


def latest_fred_date(series_id: str) -> pd.Timestamp | None:
    """Return the most recent date for a specific FRED series, or None."""
    sb = _client()
    result = (
        sb.table("fred_series")
        .select("date")
        .eq("series_id", series_id)
        .order("date", desc=True)
        .limit(1)
        .execute()
    )
    if result.data:
        return pd.Timestamp(result.data[0]["date"])
    return None


# ── Internal helpers ──────────────────────────────────────────────────────────

def _batch_upsert(sb, table: str, rows: list[dict], conflict: str) -> int:
    """Upsert rows in batches of _UPSERT_BATCH. Returns total rows written."""
    if not rows:
        return 0
    total = 0
    n_batches = math.ceil(len(rows) / _UPSERT_BATCH)
    for i in range(n_batches):
        batch = rows[i * _UPSERT_BATCH: (i + 1) * _UPSERT_BATCH]
        sb.table(table).upsert(batch, on_conflict=conflict).execute()
        total += len(batch)
    return total
