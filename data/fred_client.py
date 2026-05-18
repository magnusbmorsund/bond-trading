"""
Fetch and cache FRED macro signal data.
"""
import os
import time
import logging
import pandas as pd
import fredapi

from configs.bond_v1 import FRED_API_KEY, FRED_SERIES, DATA_DIR

logger = logging.getLogger(__name__)

_DAILY_STALE_DAYS   = 2
_MONTHLY_STALE_DAYS = 35
_WEEKLY_STALE_DAYS  = 10
_RETRY_ATTEMPTS     = 3
_RETRY_DELAY_S      = 2


def _cache_path(series_id: str) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, f"fred_{series_id}.csv")


def _stale_limit(cached: pd.Series) -> int:
    """
    Infer cache TTL from the actual observation cadence of the cached series.
    Uses the median gap between the last 10 observations to detect monthly/weekly/daily.
    This auto-handles any FRED series regardless of frequency.
    """
    recent = cached.dropna()
    if len(recent) >= 2:
        gaps = recent.index.to_series().diff().dropna().dt.days
        median_gap = float(gaps.tail(10).median())
        if median_gap >= 20:
            return _MONTHLY_STALE_DAYS
        if median_gap >= 5:
            return _WEEKLY_STALE_DAYS
    return _DAILY_STALE_DAYS


def _is_fresh(series_id: str, cached: pd.Series) -> bool:
    age = (pd.Timestamp.today() - cached.index[-1]).days
    return age <= _stale_limit(cached)


def _fetch_from_fred(series_id: str, start: str) -> pd.Series:
    """Fetch one series from FRED with retry on transient failures."""
    last_exc = None
    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            fred = fredapi.Fred(api_key=FRED_API_KEY)
            data = fred.get_series(series_id, observation_start=start)
            data.name = series_id
            return data
        except Exception as exc:
            last_exc = exc
            if attempt < _RETRY_ATTEMPTS:
                logger.debug(
                    "FRED fetch attempt %d/%d failed for %s: %s — retrying in %ds",
                    attempt, _RETRY_ATTEMPTS, series_id, exc, _RETRY_DELAY_S,
                )
                time.sleep(_RETRY_DELAY_S)
    raise last_exc


def fetch_series(series_id: str, start: str = "2000-01-01", force: bool = False) -> pd.Series:
    """Return a FRED series as a pd.Series. Priority: local CSV cache → Supabase → FRED API.
    Falls back to stale cache if all remote sources are unreachable."""
    path = _cache_path(series_id)

    if os.path.exists(path):
        cached = pd.read_csv(path, index_col=0, parse_dates=True).squeeze("columns")
        if not force and _is_fresh(series_id, cached):
            return cached

        age_days = (pd.Timestamp.today() - cached.index[-1]).days

        # Try Supabase first when configured
        from data.supabase_client import is_configured, fetch_fred as sb_fetch_fred
        if is_configured():
            try:
                df = sb_fetch_fred([series_id], start)
                if series_id in df.columns:
                    data = df[series_id].dropna()
                    data.name = series_id
                    # Preserve any historical rows Supabase may not have
                    if len(cached) > 0 and cached.index[0] < data.index[0]:
                        pre = cached[cached.index < data.index[0]]
                        data = pd.concat([pre, data]).sort_index()
                    data.to_csv(path, header=True)
                    logger.info("Fetched FRED:%s from Supabase (%d obs)", series_id, len(data))
                    return data
            except Exception as exc:
                logger.warning("Supabase FRED fetch failed for %s, trying FRED API: %s", series_id, exc)

        # Fall back to FRED API
        try:
            data = _fetch_from_fred(series_id, start)
            if len(data) < len(cached) and len(data) > 0 and data.index[0] > cached.index[0]:
                pre = cached[cached.index < data.index[0]]
                data = pd.concat([pre, data]).sort_index()
                logger.info(
                    "FRED:%s truncated upstream — preserved %d historical rows, combined=%d",
                    series_id, len(pre), len(data),
                )
            data.to_csv(path, header=True)
            logger.info("Fetched FRED:%s from API (%d obs)", series_id, len(data))
            return data
        except Exception as exc:
            logger.warning(
                "FRED:%s unavailable after %d attempts (cache is %d days old): %s",
                series_id, _RETRY_ATTEMPTS, age_days, exc,
            )
            return cached

    # No cache — try Supabase, then FRED API; let exception propagate on total failure
    from data.supabase_client import is_configured, fetch_fred as sb_fetch_fred
    if is_configured():
        try:
            df = sb_fetch_fred([series_id], start)
            if series_id in df.columns:
                data = df[series_id].dropna()
                data.name = series_id
                data.to_csv(path, header=True)
                logger.info("Fetched FRED:%s from Supabase (%d obs)", series_id, len(data))
                return data
        except Exception as exc:
            logger.warning("Supabase FRED fetch failed for %s, trying FRED API: %s", series_id, exc)

    data = _fetch_from_fred(series_id, start)
    data.to_csv(path, header=True)
    logger.info("Fetched FRED:%s from API (%d obs)", series_id, len(data))
    return data


def fetch_all(start: str = "2000-01-01", force: bool = False) -> pd.DataFrame:
    """Fetch all configured FRED series and return as a single aligned DataFrame."""
    frames = {}
    for label, series_id in FRED_SERIES.items():
        try:
            s = fetch_series(series_id, start=start, force=force)
            frames[label] = s
        except Exception as exc:
            logger.error("Skipping FRED:%s — no cache and fetch failed: %s", series_id, exc)

    if not frames:
        raise RuntimeError("All FRED series failed and no cache exists. Check FRED_API_KEY.")

    missing = set(FRED_SERIES.keys()) - set(frames.keys())
    if missing:
        logger.warning("Missing FRED series (will be absent from macro DataFrame): %s", sorted(missing))

    df = pd.DataFrame(frames)
    df = df.ffill()
    return df
