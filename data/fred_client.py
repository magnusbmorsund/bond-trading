"""
Fetch and cache FRED macro signal data.
"""
import os
import pandas as pd
import fredapi

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import FRED_API_KEY, FRED_SERIES, DATA_DIR

# Monthly series are only published once a month — treat as fresh for 35 days
_MONTHLY_SERIES = {"CPIAUCSL", "FEDFUNDS", "UNRATE", "INDPRO"}
_WEEKLY_SERIES  = {"WALCL", "TEDRATE"}   # Fed balance sheet + TED spread weekly
_DAILY_STALE_DAYS   = 2
_MONTHLY_STALE_DAYS = 35


def _cache_path(series_id: str) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, f"fred_{series_id}.csv")


def _is_fresh(series_id: str, cached: pd.Series) -> bool:
    age = (pd.Timestamp.today() - cached.index[-1]).days
    if series_id in _MONTHLY_SERIES:
        limit = _MONTHLY_STALE_DAYS
    elif series_id in _WEEKLY_SERIES:
        limit = 10   # weekly data: stale after 10 days
    else:
        limit = _DAILY_STALE_DAYS
    return age <= limit


def fetch_series(series_id: str, start: str = "2000-01-01", force: bool = False) -> pd.Series:
    """Return a FRED series as a daily pd.Series, using local cache when fresh.
    Falls back to stale cache if FRED is unreachable."""
    path = _cache_path(series_id)

    if os.path.exists(path):
        cached = pd.read_csv(path, index_col=0, parse_dates=True).squeeze()
        if not force and _is_fresh(series_id, cached):
            return cached
        # Stale cache exists — try to refresh, fall back if FRED errors
        try:
            fred = fredapi.Fred(api_key=FRED_API_KEY)
            data = fred.get_series(series_id, observation_start=start)
            data.name = series_id
            data.to_csv(path, header=True)
            print(f"  Fetched FRED:{series_id} ({len(data)} obs)")
            return data
        except Exception as e:
            print(f"  FRED:{series_id} unavailable ({e}), using cached data")
            return cached

    # No cache — must fetch
    fred = fredapi.Fred(api_key=FRED_API_KEY)
    data = fred.get_series(series_id, observation_start=start)
    data.name = series_id
    data.to_csv(path, header=True)
    print(f"  Fetched FRED:{series_id} ({len(data)} obs)")
    return data


def fetch_all(start: str = "2000-01-01", force: bool = False) -> pd.DataFrame:
    """Fetch all configured FRED series and return as a single aligned DataFrame."""
    frames = {}
    for label, series_id in FRED_SERIES.items():
        try:
            s = fetch_series(series_id, start=start, force=force)
            frames[label] = s
        except Exception as e:
            print(f"  Skipping FRED:{series_id} ({e})")
            continue

    df = pd.DataFrame(frames)
    df = df.ffill()
    return df
