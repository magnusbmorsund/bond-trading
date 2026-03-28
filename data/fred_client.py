"""
Fetch and cache FRED macro signal data.
"""
import os
import pandas as pd
import fredapi

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import FRED_API_KEY, FRED_SERIES, DATA_DIR


def _cache_path(series_id: str) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, f"fred_{series_id}.csv")


def fetch_series(series_id: str, start: str = "2000-01-01", force: bool = False) -> pd.Series:
    """Return a FRED series as a daily pd.Series, using local cache when fresh."""
    path = _cache_path(series_id)

    if not force and os.path.exists(path):
        cached = pd.read_csv(path, index_col=0, parse_dates=True).squeeze()
        # Refresh if last observation is more than 2 days old
        if (pd.Timestamp.today() - cached.index[-1]).days <= 2:
            return cached

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
        s = fetch_series(series_id, start=start, force=force)
        frames[label] = s

    df = pd.DataFrame(frames)
    # Forward-fill weekends/holidays (FRED reports business days)
    df = df.ffill()
    return df
