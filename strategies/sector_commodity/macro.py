"""Commodity macro exposure overlay.

Builds a daily exposure multiplier in [MACRO_FLOOR, MACRO_CEIL] from a small,
economically-motivated FRED z-score composite. The multiplier is applied as the
outermost exposure scaler on strategy returns (see sector_shared.backtest.run).

Pipeline (all look-ahead-safe):
  1. Fetch broad USD (DTWEXBGS), 10y real yield (DFII10), CPI (CPIAUCSL) and
     industrial production (INDPRO) from the FRED client (CSV cache → Supabase →
     FRED API; falls back to stale cache offline).
  2. Resample each to month-end and z-score over MACRO_Z_LOOKBACK_M months.
  3. Combine into macro_z (positive ⇒ commodity-friendly).
  4. Map to an exposure multiplier, shift by MACRO_PUB_LAG_M months (publication
     lag), and forward-fill onto the daily calendar — so the multiplier on day T
     uses only macro data released before T.

Economic thesis (CONFIRMATION, not prediction):
  + falling / weak broad USD  → tailwind for $-priced commodities
  + falling 10y real yield    → tailwind (lower carry cost; gold)
  + rising CPI / industrial   → inflation / demand tailwind
With MACRO_CEIL = 1.0 the overlay can only REDUCE exposure when the backdrop is
hostile — it never levers up, so it cannot inflate returns, only protect against
macro headwinds. MACRO_FLOOR = 1.0 would disable the overlay entirely.
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _z(s: pd.Series, lookback: int) -> pd.Series:
    minp = max(12, lookback // 3)
    mu = s.rolling(lookback, min_periods=minp).mean()
    sd = s.rolling(lookback, min_periods=minp).std()
    return (s - mu) / sd.clip(lower=1e-6)


def _monthly(series: pd.Series) -> pd.Series:
    return series.dropna().resample("ME").last()


# The raw FRED series are deterministic given `start`, but the optimiser calls
# compute_exposure once per trial (only the z-score *weights* change). Cache the
# fetched monthly series so a 250-trial study fetches FRED once, not 250×.
_RAW_CACHE: dict = {}


def _load_monthly_series(start: str):
    if start in _RAW_CACHE:
        return _RAW_CACHE[start]
    from data.fred_client import fetch_series
    series = (
        _monthly(fetch_series("DTWEXBGS", start)),   # broad USD index
        _monthly(fetch_series("DFII10",   start)),   # 10y TIPS real yield (%)
        _monthly(fetch_series("CPIAUCSL", start)),   # CPI index
        _monthly(fetch_series("INDPRO",   start)),   # industrial production
    )
    _RAW_CACHE[start] = series
    return series


def compute_exposure(config, daily_index: pd.DatetimeIndex) -> pd.Series:
    """Return a daily exposure multiplier aligned to `daily_index`."""
    full = pd.Series(float(getattr(config, "MACRO_CEIL", 1.0)),
                     index=daily_index, name="macro_exposure")
    if not getattr(config, "MACRO_ENABLED", True):
        return full

    start = config.BACKTEST_START
    try:
        usd, real, cpi, indpro = _load_monthly_series(start)
    except Exception as exc:  # offline + no cache → full exposure (overlay off)
        logger.warning("Macro overlay disabled (FRED fetch failed: %s) — full exposure", exc)
        return full

    lb = config.MACRO_Z_LOOKBACK_M
    usd_sig  = -_z(usd.pct_change(config.MACRO_USD_MOM_M),  lb)   # rising USD = headwind
    real_sig = -_z(real.diff(config.MACRO_REAL_MOM_M),      lb)   # rising real yield = headwind
    h        = config.MACRO_INFL_HORIZON_M
    infl_sig = 0.5 * _z(cpi.pct_change(h), lb) + 0.5 * _z(indpro.pct_change(h), lb)

    idx = usd_sig.index.union(real_sig.index).union(infl_sig.index)
    usd_sig  = usd_sig.reindex(idx).ffill()
    real_sig = real_sig.reindex(idx).ffill()
    infl_sig = infl_sig.reindex(idx).ffill()

    wsum = config.MACRO_W_USD + config.MACRO_W_REAL + config.MACRO_W_INFL
    wsum = wsum if wsum > 0 else 1.0
    macro_z = (
        config.MACRO_W_USD  * usd_sig
        + config.MACRO_W_REAL * real_sig
        + config.MACRO_W_INFL * infl_sig
    ) / wsum

    expo = (1.0 + config.MACRO_GAIN * macro_z).clip(
        lower=config.MACRO_FLOOR, upper=config.MACRO_CEIL
    )
    # Publication lag, then project month-end → daily by forward-fill (no look-ahead).
    expo = expo.shift(config.MACRO_PUB_LAG_M)
    expo_daily = (
        expo.reindex(expo.index.union(daily_index)).ffill().reindex(daily_index)
    )
    # Before the z-score warmup completes → no signal → stay fully invested.
    expo_daily = expo_daily.fillna(float(config.MACRO_CEIL))
    expo_daily.name = "macro_exposure"
    return expo_daily
