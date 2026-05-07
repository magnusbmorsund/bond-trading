"""
v2b — extends v2 with a short-lookback momentum gate on long-duration ETFs.

Motivation: the standard 12-1 month momentum filter reacts too slowly to
sudden rate-hike regimes (e.g. 2022). TLT/IEF were above their year-ago
price for months into the sell-off. A 3-month gate catches the trend faster
without disturbing the rest of the portfolio logic.

Everything else is identical to v2.
"""
from config_v2 import *  # noqa: F401, F403 — inherit all v2 parameters

# Short-window duration gate — applied to long-duration ETFs only
DURATION_MOM_LOOKBACK  = 63   # 3-month price return
DURATION_MOM_SKIP      = 0    # no skip (unlike the 12-1 month's 21-day skip)
DURATION_MOM_GATE_ETFS = ["TLT", "IEF"]   # SHY excluded — it is the cash bucket
