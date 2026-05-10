import os
from configs.sector_base import *  # noqa: F401,F403

# ── V2e: V2d + supercycle lookbacks (24m, 36m momentum components) ───────────
# Same liquid universe as V2d (all ETFs ≥$100M avg daily dollar vol).
# Key addition: W_SLOW_24M and W_SLOW_36M extend the slow momentum blend
# so persistent multi-year cycles (commodity supercycles, tech booms, rate
# cycles) contribute to the ranking signal alongside 12m/18m momentum.

# V2b base sectors
SECTOR_CORE   = ["XLE", "XLK", "XLV", "XLF", "XLI", "XLY", "XLP", "XLU", "XLB", "VNQ"]

# Compute: SMH + ARKK + IGV (same as V2d)
COMPUTE_ETFS  = ["SMH", "ARKK", "IGV"]

METALS_ETFS   = ["GDX", "GDXJ", "SIL", "XME", "COPX", "REMX"]
ENERGY_ETFS   = ["XOP", "OIH"]
GREEN_ETFS    = ["ICLN", "URA"]
DEFENSE_ETFS  = ["ITA"]
GOLD_ETFS     = ["GLD"]
BIOTECH_ETFS  = ["XBI", "IBB"]
CHINA_ETFS    = ["KWEB"]

# Cross-asset diversifiers (same as V2d)
BOND_ETFS      = ["TLT", "IEF", "HYG"]
INTL_ETFS      = ["EFA", "EEM", "EWJ", "EWZ", "INDA"]
COMMODITY_ETFS = ["PDBC"]

ETF_UNIVERSE = (
    SECTOR_CORE + COMPUTE_ETFS +
    METALS_ETFS + ENERGY_ETFS + GREEN_ETFS +
    DEFENSE_ETFS + GOLD_ETFS + BIOTECH_ETFS + CHINA_ETFS +
    BOND_ETFS + INTL_ETFS + COMMODITY_ETFS +
    [CASH_ETF]
)
ALL_TICKERS        = ETF_UNIVERSE + [SPY_TICKER]
TRAILING_STOP_ETFS = [e for e in ETF_UNIVERSE if e != CASH_ETF]

# ── V2e-specific overrides ────────────────────────────────────────────────────
# New supercycle lookback weights (within the slow blend, NaN-safe during warmup)
W_SLOW_24M = 0.20   # 2-year momentum
W_SLOW_36M = 0.10   # 3-year momentum

REBALANCE_FREQ  = "W"
N_POSITIONS     = 6
MAX_WEIGHT      = 0.25
CORR_LOOKBACK   = 60
CORR_THRESHOLD  = 0.70

BACKTEST_START  = "2000-01-01"

# ── Cluster caps (identical to V2d) ───────────────────────────────────────────
CLUSTER_CAPS = {
    "precious_miners": 1,   # GDX, GDXJ, SIL
    "base_miners":     1,   # XME, COPX, REMX
    "green_energy":    1,   # ICLN, URA
    "biotech":         1,   # XBI, IBB
    "energy":          1,   # XOP, OIH
    "semis_ai":        2,   # SMH, ARKK, IGV
    "bonds":           2,   # TLT, IEF, HYG
    "intl_equity":     2,   # EFA, EEM, EWJ, EWZ, INDA
}

CLUSTERS = {
    "precious_miners": ["GDX", "GDXJ", "SIL"],
    "base_miners":     ["XME", "COPX", "REMX"],
    "green_energy":    ["ICLN", "URA"],
    "biotech":         ["XBI", "IBB"],
    "energy":          ["XOP", "OIH"],
    "semis_ai":        ["SMH", "ARKK", "IGV"],
    "bonds":           ["TLT", "IEF", "HYG"],
    "intl_equity":     ["EFA", "EEM", "EWJ", "EWZ", "INDA"],
}

ETF_TO_CLUSTER = {
    etf: cluster
    for cluster, etfs in CLUSTERS.items()
    for etf in etfs
}

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cache")
LOG_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")

# ── Optimisation search space ─────────────────────────────────────────────────
PARAM_SPACE = {
    **BASE_PARAM_SPACE,
    "MAX_WEIGHT":     ("float", 0.10, 0.35),
    "N_POSITIONS":    ("int",   5,   10),
    "CORR_THRESHOLD": ("float", 0.40, 0.90),
    "W_SLOW_24M":     ("float", 0.00, 0.35),
    "W_SLOW_36M":     ("float", 0.00, 0.30),
}
