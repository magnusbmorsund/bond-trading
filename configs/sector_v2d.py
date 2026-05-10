import os
from configs.sector_base import *  # noqa: F401,F403

# ── V2d: liquid-filtered V2c universe (all ETFs ≥ $100M avg daily dollar vol) ──
# Removals vs V2c: CLOU($6M), LIT($32M), MOO($37M), BOTZ($38M), FCG($41M),
#                  URNM($49M), QTUM($51M), NLR($64M), TAN($77M)
# Additions vs V2c: IGV (replaces CLOU; cloud/software, $2.4B/day),
#                   URA (consolidates URNM+NLR; uranium/nuclear, $226M/day)

# V2b base sectors
SECTOR_CORE   = ["XLE", "XLK", "XLV", "XLF", "XLI", "XLY", "XLP", "XLU", "XLB", "VNQ"]

# Compute: SMH + ARKK kept; BOTZ/QTUM removed; CLOU → IGV
COMPUTE_ETFS  = ["SMH", "ARKK", "IGV"]

METALS_ETFS   = ["GDX", "GDXJ", "SIL", "XME", "COPX", "REMX"]  # unchanged
ENERGY_ETFS   = ["XOP", "OIH"]                                   # FCG removed
GREEN_ETFS    = ["ICLN", "URA"]                                   # URNM+NLR→URA; LIT+TAN removed
DEFENSE_ETFS  = ["ITA"]
GOLD_ETFS     = ["GLD"]
BIOTECH_ETFS  = ["XBI", "IBB"]                                   # MOO removed
CHINA_ETFS    = ["KWEB"]

# Cross-asset diversifiers (same as V2c)
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

# ── V2d-specific overrides ────────────────────────────────────────────────────
REBALANCE_FREQ  = "W"
N_POSITIONS     = 6
MAX_WEIGHT      = 0.25
CORR_LOOKBACK   = 60
CORR_THRESHOLD  = 0.70

# Same extended history as V2c
BACKTEST_START  = "2000-01-01"

# ── Cluster caps ──────────────────────────────────────────────────────────────
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
}
