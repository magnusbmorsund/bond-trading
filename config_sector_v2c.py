import os
from config_sector_base import *  # noqa: F401,F403

# ── V2c universe (V2b + cross-asset diversifiers) ─────────────────────────────
# V2b base
SECTOR_CORE   = ["XLE", "XLK", "XLV", "XLF", "XLI", "XLY", "XLP", "XLU", "XLB", "VNQ"]
COMPUTE_ETFS  = ["SMH", "BOTZ", "QTUM", "ARKK", "CLOU"]
METALS_ETFS   = ["GDX", "GDXJ", "SIL", "XME", "COPX", "REMX"]
ENERGY_ETFS   = ["XOP", "OIH", "FCG"]
GREEN_ETFS    = ["ICLN", "URNM", "NLR", "LIT", "TAN"]
DEFENSE_ETFS  = ["ITA"]
GOLD_ETFS     = ["GLD"]
BIOTECH_ETFS  = ["XBI", "IBB", "MOO"]
CHINA_ETFS    = ["KWEB"]

# New cross-asset diversifiers (V2c additions — all >$6B AUM, >$180M daily vol)
BOND_ETFS     = ["TLT", "IEF", "HYG"]   # Long-term, intermediate, high-yield
INTL_ETFS     = ["EFA", "EEM", "EWJ", "EWZ", "INDA"]  # Developed ex-US, EM, Japan, Brazil, India
COMMODITY_ETFS = ["PDBC"]               # Broad commodity index (replaces DBC)

ETF_UNIVERSE = (
    SECTOR_CORE + COMPUTE_ETFS +
    METALS_ETFS + ENERGY_ETFS + GREEN_ETFS +
    DEFENSE_ETFS + GOLD_ETFS + BIOTECH_ETFS + CHINA_ETFS +
    BOND_ETFS + INTL_ETFS + COMMODITY_ETFS +
    [CASH_ETF]
)
ALL_TICKERS        = ETF_UNIVERSE + [SPY_TICKER]
TRAILING_STOP_ETFS = [e for e in ETF_UNIVERSE if e != CASH_ETF]

# ── V2c-specific overrides ────────────────────────────────────────────────────
REBALANCE_FREQ   = "W"
N_POSITIONS      = 6    # more slots to exploit the wider universe
MAX_WEIGHT       = 0.25  # tighter cap — more diversified universe
CORR_LOOKBACK    = 60   # days of daily returns used to compute correlation at rebalance
CORR_THRESHOLD   = 0.70  # reject a candidate if it exceeds this correlation with any held position

# ── Cluster caps ──────────────────────────────────────────────────────────────
# At most CLUSTER_CAPS[cluster] ETFs from a correlated group held simultaneously.
# ETFs not listed in any cluster are unconstrained.
CLUSTER_CAPS = {
    "precious_miners": 1,   # GDX, GDXJ, SIL
    "base_miners":     1,   # XME, COPX, REMX
    "green_energy":    1,   # ICLN, URNM, NLR, LIT, TAN
    "biotech":         1,   # XBI, IBB, MOO
    "energy":          1,   # XOP, OIH, FCG
    "semis_ai":        2,   # SMH, BOTZ, QTUM, ARKK, CLOU
    "bonds":           2,   # TLT, IEF, HYG
    "intl_equity":     2,   # EFA, EEM, EWJ, EWZ, INDA
}

CLUSTERS = {
    "precious_miners": ["GDX", "GDXJ", "SIL"],
    "base_miners":     ["XME", "COPX", "REMX"],
    "green_energy":    ["ICLN", "URNM", "NLR", "LIT", "TAN"],
    "biotech":         ["XBI", "IBB", "MOO"],
    "energy":          ["XOP", "OIH", "FCG"],
    "semis_ai":        ["SMH", "BOTZ", "QTUM", "ARKK", "CLOU"],
    "bonds":           ["TLT", "IEF", "HYG"],
    "intl_equity":     ["EFA", "EEM", "EWJ", "EWZ", "INDA"],
}

# Reverse lookup: ETF → cluster name (built once at import)
ETF_TO_CLUSTER = {
    etf: cluster
    for cluster, etfs in CLUSTERS.items()
    for etf in etfs
}

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "cache")
LOG_DIR  = os.path.join(os.path.dirname(__file__), "logs")

# ── Optimisation search space ─────────────────────────────────────────────────
PARAM_SPACE = {
    **BASE_PARAM_SPACE,
    "MAX_WEIGHT":      ("float", 0.10, 0.35),
    "N_POSITIONS":     ("int",   5,   10),
    "CORR_THRESHOLD":  ("float", 0.40, 0.90),
}
