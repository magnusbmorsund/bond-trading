import os
from configs.sector_base import *  # noqa: F401,F403

# ── V2f: V2e universe minus XBI and IGV ──────────────────────────────────────
# Same multi-timescale momentum logic as V2e (24m/36m supercycle lookbacks).
# The only difference vs V2e is that XBI and IGV are removed from the ETF
# universe — neither has an acceptable UCITS equivalent on Nordnet, so this
# variant is the one a Norwegian retail investor can actually execute live
# via the UCITS mapping. The full UCITS-tradeable subset of V2e.
#
# What-if backtest (US prices as UCITS proxy, +15bp TER drag) showed only
# -0.3pp CAGR loss vs V2e (37.9% vs 38.2%, Sharpe 3.23, MaxDD -7.5%) over
# 2000-2026 — see scripts/backtest_v2e_ucits.py and memory note.

# V2b base sectors
SECTOR_CORE   = ["XLE", "XLK", "XLV", "XLF", "XLI", "XLY", "XLP", "XLU", "XLB", "VNQ"]

# Compute: SMH + ARKK only (drop IGV — no UCITS equivalent)
COMPUTE_ETFS  = ["SMH", "ARKK"]

METALS_ETFS   = ["GDX", "GDXJ", "SIL", "XME", "COPX", "REMX"]
ENERGY_ETFS   = ["XOP", "OIH"]
GREEN_ETFS    = ["ICLN", "URA"]
DEFENSE_ETFS  = ["ITA"]
GOLD_ETFS     = ["GLD"]

# Biotech: IBB only (drop XBI — no UCITS equivalent for equal-weight biotech)
BIOTECH_ETFS  = ["IBB"]

CHINA_ETFS    = ["KWEB"]

# Cross-asset diversifiers (same as V2e)
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

# ── V2e-inherited overrides ───────────────────────────────────────────────────
W_SLOW_24M = 0.20
W_SLOW_36M = 0.10

REBALANCE_FREQ  = "W"
N_POSITIONS     = 6
MAX_WEIGHT      = 0.25
CORR_LOOKBACK   = 60
CORR_THRESHOLD  = 0.70

BACKTEST_START  = "2000-01-01"

# ── Cluster caps (XBI removed from biotech, IGV removed from semis_ai) ───────
CLUSTER_CAPS = {
    "precious_miners": 1,   # GDX, GDXJ, SIL
    "base_miners":     1,   # XME, COPX, REMX
    "green_energy":    1,   # ICLN, URA
    "biotech":         1,   # IBB only
    "energy":          1,   # XOP, OIH
    "semis_ai":        2,   # SMH, ARKK
    "bonds":           2,   # TLT, IEF, HYG
    "intl_equity":     2,   # EFA, EEM, EWJ, EWZ, INDA
}

CLUSTERS = {
    "precious_miners": ["GDX", "GDXJ", "SIL"],
    "base_miners":     ["XME", "COPX", "REMX"],
    "green_energy":    ["ICLN", "URA"],
    "biotech":         ["IBB"],
    "energy":          ["XOP", "OIH"],
    "semis_ai":        ["SMH", "ARKK"],
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

# ── Optimisation search space (same as V2e) ──────────────────────────────────
PARAM_SPACE = {
    **BASE_PARAM_SPACE,
    "MAX_WEIGHT":     ("float", 0.10, 0.35),
    "N_POSITIONS":    ("int",   5,   10),
    "CORR_THRESHOLD": ("float", 0.40, 0.90),
    "W_SLOW_24M":     ("float", 0.00, 0.35),
    "W_SLOW_36M":     ("float", 0.00, 0.30),
}
