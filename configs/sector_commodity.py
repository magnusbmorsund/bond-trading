import os
from configs.sector_base import *  # noqa: F401,F403

# ── Commodity Supercycle: pure commodity-complex momentum rotation ───────────
# Rides multi-year commodity up-legs ("supercycles") via the V2e multi-timescale
# momentum signal (incl. 24m/36m lookbacks), restricted to a COMMODITY-ONLY
# universe, with a FRED MACRO EXPOSURE OVERLAY (broad USD + 10y real yield +
# CPI/industrial-production momentum) that scales basket exposure DOWN when the
# macro backdrop is hostile to commodities.
#
# Design honesty notes:
#   • A supercycle PREDICTOR is not backtestable (~1.5 supercycles in the data).
#     The cross-sectional momentum engine RIDES the up-leg; it does not predict
#     the cycle. The macro overlay is CONFIRMATION/exposure-scaling, never a
#     timing oracle (macro timing is "deceptively difficult" — Asness 2017).
#   • Decoupled from equities on purpose: SPY-trend regime gate is NEUTRALISED
#     (see strategies/sector_commodity/signals.spy_regime) so the sleeve stays a
#     genuine equity diversifier (commodities can rally when stocks fall, 2022).
#   • No per-position trailing stops (TRAILING_STOP_ETFS=[]) — parsimony; defense
#     is the momentum re-rank + macro overlay + vol-target + drawdown overlay.

# ── Universe: commodity complexes only ───────────────────────────────────────
BROAD_ETFS      = ["PDBC", "DBC", "GSG"]            # broad commodity baskets
ENERGY_ETFS     = ["USO", "UNG", "XOP", "OIH"]      # crude, natgas, E&P, oil services
PRECIOUS_ETFS   = ["GLD", "SLV", "GDX", "GDXJ", "SIL"]
BASE_ETFS       = ["DBB", "CPER", "XME", "COPX"]    # industrial/base metals + miners
AG_ETFS         = ["DBA"]
TRANSITION_ETFS = ["LIT", "URA", "REMX"]            # lithium, uranium, rare earth

COMMODITY_UNIVERSE = (
    BROAD_ETFS + ENERGY_ETFS + PRECIOUS_ETFS + BASE_ETFS + AG_ETFS + TRANSITION_ETFS
)
ETF_UNIVERSE       = COMMODITY_UNIVERSE + [CASH_ETF]
ALL_TICKERS        = ETF_UNIVERSE + [SPY_TICKER]
SECTOR_CORE        = BROAD_ETFS              # benchmark = equal-weight broad commodities
TRAILING_STOP_ETFS = []                      # no per-position stops

# ── Supercycle momentum lookbacks (NaN-safe blend; warms ~3y after start) ────
W_SLOW_24M = 0.20
W_SLOW_36M = 0.10

REBALANCE_FREQ = "W"
N_POSITIONS    = 5
MAX_WEIGHT     = 0.30
CORR_LOOKBACK  = 60
CORR_THRESHOLD = 0.75
BACKTEST_START = "2010-01-01"

# ── Cluster caps (don't load several correlated sub-baskets at once) ─────────
CLUSTER_CAPS = {
    "broad":       1,
    "energy":      2,
    "precious":    2,
    "base_metals": 2,
    "agriculture": 1,
    "transition":  2,
}
CLUSTERS = {
    "broad":       BROAD_ETFS,
    "energy":      ENERGY_ETFS,
    "precious":    PRECIOUS_ETFS,
    "base_metals": BASE_ETFS,
    "agriculture": AG_ETFS,
    "transition":  TRANSITION_ETFS,
}
ETF_TO_CLUSTER = {e: c for c, etfs in CLUSTERS.items() for e in etfs}

# ── Macro exposure overlay (commodity-appropriate regime control) ────────────
# Daily multiplier ∈ [MACRO_FLOOR, MACRO_CEIL], applied as the OUTERMOST scaler on
# strategy returns (after vol-target + DD overlay), lagged so day-T exposure uses
# only macro data known before T. macro_z>0 ⇒ commodity-friendly backdrop (weak/
# falling USD, falling real yields, rising inflation/industrial demand). With
# MACRO_CEIL=1.0 the overlay can ONLY de-risk — it cannot manufacture return, so
# it cannot inflate the Sharpe (it can only protect against macro headwinds).
# MACRO_FLOOR=1.0 ⇒ overlay effectively OFF, so the optimiser is free to discover
# the macro overlay adds nothing (an honest, falsifiable test).
MACRO_ENABLED        = True
MACRO_Z_LOOKBACK_M   = 60     # months for z-score normalisation
MACRO_PUB_LAG_M      = 1      # publication-lag shift (months) — conservative
MACRO_USD_MOM_M      = 6      # broad-USD momentum horizon (months)
MACRO_REAL_MOM_M     = 3      # 10y real-yield momentum horizon (months)
MACRO_INFL_HORIZON_M = 12     # YoY horizon for CPI / industrial production
MACRO_W_USD          = 0.40
MACRO_W_REAL         = 0.35
MACRO_W_INFL         = 0.25
MACRO_GAIN           = 0.35   # sensitivity of exposure to macro_z
MACRO_FLOOR          = 0.30   # never de-risk below 30% on macro alone (tilt, not gate)
MACRO_CEIL           = 1.00   # overlay only reduces exposure (no leverage stacking)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cache")
LOG_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")

# ── Optimisation search space ─────────────────────────────────────────────────
PARAM_SPACE = {
    **BASE_PARAM_SPACE,
    "MAX_WEIGHT":     ("float", 0.15, 0.40),
    "N_POSITIONS":    ("int",   4,   8),
    "CORR_THRESHOLD": ("float", 0.50, 0.90),
    "W_SLOW_24M":     ("float", 0.00, 0.35),
    "W_SLOW_36M":     ("float", 0.00, 0.30),
    # macro overlay (FLOOR upper bound = 1.0 lets the search disable it entirely)
    "MACRO_GAIN":     ("float", 0.00, 0.80),
    "MACRO_FLOOR":    ("float", 0.20, 1.00),
    "MACRO_W_USD":    ("float", 0.00, 1.00),
    "MACRO_W_REAL":   ("float", 0.00, 1.00),
    "MACRO_W_INFL":   ("float", 0.00, 1.00),
}
