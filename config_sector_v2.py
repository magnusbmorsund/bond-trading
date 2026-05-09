import os
from config_sector_base import *  # noqa: F401,F403

# ── V2 universe (32 ETFs) ─────────────────────────────────────────────────────
SECTOR_CORE   = ["XLE", "XLK", "XLV", "XLF", "XLI", "XLY", "XLP", "XLU", "XLB", "VNQ"]
COMPUTE_ETFS  = ["SMH", "BOTZ", "QTUM", "ARKK", "CLOU"]  # CLOU=cloud/data-centre
SHIPPING_ETFS = []                                         # BDRY dropped (not on Nordnet, illiquid)
METALS_ETFS   = ["GDX", "GDXJ", "SIL", "XME", "COPX", "REMX"]
ENERGY_ETFS   = ["XOP", "OIH", "FCG"]
GREEN_ETFS    = ["ICLN", "URNM", "NLR", "LIT"]            # HYDR dropped ($118M AUM)
BIO_ETFS      = ["IBB", "MOO"]

ETF_UNIVERSE = (
    SECTOR_CORE + COMPUTE_ETFS +
    METALS_ETFS + ENERGY_ETFS + GREEN_ETFS + BIO_ETFS + [CASH_ETF]
)
ALL_TICKERS        = ETF_UNIVERSE + [SPY_TICKER]
TRAILING_STOP_ETFS = [e for e in ETF_UNIVERSE if e != CASH_ETF]

# ── V2-specific overrides ─────────────────────────────────────────────────────
N_POSITIONS = 8
MAX_WEIGHT  = 0.35   # wider cap vs V2b — smaller universe

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "cache")
LOG_DIR  = os.path.join(os.path.dirname(__file__), "logs")

# ── Optimisation search space ─────────────────────────────────────────────────
PARAM_SPACE = {**BASE_PARAM_SPACE, "MAX_WEIGHT": ("float", 0.20, 0.50)}
