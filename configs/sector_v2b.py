import os
from configs.sector_base import *  # noqa: F401,F403

# ── V2b universe (37 ETFs — adds supercycle themes absent from V2) ────────────
SECTOR_CORE   = ["XLE", "XLK", "XLV", "XLF", "XLI", "XLY", "XLP", "XLU", "XLB", "VNQ"]
COMPUTE_ETFS  = ["SMH", "BOTZ", "QTUM", "ARKK", "CLOU"]
METALS_ETFS   = ["GDX", "GDXJ", "SIL", "XME", "COPX", "REMX"]
ENERGY_ETFS   = ["XOP", "OIH", "FCG"]
GREEN_ETFS    = ["ICLN", "URNM", "NLR", "LIT", "TAN"]   # TAN: solar supercycle ($2.5B AUM, 2008)
DEFENSE_ETFS  = ["ITA"]    # Aerospace & Defense ($8B AUM, 2001) — geopolitical cycle
GOLD_ETFS     = ["GLD"]    # Gold ($70B AUM, 2004) — inflation / safe-haven cycle
BIOTECH_ETFS  = ["XBI", "IBB", "MOO"]  # XBI=speculative genomics ($7B AUM, 2006)
CHINA_ETFS    = ["KWEB"]   # China Internet ($5B AUM, 2013) — EM tech supercycle

ETF_UNIVERSE = (
    SECTOR_CORE + COMPUTE_ETFS +
    METALS_ETFS + ENERGY_ETFS + GREEN_ETFS +
    DEFENSE_ETFS + GOLD_ETFS + BIOTECH_ETFS + CHINA_ETFS +
    [CASH_ETF]
)
ALL_TICKERS        = ETF_UNIVERSE + [SPY_TICKER]
TRAILING_STOP_ETFS = [e for e in ETF_UNIVERSE if e != CASH_ETF]

# ── V2b-specific overrides ────────────────────────────────────────────────────
REBALANCE_FREQ = "W"    # weekly (base default is "ME")
N_POSITIONS    = 5
MAX_WEIGHT     = 0.30   # tighter cap — more ETFs means easier diversification

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cache")
LOG_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")

# ── Optimisation search space ─────────────────────────────────────────────────
PARAM_SPACE = {**BASE_PARAM_SPACE, "MAX_WEIGHT": ("float", 0.15, 0.45)}
