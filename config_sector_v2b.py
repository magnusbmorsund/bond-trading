import os

# --- Universe ---
SECTOR_CORE   = ["XLE", "XLK", "XLV", "XLF", "XLI", "XLY", "XLP", "XLU", "XLB", "VNQ"]
COMPUTE_ETFS  = ["SMH", "BOTZ", "QTUM", "ARKK", "CLOU"]
METALS_ETFS   = ["GDX", "GDXJ", "SIL", "XME", "COPX", "REMX"]
ENERGY_ETFS   = ["XOP", "OIH", "FCG"]
GREEN_ETFS    = ["ICLN", "URNM", "NLR", "LIT", "TAN"]   # TAN: solar supercycle ($2.5B AUM, 2008)

# New supercycle themes absent from V2
DEFENSE_ETFS  = ["ITA"]    # Aerospace & Defense ($8B AUM, 2001) — geopolitical cycle
GOLD_ETFS     = ["GLD"]    # Gold ($70B AUM, 2004) — inflation / safe-haven cycle
BIOTECH_ETFS  = ["XBI", "IBB", "MOO"]  # XBI=speculative genomics ($7B AUM, 2006)
CHINA_ETFS    = ["KWEB"]   # China Internet ($5B AUM, 2013) — EM tech supercycle

CASH_ETF   = "SHY"
SPY_TICKER = "SPY"

ETF_UNIVERSE = (
    SECTOR_CORE + COMPUTE_ETFS +
    METALS_ETFS + ENERGY_ETFS + GREEN_ETFS +
    DEFENSE_ETFS + GOLD_ETFS + BIOTECH_ETFS + CHINA_ETFS +
    [CASH_ETF]
)
ALL_TICKERS = ETF_UNIVERSE + [SPY_TICKER]

TRAILING_STOP_ETFS = [e for e in ETF_UNIVERSE if e != CASH_ETF]

# --- Rebalancing ---
REBALANCE_FREQ = "W"   # weekly  ("ME" = monthly like V2)

# --- Multi-timescale momentum signal ---
W_SLOW_12M = 0.35
W_SLOW_18M = 0.25
W_FAST_1M  = 0.15
W_FAST_3M  = 0.25
ALPHA_SLOW = 0.60
BETA_ACCEL = 0.10

ONLY_POSITIVE_COMPOSITE = True

# --- Portfolio construction ---
N_POSITIONS = 5       # slightly more positions vs V2's optimised 4, given wider universe

# --- Regime filter ---
SPY_MA_WINDOW  = 200
SPY_MA_CONFIRM = 2

# --- Adaptive trailing stops ---
SUPERCYCLE_MOM_THRESHOLD = 0.80
TACTICAL_MOM_THRESHOLD   = 0.20
STOP_SUPERCYCLE = 0.22
STOP_TACTICAL   = 0.10
TRAILING_STOP_WINDOW = 63

# --- Weighting ---
VOL_WINDOW = 20
MAX_WEIGHT = 0.30    # tighter cap — more ETFs means easier diversification

# --- Volatility targeting ---
VOL_TARGET   = 0.14
MAX_LEVERAGE = 1.0
VOL_LOOKBACK = 21

# --- Drawdown overlay ---
DD_THRESHOLD = -0.10
DD_SCALE     = 0.00

# --- Backtest ---
BACKTEST_START = "2010-01-01"

# --- Numeric stability ---
MIN_VOL_CLIP          = 0.01
MIN_WEIGHT_THRESHOLD  = 1e-4
PRICE_SPIKE_THRESHOLD = 0.25

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "cache")
LOG_DIR  = os.path.join(os.path.dirname(__file__), "logs")

# --- Optimisation search space ---
PARAM_SPACE = {
    "W_SLOW_12M":   ("float", 0.15, 0.55),
    "W_SLOW_18M":   ("float", 0.10, 0.45),
    "W_FAST_1M":    ("float", 0.05, 0.30),
    "W_FAST_3M":    ("float", 0.10, 0.40),
    "ALPHA_SLOW":   ("float", 0.35, 0.80),
    "BETA_ACCEL":   ("float", 0.00, 0.30),
    "N_POSITIONS":  ("int",   4,   12),
    "MAX_WEIGHT":   ("float", 0.15, 0.45),
    "VOL_WINDOW":   ("int",   10,   40),
    "SPY_MA_WINDOW": ("int",  100, 250),
    "SUPERCYCLE_MOM_THRESHOLD": ("float", 0.50, 1.50),
    "TACTICAL_MOM_THRESHOLD":   ("float", 0.10, 0.45),
    "STOP_SUPERCYCLE":          ("float", 0.14, 0.30),
    "STOP_TACTICAL":            ("float", 0.04, 0.16),
    "TRAILING_STOP_WINDOW":     ("int",   42,  126),
    "VOL_TARGET":   ("float", 0.08, 0.22),
    "DD_THRESHOLD": ("float", -0.20, -0.04),
}
