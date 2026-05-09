import os

# --- Universe ---
SECTOR_CORE   = ["XLE", "XLK", "XLV", "XLF", "XLI", "XLY", "XLP", "XLU", "XLB", "VNQ"]
COMPUTE_ETFS  = ["SMH", "BOTZ", "QTUM", "ARKK", "CLOU"]  # CLOU=cloud/data-centre; VPN dropped (invalid ticker)
SHIPPING_ETFS = []                                         # BDRY dropped (not on Nordnet, illiquid)
METALS_ETFS   = ["GDX", "GDXJ", "SIL", "XME", "COPX", "REMX"]
ENERGY_ETFS   = ["XOP", "OIH", "FCG"]
GREEN_ETFS    = ["ICLN", "URNM", "NLR", "LIT"]            # HYDR dropped ($118M AUM, ~$181K daily vol)
BIO_ETFS      = ["IBB", "MOO"]

CASH_ETF   = "SHY"
SPY_TICKER = "SPY"

# All tradeable ETFs (no SPY — SPY is regime filter only)
ETF_UNIVERSE = (
    SECTOR_CORE + COMPUTE_ETFS +
    METALS_ETFS + ENERGY_ETFS + GREEN_ETFS + BIO_ETFS + [CASH_ETF]
)
ALL_TICKERS = ETF_UNIVERSE + [SPY_TICKER]

# ETFs subject to trailing stops (everything except cash)
TRAILING_STOP_ETFS = [e for e in ETF_UNIVERSE if e != CASH_ETF]

# --- Multi-timescale momentum signal ---
# Composite score = ALPHA_SLOW * slow + (1-ALPHA_SLOW) * fast + BETA_ACCEL * accel
# slow  = weighted avg of 12M + 18M lookbacks  (supercycle detection)
# fast  = weighted avg of 1M + 3M lookbacks    (tactical entry)
# accel = (mom_3m - mom_6m).clip(0)            (positive acceleration bonus)
W_SLOW_12M = 0.35    # 12-month weight inside slow composite
W_SLOW_18M = 0.25    # 18-month weight inside slow composite
W_FAST_1M  = 0.15    # 1-month weight inside fast composite
W_FAST_3M  = 0.25    # 3-month weight inside fast composite
ALPHA_SLOW = 0.60    # blend: ALPHA_SLOW * slow + (1-ALPHA_SLOW) * fast
BETA_ACCEL = 0.10    # additive weight on acceleration term

ONLY_POSITIVE_COMPOSITE = True   # skip ETFs whose composite score ≤ 0

# --- Portfolio construction ---
N_POSITIONS = 8      # max simultaneous holdings (excluding SHY)

# --- Regime filter ---
SPY_MA_WINDOW  = 200
SPY_MA_CONFIRM = 2   # consecutive below-MA days before flipping defensive

# --- Adaptive trailing stops ---
# Stop % linearly interpolates between STOP_TACTICAL and STOP_SUPERCYCLE
# based on where the ETF's 12M momentum sits in [TACTICAL_MOM_THRESHOLD, SUPERCYCLE_MOM_THRESHOLD]
SUPERCYCLE_MOM_THRESHOLD = 0.80   # 80%+ 12M return → use wide supercycle stop
TACTICAL_MOM_THRESHOLD   = 0.20   # <20% 12M return → use tight tactical stop
STOP_SUPERCYCLE = 0.22            # 22% wide stop for confirmed supercycle
STOP_TACTICAL   = 0.10            # 10% tight stop for tactical/low-momentum positions
TRAILING_STOP_WINDOW = 63         # rolling peak lookback (days)

# --- Weighting ---
VOL_WINDOW = 20      # days of realised vol for inverse-vol weighting
MAX_WEIGHT = 0.35    # max single-ETF weight (lower cap vs V1 for better diversification)

# --- Volatility targeting ---
VOL_TARGET   = 0.14
MAX_LEVERAGE = 1.0
VOL_LOOKBACK = 21

# --- Drawdown overlay ---
DD_THRESHOLD = -0.10
DD_SCALE     = 0.00   # full exit on distress

# --- Backtest ---
BACKTEST_START = "2010-01-01"   # SMH/GDX/XME all have data; BDRY/QTUM etc join later via NaN
REBALANCE_FREQ = "ME"

# --- Numeric stability ---
MIN_VOL_CLIP          = 0.01
MIN_WEIGHT_THRESHOLD  = 1e-4
PRICE_SPIKE_THRESHOLD = 0.25   # wider for thematic ETFs (e.g. BDRY can move 15%/day)

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "cache")
LOG_DIR  = os.path.join(os.path.dirname(__file__), "logs")

# --- Optimisation search space ---
PARAM_SPACE = {
    # Multi-timescale blend weights
    "W_SLOW_12M":   ("float", 0.15, 0.55),
    "W_SLOW_18M":   ("float", 0.10, 0.45),
    "W_FAST_1M":    ("float", 0.05, 0.30),
    "W_FAST_3M":    ("float", 0.10, 0.40),
    "ALPHA_SLOW":   ("float", 0.35, 0.80),
    "BETA_ACCEL":   ("float", 0.00, 0.30),
    # Portfolio structure
    "N_POSITIONS":  ("int",   4,   12),
    "MAX_WEIGHT":   ("float", 0.20, 0.50),
    "VOL_WINDOW":   ("int",   10,   40),
    # Regime
    "SPY_MA_WINDOW": ("int",  100, 250),
    # Adaptive stops
    "SUPERCYCLE_MOM_THRESHOLD": ("float", 0.50, 1.50),
    "TACTICAL_MOM_THRESHOLD":   ("float", 0.10, 0.45),
    "STOP_SUPERCYCLE":          ("float", 0.14, 0.30),
    "STOP_TACTICAL":            ("float", 0.04, 0.16),
    "TRAILING_STOP_WINDOW":     ("int",   42,  126),
    # Vol targeting
    "VOL_TARGET":   ("float", 0.08, 0.22),
    # DD overlay
    "DD_THRESHOLD": ("float", -0.20, -0.04),
}
