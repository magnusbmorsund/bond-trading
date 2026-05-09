import os

# --- Universe ---
SECTOR_CORE = ["XLE", "XLK", "XLV", "XLF", "XLI", "XLY", "XLP", "XLU", "XLB", "VNQ"]
SECTOR_SUB  = ["SMH", "IBB", "NLR", "MOO"]
CASH_ETF    = "SHY"
SPY_TICKER  = "SPY"

# All ETFs the backtest needs prices for
ETF_UNIVERSE = SECTOR_CORE + SECTOR_SUB + [CASH_ETF]
ALL_TICKERS  = ETF_UNIVERSE + [SPY_TICKER]

# ETFs that get trailing stops in the backtest (all non-cash positions)
TRAILING_STOP_ETFS = SECTOR_CORE + SECTOR_SUB

# --- Signal Parameters ---
MOMENTUM_LOOKBACK = 126   # 6-month momentum (trading days)
MOMENTUM_SKIP     = 0     # no reversal skip for sector ETFs

N_POSITIONS       = 5     # how many sectors to hold simultaneously
ONLY_POSITIVE_MOM = True  # exclude sectors with negative absolute momentum

# --- Regime Filter ---
SPY_MA_WINDOW  = 200   # 200-day MA
SPY_MA_CONFIRM = 2     # consecutive days below MA before going defensive

# --- Trailing Stops ---
TRAILING_STOP_PCT    = 0.15   # 15% below rolling peak → exit
TRAILING_STOP_WINDOW = 63     # rolling peak lookback (days)

# --- Weighting ---
VOL_WINDOW   = 20     # days of realized vol for inverse-vol weighting
MAX_WEIGHT   = 0.40   # max single-sector weight
MIN_WEIGHT   = 0.05   # min weight if selected (prevents rounding-out)

# --- Volatility Targeting ---
VOL_TARGET   = 0.12   # 12% annual vol target (sectors more volatile than bonds)
MAX_LEVERAGE = 1.0    # no leverage
VOL_LOOKBACK = 21

# --- Drawdown Overlay ---
DD_THRESHOLD = -0.08   # 8% portfolio drawdown triggers defensive exit
DD_SCALE     = 0.00    # full exit (consistent with bond strategy design)

# --- Backtest ---
BACKTEST_START = "2007-01-01"   # NLR inception; SMH reliable from ~2011 (NaN-filled before)
REBALANCE_FREQ = "ME"

# --- Numeric stability ---
MIN_VOL_CLIP         = 0.01
MIN_WEIGHT_THRESHOLD = 1e-4
PRICE_SPIKE_THRESHOLD = 0.20   # wider threshold for volatile sector ETFs

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "cache")
LOG_DIR  = os.path.join(os.path.dirname(__file__), "logs")

# --- Optimisation search space ---
# Keys must match attribute names above; used by optimize.py --sector
PARAM_SPACE = {
    "MOMENTUM_LOOKBACK": ("int",   63,  252),    # 3–12 months
    "N_POSITIONS":        ("int",   3,    6),
    "TRAILING_STOP_PCT":  ("float", 0.08, 0.22),
    "VOL_WINDOW":         ("int",   10,   40),
    "MAX_WEIGHT":         ("float", 0.25, 0.55),
    "SPY_MA_WINDOW":      ("int",   100,  250),
    "VOL_TARGET":         ("float", 0.08, 0.18),
    "DD_THRESHOLD":       ("float", -0.15, -0.04),
}
