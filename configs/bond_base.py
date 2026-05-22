"""
Shared defaults for all bond strategy config modules (V1 / V2 / V3).

Usage in version configs:
    from configs.bond_base import *   # pull in all shared defaults
    # then override only what differs (ETF_UNIVERSE, FRED_SERIES, signal weights, etc.)
    # PARAM_SPACE = {**BASE_PARAM_SPACE, <version-specific params>}

Optuna patches individual module attributes via setattr(config_module, k, v), so
each version config must be its own module — but they share defaults here.
"""
import os

# --- API Keys ---
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# --- Backtest ---
BACKTEST_START  = "2003-01-01"
REBALANCE_FREQ  = "ME"

# --- Signal windows ---
LOOKBACK_SIGNAL = 252
LOOKBACK_VOL    = 63
MOMENTUM_WINDOW = 252
MOMENTUM_SKIP   = 21

# --- Allocation limits (shared starting point; versions may override) ---
MAX_CREDIT_ALLOC = 0.50
MAX_TIP_ALLOC    = 0.15
MAX_ALT_ALLOC    = 0.40
SIGNAL_BLEND     = 0.30

# --- Drawdown overlay ---
DD_THRESHOLD = -0.05
DD_SCALE     = 0.00

# --- Trailing stops ---
TRAILING_STOP_PCT    = 0.04
TRAILING_STOP_WINDOW = 21

# --- Volatility targeting ---
VOL_TARGET   = 0.08
MAX_LEVERAGE = 1.0
VOL_LOOKBACK = 21

# --- VIX thresholds ---
VIX_RISK_OFF = 25.0
VIX_RISK_ON  = 15.0

# --- Numeric stability ---
MIN_ZSCORE_CLIP      = 1e-6
MIN_WEIGHT_THRESHOLD = 1e-4
MIN_VOL_CLIP         = 0.01
INV_VOL_CLIP         = 0.001
TANH_COMM_SCALE      = 0.8

# --- Data quality ---
PRICE_SPIKE_THRESHOLD = 0.15

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cache")
LOG_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")

# --- Base optimisation search space (all 3 versions share these 15 entries) ---
# Version configs extend: PARAM_SPACE = {**BASE_PARAM_SPACE, <version-specific>}
BASE_PARAM_SPACE = {
    "LOOKBACK_SIGNAL":       ("int",    84,   504,  21),
    "LOOKBACK_VOL":          ("int",    21,   126,  21),
    "MOMENTUM_WINDOW":       ("int",   126,   504,  21),
    "MOMENTUM_SKIP":         ("int",     0,    42,   5),
    "MAX_CREDIT_ALLOC":      ("float", 0.30,  0.80, 0.05),
    "MAX_TIP_ALLOC":         ("float", 0.00,  0.30, 0.05),
    "SIGNAL_BLEND":          ("float", 0.00,  1.00, 0.10),
    "VOL_TARGET":            ("float", 0.05,  0.15, 0.01),
    "VIX_RISK_OFF":          ("float", 18.0,  40.0, 1.0),
    "VIX_RISK_ON":           ("float", 10.0,  22.0, 1.0),
    "DD_THRESHOLD":          ("float", -0.15, -0.02, 0.01),
    "DD_SCALE":              ("float",  0.00,  0.50, 0.05),
    "TRAILING_STOP_PCT":     ("float", 0.03,  0.15, 0.01),
    "TRAILING_STOP_WINDOW":  ("int",   21,   126,  21),
    "MAX_ALT_ALLOC":         ("float", 0.20,  0.60, 0.05),
}
