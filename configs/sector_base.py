"""
Shared defaults and BASE_PARAM_SPACE for V2/V2b sector rotation strategies.

Usage in version configs:
    from config_sector_base import *   # pull in all shared defaults
    # then override only what differs (REBALANCE_FREQ, N_POSITIONS, etc.)
    # and define DATA_DIR / LOG_DIR from os.path.dirname(__file__) locally.

Optuna patches individual module attributes via setattr(config_module, k, v),
so each version config must be its own module — but they share defaults here.
"""

CASH_ETF   = "SHY"
SPY_TICKER = "SPY"

# ── Multi-timescale momentum signal ─────────────────────────────────────────
W_SLOW_12M = 0.35
W_SLOW_18M = 0.25
W_FAST_1M  = 0.15
W_FAST_3M  = 0.25
ALPHA_SLOW = 0.60
BETA_ACCEL = 0.10

ONLY_POSITIVE_COMPOSITE = True

# ── Regime filter ────────────────────────────────────────────────────────────
SPY_MA_WINDOW  = 200
SPY_MA_CONFIRM = 2

# ── Adaptive trailing stops ──────────────────────────────────────────────────
SUPERCYCLE_MOM_THRESHOLD = 0.80
TACTICAL_MOM_THRESHOLD   = 0.20
STOP_SUPERCYCLE          = 0.22
STOP_TACTICAL            = 0.10
TRAILING_STOP_WINDOW     = 63

# ── Portfolio construction ───────────────────────────────────────────────────
VOL_WINDOW = 20

# ── Volatility targeting ─────────────────────────────────────────────────────
VOL_TARGET   = 0.14
MAX_LEVERAGE = 1.0
VOL_LOOKBACK = 21

# ── Drawdown overlay ─────────────────────────────────────────────────────────
DD_THRESHOLD = -0.10
DD_SCALE     = 0.00

# ── Backtest ─────────────────────────────────────────────────────────────────
BACKTEST_START = "2010-01-01"
REBALANCE_FREQ = "ME"

# ── Numeric stability ─────────────────────────────────────────────────────────
MIN_VOL_CLIP          = 0.01
MIN_WEIGHT_THRESHOLD  = 1e-4
PRICE_SPIKE_THRESHOLD = 0.25

# ── Base optimisation search space ───────────────────────────────────────────
# Version configs extend this with their own MAX_WEIGHT range:
#   PARAM_SPACE = {**BASE_PARAM_SPACE, "MAX_WEIGHT": ("float", lo, hi)}
BASE_PARAM_SPACE = {
    "W_SLOW_12M":               ("float", 0.15, 0.55),
    "W_SLOW_18M":               ("float", 0.10, 0.45),
    "W_FAST_1M":                ("float", 0.05, 0.30),
    "W_FAST_3M":                ("float", 0.10, 0.40),
    "ALPHA_SLOW":               ("float", 0.35, 0.80),
    "BETA_ACCEL":               ("float", 0.00, 0.30),
    "N_POSITIONS":              ("int",   4,   12),
    "VOL_WINDOW":               ("int",   10,  40),
    "SPY_MA_WINDOW":            ("int",   100, 250),
    "SUPERCYCLE_MOM_THRESHOLD": ("float", 0.50, 1.50),
    "TACTICAL_MOM_THRESHOLD":   ("float", 0.10, 0.45),
    "STOP_SUPERCYCLE":          ("float", 0.14, 0.30),
    "STOP_TACTICAL":            ("float", 0.04, 0.16),
    "TRAILING_STOP_WINDOW":     ("int",   42,  126),
    "VOL_TARGET":               ("float", 0.08, 0.22),
    "DD_THRESHOLD":             ("float", -0.20, -0.04),
}
