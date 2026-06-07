import os
from configs.sector_v2f import *  # noqa: F401,F403 — inherit UCITS universe, clusters, signal

# ── V2h: tight trailing stop, FILLED AT THE STOP PRICE ───────────────────────
# Research (see memory: stop-fill-model) showed the value of a trailing stop on
# this momentum/supercycle universe depends almost entirely on the FILL price.
# Booking exits at the close BELOW the stop makes stops look useless; booking
# them at the STOP LEVEL (the realistic resting-stop fill on a liquid ETF) makes
# a tight ~10% trailing stop excellent and OOS-robust — consistent with
# Kaminski & Lo (stops add 50–100 bps/month for momentum, using a 10% stop).
#
# Execution model (baked into the backtest so the headline is GAP-AWARE, not the
# clean-fill best case):
#   • fill at the stop level, minus STOP_SLIP_BPS of slippage, and
#   • a deterministic fat-tail gap: every 1/GAP_FRAC-th trigger fills GAP_EXTRA
#     worse than the stop (proxy for overnight gaps through the stop).
# Nordnet trailing stops are not guaranteed but let you cap the trigger/exec
# deviation; fills are near the stop in normal liquidity, worse in gaps.

EXIT_MODE      = "fixed_trail"   # fixed % trailing stop (not adaptive)
STOP_FILL      = "stop"          # book the trigger day at the stop LEVEL
STOP_PERSIST   = True            # stopped → cash until the next rebalance
STOP_FIXED_PCT = 0.10            # ~10% trailing distance (Kaminski–Lo)

STOP_SLIP_BPS  = 30.0            # slippage haircut on every fill
GAP_FRAC       = 0.10            # 1-in-10 triggers …
GAP_EXTRA      = 0.05            # … fill 5% worse than the stop (gap proxy)

N_POSITIONS    = 4               # concentrate into winners
MAX_WEIGHT     = 0.40
VOL_TARGET     = 0.14
MAX_LEVERAGE   = 1.0              # FIXED, not optimized — leverage only scales
                                 # CAGR & drawdown together, never Sharpe, and
                                 # lets the optimizer game a tight-stop corner.
TRAILING_STOP_WINDOW = 86

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cache")
LOG_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")

# Tight, honest search space. The execution-realism knobs (STOP_SLIP_BPS,
# GAP_FRAC, GAP_EXTRA, STOP_FILL) are deliberately NOT optimized — they are
# conservative assumptions, not free parameters to game.
PARAM_SPACE = {
    "W_SLOW_12M":           ("float", 0.15, 0.55),
    "W_SLOW_18M":           ("float", 0.10, 0.45),
    "W_FAST_1M":            ("float", 0.05, 0.30),
    "W_FAST_3M":            ("float", 0.10, 0.40),
    "ALPHA_SLOW":           ("float", 0.35, 0.80),
    "BETA_ACCEL":           ("float", 0.00, 0.30),
    "N_POSITIONS":          ("int",   3,    6),
    "VOL_WINDOW":           ("int",   10,   40),
    "SPY_MA_WINDOW":        ("int",   100,  250),
    "STOP_FIXED_PCT":       ("float", 0.08, 0.16),   # honest ~8-16%; no ultra-tight
                                                     # corner that over-relies on fills
    "TRAILING_STOP_WINDOW": ("int",   42,   126),
    "VOL_TARGET":           ("float", 0.08, 0.18),
    # MAX_LEVERAGE intentionally NOT optimized (fixed at 1.0 in config) — it only
    # scales CAGR & drawdown together and lets the optimizer game a tight corner.
    "DD_THRESHOLD":         ("float", -0.20, -0.04),
    "MAX_WEIGHT":           ("float", 0.20, 0.50),
    "CORR_THRESHOLD":       ("float", 0.40, 0.90),
    "W_SLOW_24M":           ("float", 0.00, 0.35),
    "W_SLOW_36M":           ("float", 0.00, 0.30),
}
