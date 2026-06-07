import os
from configs.sector_v2f import *  # noqa: F401,F403 — inherit universe, clusters, signal

# ── V2g: HONEST rebuild — concentrated + slow trend-break exit ───────────────
# Built AFTER the trailing-stop look-ahead bug was fixed (see memory:
# stop-lookahead-bias). On the corrected engine, research showed the adaptive
# %-off-peak trailing stop is the WORST exit — it whipsaws volatile thematic /
# supercycle ETFs and destroys value vs holding. This variant instead:
#   • exits on a slow moving-average trend break (EXIT_MODE="ma_break"), and
#   • concentrates into fewer high-conviction winners (N_POSITIONS small),
# i.e. "ride winners, cut losers slowly" — the design the data actually favours.
# Same UCITS-tradeable universe as V2f (Nordnet-executable). NOTE: leverage
# scales CAGR and drawdown together — it does NOT improve Sharpe.

EXIT_MODE      = "ma_break"
MA_EXIT_WINDOW = 200          # exit a holding when its close < its own 200d MA

N_POSITIONS    = 3            # concentrate
MAX_WEIGHT     = 0.40
VOL_TARGET     = 0.14
MAX_LEVERAGE   = 1.0

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cache")
LOG_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")

# Focused search space: drop the %-stop params (irrelevant under ma_break),
# add MA_EXIT_WINDOW and MAX_LEVERAGE.
PARAM_SPACE = {
    "W_SLOW_12M":     ("float", 0.15, 0.55),
    "W_SLOW_18M":     ("float", 0.10, 0.45),
    "W_FAST_1M":      ("float", 0.05, 0.30),
    "W_FAST_3M":      ("float", 0.10, 0.40),
    "ALPHA_SLOW":     ("float", 0.35, 0.80),
    "BETA_ACCEL":     ("float", 0.00, 0.30),
    "N_POSITIONS":    ("int",   3,   8),
    "VOL_WINDOW":     ("int",   10,  40),
    "SPY_MA_WINDOW":  ("int",   100, 250),
    "MA_EXIT_WINDOW": ("int",   100, 250),
    "VOL_TARGET":     ("float", 0.08, 0.22),
    "MAX_LEVERAGE":   ("float", 1.0,  2.0),
    "DD_THRESHOLD":   ("float", -0.20, -0.04),
    "MAX_WEIGHT":     ("float", 0.20, 0.50),
    "CORR_THRESHOLD": ("float", 0.40, 0.90),
    "W_SLOW_24M":     ("float", 0.00, 0.35),
    "W_SLOW_36M":     ("float", 0.00, 0.30),
}
