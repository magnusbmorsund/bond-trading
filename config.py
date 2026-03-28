import os

# --- API Keys ---
FRED_API_KEY = os.getenv("FRED_API_KEY", "08b619abe65ba421c8b44520629987ba")

# --- Universe ---
# Duration
DURATION_ETFS = ["TLT", "IEF", "SHY"]
# Inflation-linked
INFLATION_ETF = "TIP"
# Credit / spread (IG → short-duration HY → floating → EM → preferred)
CREDIT_ETFS   = ["LQD", "HYG", "ANGL", "SJNK", "BKLN", "EMB", "PFF"]
# Crisis / alternative hedge — activates when both duration + credit signals turn negative
HEDGE_ETFS    = ["GLD"]
# T-bill cash parking — activates when duration signal is negative (inverted curve / rate hikes)
CASH_ETFS     = ["BIL"]

ETF_UNIVERSE  = DURATION_ETFS + [INFLATION_ETF] + CREDIT_ETFS + HEDGE_ETFS + CASH_ETFS

# --- FRED Series ---
FRED_SERIES = {
    "spread_2s10s":   "T10Y2Y",        # 2s10s yield curve spread
    "spread_10y3m":   "T10Y3M",        # 10Y-3M spread (better recession predictor)
    "hy_oas":         "BAMLH0A0HYM2",  # HY option-adjusted spread
    "ig_oas":         "BAMLC0A0CM",    # IG option-adjusted spread
    "breakeven_10y":  "T10YIE",        # 10Y breakeven inflation
    "cpi":            "CPIAUCSL",      # CPI All Items (monthly → ffilled)
    "dgs2":           "DGS2",          # 2Y Treasury yield
    "dgs10":          "DGS10",         # 10Y Treasury yield
    "fedfunds":       "FEDFUNDS",      # Fed Funds rate (monthly → ffilled)
}

# --- Backtest Parameters ---
BACKTEST_START  = "2005-01-01"
REBALANCE_FREQ  = "ME"

LOOKBACK_SIGNAL = 252
LOOKBACK_VOL    = 63
MOMENTUM_WINDOW = 252
MOMENTUM_SKIP   = 21

# --- Allocation Limits ---
MAX_CREDIT_ALLOC = 0.65   # max credit bucket allocation (risk-on)
MAX_TIP_ALLOC    = 0.30   # max TIP allocation
MAX_ALT_ALLOC    = 0.12   # max gold/hedge allocation (activates in stress)
MAX_CASH_ALLOC   = 0.40   # max BIL (T-bill) allocation (activates when curve inverted)
SIGNAL_BLEND     = 0.00   # 0=pure inv-vol, 1=pure signal weights

# --- Volatility Targeting ---
VOL_TARGET   = 0.07
MAX_LEVERAGE = 1.5
VOL_LOOKBACK = 21

# --- VIX Thresholds ---
VIX_RISK_OFF = 25.0
VIX_RISK_ON  = 15.0

# --- Composite Signal Weights ---
W_DURATION_2S10S = 0.35
W_DURATION_10Y3M = 0.35
W_DURATION_FED   = 0.30
W_CREDIT_HYOAS   = 0.60
W_CREDIT_VIX     = 0.40
W_INFLATION_BEI  = 0.50
W_INFLATION_CPI  = 0.50

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "cache")
