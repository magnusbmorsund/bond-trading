import os

# --- API Keys ---
FRED_API_KEY = os.getenv("FRED_API_KEY", "08b619abe65ba421c8b44520629987ba")

# --- Universe ---
ETF_UNIVERSE = ["TLT", "IEF", "SHY", "TIP", "LQD", "HYG"]

# Characteristics used for signal mapping
DURATION_ETFS = ["TLT", "IEF", "SHY"]   # high → low duration
CREDIT_ETFS   = ["LQD", "HYG"]           # IG → HY
INFLATION_ETF = "TIP"

# --- FRED Series ---
FRED_SERIES = {
    "spread_2s10s":   "T10Y2Y",          # 2s10s yield curve spread
    "hy_oas":         "BAMLH0A0HYM2",    # HY option-adjusted spread
    "ig_oas":         "BAMLC0A0CM",      # IG option-adjusted spread
    "breakeven_10y":  "T10YIE",          # 10Y breakeven inflation
    "dgs2":           "DGS2",            # 2Y Treasury yield
    "dgs10":          "DGS10",           # 10Y Treasury yield
    "fedfunds":       "FEDFUNDS",        # Fed Funds rate
}

# --- Backtest Parameters ---
BACKTEST_START   = "2005-01-01"          # HYG/LQD have data from ~2002-2003
REBALANCE_FREQ   = "ME"                  # Month-end rebalancing

LOOKBACK_SIGNAL  = 252                   # Trading days for z-score (≈12m)
LOOKBACK_VOL     = 63                    # Trading days for volatility (≈3m)
MOMENTUM_WINDOW  = 252                   # 12-month momentum
MOMENTUM_SKIP    = 21                    # Skip last month (≈1m)

# Maximum credit allocation (LQD+HYG combined)
MAX_CREDIT_ALLOC = 0.40
# Maximum TIP allocation
MAX_TIP_ALLOC    = 0.35

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "cache")
