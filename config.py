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
    "spread_10y3m":   "T10Y3M",          # 10Y-3M spread (better recession predictor)
    "hy_oas":         "BAMLH0A0HYM2",    # HY option-adjusted spread
    "ig_oas":         "BAMLC0A0CM",      # IG option-adjusted spread
    "breakeven_10y":  "T10YIE",          # 10Y breakeven inflation
    "cpi":            "CPIAUCSL",        # CPI All Items (monthly → ffilled)
    "dgs2":           "DGS2",            # 2Y Treasury yield
    "dgs10":          "DGS10",           # 10Y Treasury yield
    "fedfunds":       "FEDFUNDS",        # Fed Funds rate (monthly → ffilled)
}

# --- Backtest Parameters ---
BACKTEST_START   = "2005-01-01"          # HYG/LQD have data from ~2002-2003
REBALANCE_FREQ   = "ME"                  # Month-end rebalancing

LOOKBACK_SIGNAL  = 252                   # Trading days for z-score (≈12m)
LOOKBACK_VOL     = 63                    # Trading days for volatility (≈3m)
MOMENTUM_WINDOW  = 252                   # 12-month momentum
MOMENTUM_SKIP    = 21                    # Skip last month (≈1m)

# Maximum credit allocation (LQD+HYG combined)
MAX_CREDIT_ALLOC  = 0.40
# Maximum TIP allocation
MAX_TIP_ALLOC     = 0.35
# Blend ratio: 1.0 = pure signal weights, 0.0 = pure inverse-vol weights
SIGNAL_BLEND      = 0.50
# LQD share within the credit bucket (remainder goes to HYG)
CREDIT_LQD_SPLIT  = 0.55

# --- Volatility Targeting ---
VOL_TARGET   = 0.06   # annualised target vol (6% → ~5% return at Sharpe ~0.83)
MAX_LEVERAGE = 1.5    # cap on vol-scaling leverage
VOL_LOOKBACK = 21     # days of realised vol used to compute scaling factor

# --- VIX Thresholds ---
VIX_RISK_OFF = 25.0   # VIX above this → cap credit exposure to 10 %
VIX_RISK_ON  = 15.0   # VIX below this → full credit allowed

# --- Composite Signal Weights (all Optuna-tunable) ---
W_DURATION_2S10S = 0.35   # weight of 2s10s in duration composite
W_DURATION_10Y3M = 0.35   # weight of 10Y-3M in duration composite
W_DURATION_FED   = 0.30   # weight of fed rate direction in duration composite
W_CREDIT_HYOAS   = 0.60   # weight of HY OAS in credit composite
W_CREDIT_VIX     = 0.40   # weight of VIX regime in credit composite
W_INFLATION_BEI  = 0.50   # weight of breakeven inflation
W_INFLATION_CPI  = 0.50   # weight of realised CPI momentum

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "cache")
