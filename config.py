import os

# --- API Keys ---
FRED_API_KEY = os.getenv("FRED_API_KEY", "08b619abe65ba421c8b44520629987ba")

# --- Universe ---
# Expanded to include high-yield, floating-rate and EM instruments
ETF_UNIVERSE  = ["TLT", "IEF", "SHY", "TIP", "LQD", "HYG", "ANGL", "BKLN", "EMB"]

DURATION_ETFS = ["TLT", "IEF", "SHY"]                  # high → low duration
CREDIT_ETFS   = ["LQD", "HYG", "ANGL", "BKLN", "EMB"]  # all credit/spread instruments
INFLATION_ETF = "TIP"

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
BACKTEST_START  = "2005-01-01"   # ANGL/BKLN live 2011-2012; handled via NaN until sufficient history
REBALANCE_FREQ  = "ME"           # Month-end rebalancing

LOOKBACK_SIGNAL = 252            # Trading days for z-score (≈12m)
LOOKBACK_VOL    = 63             # Trading days for volatility (≈3m)
MOMENTUM_WINDOW = 252            # 12-month momentum lookback
MOMENTUM_SKIP   = 21             # Skip last month to avoid reversal bias

# --- Allocation Limits ---
MAX_CREDIT_ALLOC = 0.65          # Up to 65% in credit when fully risk-on
MAX_TIP_ALLOC    = 0.30          # Max TIP allocation
SIGNAL_BLEND     = 0.00          # 1.0=pure signal, 0.0=pure inverse-vol (optimiser will tune)

# --- Volatility Targeting ---
VOL_TARGET   = 0.07   # 7% annualised vol target → ~5-7% return at Sharpe 0.7-1.0
MAX_LEVERAGE = 1.5    # cap on vol-scaling leverage
VOL_LOOKBACK = 21     # days of realised vol for scaling factor

# --- VIX Thresholds ---
VIX_RISK_OFF = 25.0   # VIX above this → cap credit at 10%
VIX_RISK_ON  = 15.0   # VIX below this → full credit budget allowed

# --- Composite Signal Weights (Optuna-tunable) ---
W_DURATION_2S10S = 0.35
W_DURATION_10Y3M = 0.35
W_DURATION_FED   = 0.30
W_CREDIT_HYOAS   = 0.60
W_CREDIT_VIX     = 0.40
W_INFLATION_BEI  = 0.50
W_INFLATION_CPI  = 0.50

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "cache")
