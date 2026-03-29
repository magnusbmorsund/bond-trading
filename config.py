import os

# --- API Keys ---
FRED_API_KEY = os.getenv("FRED_API_KEY", "")   # set FRED_API_KEY env var before running

# --- Universe ---
# Duration (bonds as the defensive "cash pool")
DURATION_ETFS = ["TLT", "IEF", "SHY"]
# Inflation-linked
INFLATION_ETF = "TIP"
# Credit / spread (IG → short-duration HY → floating → EM → preferred)
CREDIT_ETFS   = ["LQD", "HYG", "ANGL", "SJNK", "BKLN", "EMB", "PFF"]
# Commodity / materials basket — momentum-gated, inverse-vol weighted
# GLD:  monetary metal — real yields falling + inflation rising
# PDBC: diversified commodities (energy+metals+ag, incl. base metals) — growth/inflation cycles, corr 0.27 to GLD
# DBA:  agriculture — food inflation, weather, lowest GLD corr (0.16), low vol (13%)
HEDGE_ETFS    = ["GLD", "PDBC", "DBA"]

ETF_UNIVERSE  = DURATION_ETFS + [INFLATION_ETF] + CREDIT_ETFS + HEDGE_ETFS

# --- FRED Series ---
FRED_SERIES = {
    # Yield curve
    "spread_2s10s":   "T10Y2Y",        # 2s10s yield curve spread
    "spread_10y3m":   "T10Y3M",        # 10Y-3M spread (better recession predictor)
    # Credit spreads
    "hy_oas":         "BAMLH0A0HYM2",  # HY option-adjusted spread
    "ig_oas":         "BAMLC0A0CM",    # IG option-adjusted spread
    # Inflation
    "breakeven_10y":  "T10YIE",        # 10Y breakeven inflation
    "cpi":            "CPIAUCSL",      # CPI All Items (monthly → ffilled)
    # Rates
    "dgs2":           "DGS2",          # 2Y Treasury yield
    "dgs10":          "DGS10",         # 10Y Treasury yield
    "fedfunds":       "FEDFUNDS",      # Fed Funds rate (monthly → ffilled)
    # Real yield — THE key 2022 signal; rising real yields crush all bonds
    "real_yield_10y": "DFII10",        # 10Y TIPS real yield (daily)
    # Labor market — rising unemployment → Fed cuts → bullish duration
    "unemployment":   "UNRATE",        # Unemployment rate (monthly → ffilled)
    # Fed balance sheet — QT (shrinking) headwinds, QE (growing) tailwinds
    "fed_assets":     "WALCL",         # Fed total assets (weekly → ffilled)
    # Industrial production — growth proxy (declining = bullish bonds)
    "indpro":         "INDPRO",        # Industrial Production Index (monthly → ffilled)
    # TED spread — T-bill to Eurodollar; financial stress indicator for credit
    "ted_spread":     "TEDRATE",       # TED spread (discontinued 2023 but historical ok)
}

# --- Backtest Parameters ---
BACKTEST_START  = "2005-01-01"
REBALANCE_FREQ  = "ME"

LOOKBACK_SIGNAL = 252
LOOKBACK_VOL    = 63
MOMENTUM_WINDOW = 252
MOMENTUM_SKIP   = 21

# --- Allocation Limits ---
MAX_CREDIT_ALLOC = 0.50   # max credit bucket allocation (risk-on)
MAX_TIP_ALLOC    = 0.15   # max TIP allocation
MAX_ALT_ALLOC    = 0.40   # max gold allocation — primary alpha source when trending
SIGNAL_BLEND     = 0.30   # 0=pure inv-vol, 1=pure signal weights

# --- Drawdown Control Overlay ---
# Scale down to DD_SCALE when portfolio is in drawdown AND momentum is negative.
# Re-enter immediately when short-term momentum turns positive (ride upswings).
DD_THRESHOLD = -0.05   # drawdown level that triggers scaling (e.g. -5%)
DD_SCALE     = 0.00    # 0 = full exit when in distress

# --- Per-Position Trailing Stops (commodity bucket) ---
# Exit a commodity ETF if its price falls more than TRAILING_STOP_PCT below
# its rolling TRAILING_STOP_WINDOW-day peak. Freed allocation parks in SHY.
# Applied daily — exits faster than monthly momentum rebalance.
TRAILING_STOP_PCT    = 0.04   # 4% trailing stop (tight — fast exit)
TRAILING_STOP_WINDOW = 21     # 21-day rolling peak

# --- Volatility Targeting ---
VOL_TARGET   = 0.08
MAX_LEVERAGE = 1.50
VOL_LOOKBACK = 21

# --- VIX Thresholds ---
VIX_RISK_OFF = 25.0
VIX_RISK_ON  = 15.0

# --- Composite Signal Weights ---
# Duration composite
W_DURATION_2S10S   = 0.20   # yield curve steepness
W_DURATION_10Y3M   = 0.20   # 10Y-3M spread (recession predictor)
W_DURATION_FED     = 0.15   # Fed rate direction
W_DURATION_REALYLD = 0.25   # real yield direction (rising = bearish duration)
W_DURATION_LABOR   = 0.10   # unemployment trend (rising = bullish duration)
W_DURATION_ISM     = 0.10   # Industrial production deceleration (falling IP = bullish bonds)
# Credit composite
W_CREDIT_HYOAS     = 0.35   # HY OAS level
W_CREDIT_IGMOM     = 0.15   # IG spread momentum (widening speed)
W_CREDIT_VIX       = 0.20   # VIX regime
W_CREDIT_FEDQT     = 0.15   # Fed QT/QE (shrinking balance sheet = headwind)
W_CREDIT_TED       = 0.15   # TED spread financial stress (high TED = bearish credit)
# Inflation composite
W_INFLATION_BEI    = 0.50   # breakeven inflation ROC
W_INFLATION_CPI    = 0.50   # CPI momentum

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "cache")
