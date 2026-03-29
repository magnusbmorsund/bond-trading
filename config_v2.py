import os

# --- API Keys ---
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# --- Universe ---
DURATION_ETFS     = ["TLT", "IEF", "SHY"]
INFLATION_ETFS    = ["TIP", "VTIP"]          # v2: short-duration TIPS added alongside TIP
INFLATION_ETF     = "TIP"                    # kept for benchmark compat
CREDIT_ETFS       = ["LQD", "HYG", "ANGL", "SJNK", "BKLN", "EMB", "PFF"]
HEDGE_ETFS        = ["GLD", "SLV", "PDBC", "DBA"]   # v2: SLV added
REAL_ASSET_ETFS   = ["VNQ"]                  # v2: REIT satellite
EQUITY_ETFS       = ["SPY"]                  # v2: equity satellite

ETF_UNIVERSE = (
    DURATION_ETFS
    + INFLATION_ETFS
    + CREDIT_ETFS
    + HEDGE_ETFS
    + REAL_ASSET_ETFS
    + EQUITY_ETFS
)

# --- FRED Series ---
FRED_SERIES = {
    # Yield curve
    "spread_2s10s":   "T10Y2Y",
    "spread_10y3m":   "T10Y3M",
    # Credit spreads
    "hy_oas":         "BAMLH0A0HYM2",
    "ig_oas":         "BAMLC0A0CM",
    # Inflation
    "breakeven_10y":  "T10YIE",
    "cpi":            "CPIAUCSL",
    # Rates
    "dgs2":           "DGS2",
    "dgs10":          "DGS10",
    "fedfunds":       "FEDFUNDS",
    # Real yield
    "real_yield_10y": "DFII10",
    # Labor
    "unemployment":   "UNRATE",
    # Fed balance sheet
    "fed_assets":     "WALCL",
    # Growth proxies
    "indpro":         "INDPRO",
    # Financial stress
    "ted_spread":     "TEDRATE",
    # v2 additions
    "usd_index":      "DTWEXBGS",   # Nominal broad trade-weighted USD (weekly)
    "ism_mfg":        "GAFDFSA066MSFRBPHI",  # Philly Fed Future General Activity — diffusion index, ISM proxy
}

# --- Backtest Parameters ---
BACKTEST_START  = "2003-01-01"
REBALANCE_FREQ  = "ME"

LOOKBACK_SIGNAL = 252
LOOKBACK_VOL    = 63
MOMENTUM_WINDOW = 252
MOMENTUM_SKIP   = 21

# --- Allocation Limits ---
MAX_CREDIT_ALLOC      = 0.50
MAX_TIP_ALLOC         = 0.15
MAX_ALT_ALLOC         = 0.40
MAX_EQUITY_ALLOC      = 0.20   # v2: SPY satellite cap (fraction of remaining after commodities)
MAX_REALESTATE_ALLOC  = 0.08   # v2: VNQ cap (fraction of remaining)
SIGNAL_BLEND          = 0.30

# --- Drawdown Control Overlay ---
DD_THRESHOLD = -0.05
DD_SCALE     = 0.00

# --- Per-Position Trailing Stops ---
TRAILING_STOP_PCT    = 0.04
TRAILING_STOP_WINDOW = 21

# --- Volatility Targeting ---
VOL_TARGET   = 0.08
MAX_LEVERAGE = 1.50
VOL_LOOKBACK = 21

# --- VIX Thresholds ---
VIX_RISK_OFF = 25.0
VIX_RISK_ON  = 15.0

# --- Composite Signal Weights ---
# Duration composite
W_DURATION_2S10S   = 0.20
W_DURATION_10Y3M   = 0.20
W_DURATION_FED     = 0.15
W_DURATION_REALYLD = 0.25
W_DURATION_LABOR   = 0.10
W_DURATION_ISM     = 0.10   # v2: Philly Fed Future Activity (diffusion, leading) — true ISM PMI proxy

# Credit composite
W_CREDIT_HYOAS  = 0.35
W_CREDIT_IGMOM  = 0.15
W_CREDIT_VIX    = 0.20
W_CREDIT_FEDQT  = 0.15
W_CREDIT_TED    = 0.15

# Inflation composite
W_INFLATION_BEI = 0.50
W_INFLATION_CPI = 0.50

# v2: USD impact on commodity budget
# Rising USD dampens commodity allocation (commodities priced in USD).
# Applied as a multiplicative damper: budget *= (1 + W_COMMODITY_USD * tanh(-usd_z * 0.5))
# Range: [1 - W_COMMODITY_USD, 1 + W_COMMODITY_USD] of the base budget.
W_COMMODITY_USD = 0.25

# v2: VTIP blend — shifts allocation toward VTIP when duration_z is negative (rising rates).
# vtip_fraction = clip(0.5 - 0.5 * tanh(duration_z * VTIP_DURATION_SCALE), 0, 1)
# At duration_z = -2: ~90% VTIP. At duration_z = +2: ~10% VTIP.
VTIP_DURATION_SCALE = 0.50

# --- Numeric stability thresholds ---
MIN_ZSCORE_CLIP      = 1e-6
MIN_WEIGHT_THRESHOLD = 1e-4
MIN_VOL_CLIP         = 0.01
INV_VOL_CLIP         = 0.001
TANH_COMM_SCALE      = 0.8

# --- Data quality ---
PRICE_SPIKE_THRESHOLD = 0.15

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "cache")
LOG_DIR  = os.path.join(os.path.dirname(__file__), "logs")
