import os

FRED_API_KEY = os.getenv("FRED_API_KEY", "")

DURATION_ETFS        = ["EDV", "TLT", "IEF", "JPST", "SHY"]
INFLATION_ETFS       = ["TIP", "VTIP"]
INFLATION_ETF        = "TIP"
CREDIT_ETFS          = ["LQD", "HYG", "EMB"]
HEDGE_ETFS           = ["GLD", "SLV", "PDBC", "DBA"]
REAL_ASSET_ETFS      = ["VNQ"]
EQUITY_ETFS          = ["MTUM", "SPY"]
MANAGED_FUTURES_ETFS = []

ETF_UNIVERSE = (
    DURATION_ETFS + INFLATION_ETFS + CREDIT_ETFS
    + HEDGE_ETFS + REAL_ASSET_ETFS + EQUITY_ETFS + MANAGED_FUTURES_ETFS
)

# ETFs that get a trailing stop in Nordnet: trending/crash-prone assets only.
# Duration, TIPS, IG credit excluded — mean-reverting, macro signal handles exit.
TRAILING_STOP_ETFS = HEDGE_ETFS + ["HYG"] + REAL_ASSET_ETFS + EQUITY_ETFS + MANAGED_FUTURES_ETFS

FRED_SERIES = {
    "spread_2s10s":   "T10Y2Y",
    "spread_10y3m":   "T10Y3M",
    "hy_oas":         "BAMLH0A0HYM2",
    "ig_oas":         "BAMLC0A0CM",
    "breakeven_10y":  "T10YIE",
    "cpi":            "CPIAUCSL",
    "dgs2":           "DGS2",
    "dgs10":          "DGS10",
    "fedfunds":       "FEDFUNDS",
    "real_yield_10y": "DFII10",
    "unemployment":   "UNRATE",
    "fed_assets":     "WALCL",
    "indpro":         "INDPRO",
    "ted_spread":     "TEDRATE",
    "usd_index":      "DTWEXBGS",
    "ism_mfg":        "GAFDFSA066MSFRBPHI",
}

BACKTEST_START  = "2003-01-01"
REBALANCE_FREQ  = "ME"

LOOKBACK_SIGNAL = 252
LOOKBACK_VOL    = 63
MOMENTUM_WINDOW = 252
MOMENTUM_SKIP   = 21

MAX_CREDIT_ALLOC         = 0.50
MAX_TIP_ALLOC            = 0.15
MAX_ALT_ALLOC            = 0.40
MAX_EQUITY_ALLOC         = 0.20
MAX_REALESTATE_ALLOC     = 0.08
MAX_MANAGED_FUTURES_ALLOC = 0.15
SIGNAL_BLEND             = 0.30

EDV_DURATION_SCORE = 2.5

DD_THRESHOLD = -0.05
DD_SCALE     = 0.00

TRAILING_STOP_PCT    = 0.04
TRAILING_STOP_WINDOW = 21

VOL_TARGET   = 0.08
MAX_LEVERAGE = 1.0
VOL_LOOKBACK = 21

VIX_RISK_OFF = 25.0
VIX_RISK_ON  = 15.0

W_DURATION_2S10S   = 0.25
W_DURATION_10Y3M   = 0.25
W_DURATION_FED     = 0.20
W_DURATION_REALYLD = 0.30

W_CREDIT_HYOAS     = 0.25
W_CREDIT_IMPULSE   = 0.15
W_CREDIT_IGMOM     = 0.10
W_CREDIT_VIX       = 0.15
W_CREDIT_VIX_TS    = 0.10
W_CREDIT_FEDQT     = 0.15
W_CREDIT_TED       = 0.10

W_GROWTH_ISM       = 0.40
W_GROWTH_INDPRO    = 0.30
W_GROWTH_LABOR     = 0.30

W_INFLATION_BEI = 0.50
W_INFLATION_CPI = 0.50

W_COMMODITY_USD     = 0.25
VTIP_DURATION_SCALE = 0.50
MF_SIGNAL_SCALE     = 0.50

MIN_ZSCORE_CLIP      = 1e-6
MIN_WEIGHT_THRESHOLD = 1e-4
MIN_VOL_CLIP         = 0.01
INV_VOL_CLIP         = 0.001
TANH_COMM_SCALE      = 0.8

PRICE_SPIKE_THRESHOLD = 0.15

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cache")
LOG_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")

# --- Optimisation search space ---
PARAM_SPACE = {
    "LOOKBACK_SIGNAL":  ("int",    84, 504, 21),
    "LOOKBACK_VOL":     ("int",    21, 126, 21),
    "MOMENTUM_WINDOW":  ("int",   126, 504, 21),
    "MOMENTUM_SKIP":    ("int",     0,  42,  5),
    "MAX_CREDIT_ALLOC": ("float", 0.30, 0.80, 0.05),
    "MAX_TIP_ALLOC":    ("float", 0.00, 0.30, 0.05),
    "SIGNAL_BLEND":     ("float", 0.00, 1.00, 0.10),
    "VOL_TARGET":       ("float", 0.05, 0.15, 0.01),
    "VIX_RISK_OFF":     ("float", 18.0, 40.0, 1.0),
    "VIX_RISK_ON":      ("float", 10.0, 22.0, 1.0),
    "DD_THRESHOLD":     ("float", -0.15, -0.02, 0.01),
    "DD_SCALE":         ("float",  0.00,  0.50, 0.05),
    "TRAILING_STOP_PCT":    ("float", 0.03, 0.15, 0.01),
    "TRAILING_STOP_WINDOW": ("int",   21,  126,  21),
    "MAX_ALT_ALLOC":      ("float", 0.20, 0.60, 0.05),
    "W_DURATION_2S10S":   ("float", 0.05, 0.40, 0.05),
    "W_DURATION_10Y3M":   ("float", 0.05, 0.40, 0.05),
    "W_DURATION_FED":     ("float", 0.05, 0.30, 0.05),
    "W_DURATION_REALYLD": ("float", 0.10, 0.50, 0.05),
    "W_DURATION_LABOR":   ("float", 0.00, 0.25, 0.05),
    "W_DURATION_ISM":     ("float", 0.00, 0.25, 0.05),
    "W_CREDIT_HYOAS":     ("float", 0.15, 0.60, 0.05),
    "W_CREDIT_IGMOM":     ("float", 0.05, 0.35, 0.05),
    "W_CREDIT_VIX":       ("float", 0.10, 0.50, 0.05),
    "W_CREDIT_FEDQT":     ("float", 0.05, 0.35, 0.05),
    "W_CREDIT_TED":       ("float", 0.05, 0.35, 0.05),
    "W_INFLATION_BEI":    ("float", 0.20, 0.80, 0.10),
    "W_INFLATION_CPI":    ("float", 0.20, 0.80, 0.10),
    "MAX_EQUITY_ALLOC":           ("float", 0.00, 0.30, 0.05),
    "MAX_REALESTATE_ALLOC":       ("float", 0.00, 0.15, 0.05),
    "MAX_MANAGED_FUTURES_ALLOC":  ("float", 0.00, 0.20, 0.05),
    "W_COMMODITY_USD":            ("float", 0.00, 0.50, 0.05),
    "VTIP_DURATION_SCALE":        ("float", 0.10, 1.50, 0.10),
    "EDV_DURATION_SCORE":         ("float", 1.50, 4.00, 0.25),
    "MF_SIGNAL_SCALE":            ("float", 0.10, 1.00, 0.10),
    "W_CREDIT_IMPULSE":           ("float", 0.05, 0.30, 0.05),
    "W_CREDIT_VIX_TS":            ("float", 0.00, 0.25, 0.05),
    "W_GROWTH_ISM":               ("float", 0.10, 0.60, 0.10),
    "W_GROWTH_INDPRO":            ("float", 0.10, 0.60, 0.10),
    "W_GROWTH_LABOR":             ("float", 0.10, 0.60, 0.10),
}
