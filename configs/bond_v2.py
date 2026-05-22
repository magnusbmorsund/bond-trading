from configs.bond_base import *  # shared defaults: windows, vol target, DD, stops, numeric stability

# --- Universe ---
DURATION_ETFS     = ["TLT", "IEF", "SHY"]
INFLATION_ETFS    = ["TIP", "VTIP"]          # v2: short-duration TIPS added alongside TIP
INFLATION_ETF     = "TIP"                    # kept for benchmark compat
CREDIT_ETFS       = ["LQD", "HYG", "EMB", "PFF"]
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

TRAILING_STOP_ETFS = HEDGE_ETFS + ["HYG"] + REAL_ASSET_ETFS + EQUITY_ETFS

# --- FRED Series ---
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
    # v2 additions
    "usd_index":      "DTWEXBGS",
    "ism_mfg":        "GAFDFSA066MSFRBPHI",
}

# --- v2-specific allocation limits ---
MAX_EQUITY_ALLOC      = 0.20
MAX_REALESTATE_ALLOC  = 0.08

# --- Composite Signal Weights ---
W_DURATION_2S10S   = 0.20
W_DURATION_10Y3M   = 0.20
W_DURATION_FED     = 0.15
W_DURATION_REALYLD = 0.25
W_DURATION_LABOR   = 0.10
W_DURATION_ISM     = 0.10   # v2: Philly Fed Future Activity — true ISM PMI proxy

W_CREDIT_HYOAS  = 0.35
W_CREDIT_IGMOM  = 0.15
W_CREDIT_VIX    = 0.20
W_CREDIT_FEDQT  = 0.15
W_CREDIT_TED    = 0.15

W_INFLATION_BEI = 0.50
W_INFLATION_CPI = 0.50

# v2: USD dampens commodity budget; rising USD ↓ commodity allocation
W_COMMODITY_USD = 0.25
# v2: VTIP blend — shifts toward VTIP when duration_z is negative (rising rates)
VTIP_DURATION_SCALE = 0.50

# --- Optimisation search space ---
PARAM_SPACE = {
    **BASE_PARAM_SPACE,
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
    "MAX_EQUITY_ALLOC":      ("float", 0.00, 0.30, 0.05),
    "MAX_REALESTATE_ALLOC":  ("float", 0.00, 0.15, 0.05),
    "W_COMMODITY_USD":       ("float", 0.00, 0.50, 0.05),
    "VTIP_DURATION_SCALE":   ("float", 0.10, 1.50, 0.10),
}
