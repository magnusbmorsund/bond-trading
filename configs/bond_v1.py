from configs.bond_base import *  # shared defaults: windows, vol target, DD, stops, numeric stability

# --- Universe ---
# Duration (bonds as the defensive "cash pool")
DURATION_ETFS = ["TLT", "IEF", "SHY"]
# Inflation-linked
INFLATION_ETF = "TIP"
# Credit / spread (IG → short-duration HY → floating → EM → preferred)
CREDIT_ETFS   = ["LQD", "HYG", "ANGL", "SJNK", "BKLN", "EMB", "PFF"]
# Commodity / materials basket — momentum-gated, inverse-vol weighted
# GLD:  monetary metal — real yields falling + inflation rising
# PDBC: diversified commodities (energy+metals+ag, incl. base metals) — growth/inflation cycles
# DBA:  agriculture — food inflation, weather, lowest GLD corr (0.16), low vol (13%)
HEDGE_ETFS    = ["GLD", "PDBC", "DBA"]

ETF_UNIVERSE  = DURATION_ETFS + [INFLATION_ETF] + CREDIT_ETFS + HEDGE_ETFS

# ETFs that get a trailing stop in Nordnet: trending/crash-prone assets only.
TRAILING_STOP_ETFS = HEDGE_ETFS + ["HYG"]

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
}

# --- Composite Signal Weights ---
W_DURATION_2S10S   = 0.20
W_DURATION_10Y3M   = 0.20
W_DURATION_FED     = 0.15
W_DURATION_REALYLD = 0.25
W_DURATION_LABOR   = 0.10
W_DURATION_ISM     = 0.10

W_CREDIT_HYOAS     = 0.35
W_CREDIT_IGMOM     = 0.15
W_CREDIT_VIX       = 0.20
W_CREDIT_FEDQT     = 0.15
W_CREDIT_TED       = 0.15

W_INFLATION_BEI    = 0.50
W_INFLATION_CPI    = 0.50

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
}
