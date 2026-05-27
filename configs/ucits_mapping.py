"""
US ETF → UCITS ETF mapping for Nordnet live trading.

Strategy backtest, optimization, and data pipeline all use US tickers
(26-year history in Supabase). For live execution on Nordnet, retail
investors should trade the UCITS equivalents — this mapping translates
US tickers to the corresponding UCITS ticker + ISIN + exchange.

See memory: sector_v2e_ucits_mapping for full research notes.

Status legend:
  direct  — UCITS tracks identical underlying index as US ETF
  proxy   — UCITS tracks similar but not identical index (style drift)
  none    — no acceptable UCITS substitute; do not trade
"""

UCITS_MAP = {
    # ── S&P Sector Core (SPDR UCITS — direct 1:1 replicas, Irish, TER 0.15%) ──
    "XLE": {"ticker": "ZPDE", "isin": "IE00BWBXM385", "exchange": "Xetra", "name": "SPDR S&P US Energy Select Sector",      "status": "direct"},
    "XLK": {"ticker": "ZPDT", "isin": "IE00BWBXM492", "exchange": "Xetra", "name": "SPDR S&P US Technology Select Sector",  "status": "direct"},
    "XLV": {"ticker": "ZPDH", "isin": "IE00BWBXM276", "exchange": "Xetra", "name": "SPDR S&P US Health Care Select Sector", "status": "direct"},
    "XLF": {"ticker": "ZPDF", "isin": "IE00BWBXM161", "exchange": "Xetra", "name": "SPDR S&P US Financials Select Sector",  "status": "direct"},
    "XLI": {"ticker": "ZPDI", "isin": "IE00BWBXM385", "exchange": "Xetra", "name": "SPDR S&P US Industrials Select Sector", "status": "direct"},
    "XLY": {"ticker": "ZPDD", "isin": "IE00BWBXM054", "exchange": "Xetra", "name": "SPDR S&P US Cons Discretionary Sector", "status": "direct"},
    "XLP": {"ticker": "ZPDS", "isin": "IE00BWBXMB69", "exchange": "Xetra", "name": "SPDR S&P US Cons Staples Select Sector","status": "direct"},
    "XLU": {"ticker": "ZPDU", "isin": "IE00BWBXMC76", "exchange": "Xetra", "name": "SPDR S&P US Utilities Select Sector",   "status": "direct"},
    "XLB": {"ticker": "ZPDM", "isin": "IE00BWBXM715", "exchange": "Xetra", "name": "SPDR S&P US Materials Select Sector",   "status": "direct"},

    # ── Real Estate ──────────────────────────────────────────────────────────
    "VNQ": {"ticker": "IQQ7", "isin": "IE00B1FZSF77", "exchange": "Xetra", "name": "iShares US Property Yield UCITS",       "status": "proxy"},

    # ── Tech / Compute / Innovation ──────────────────────────────────────────
    "SMH": {"ticker": "VVSM", "isin": "IE00BMC38736", "exchange": "Xetra", "name": "VanEck Semiconductor UCITS",            "status": "direct"},
    "ARKK":{"ticker": "ARKK", "isin": "IE000I7YL880", "exchange": "LSE",   "name": "ARK Innovation UCITS (ARK Europe)",     "status": "proxy"},
    "IGV": {"ticker": None,   "isin": None,           "exchange": None,    "name": "NO MATCH — drop from universe",         "status": "none"},

    # ── Metals & Miners ──────────────────────────────────────────────────────
    "GDX": {"ticker": "GDGB", "isin": "IE00BQQP9F84", "exchange": "Xetra", "name": "VanEck Gold Miners UCITS",              "status": "direct"},
    "GDXJ":{"ticker": "G2XJ", "isin": "IE00BQQP9G91", "exchange": "Xetra", "name": "VanEck Junior Gold Miners UCITS",       "status": "direct"},
    "SIL": {"ticker": "SILV", "isin": "IE000ZN0G541", "exchange": "LSE",   "name": "Global X Silver Miners UCITS",          "status": "direct"},
    "XME": {"ticker": "FAMAMW","isin":"IE000EE3Q489", "exchange": "Borsa Italiana", "name": "Fineco MSCI World Metals & Mining","status": "proxy"},
    "COPX":{"ticker": "COPG", "isin": "IE0003Z9E2Y3", "exchange": "Xetra", "name": "Global X Copper Miners UCITS",          "status": "direct"},
    "REMX":{"ticker": "REMX", "isin": "IE0002PG6CA6", "exchange": "LSE",   "name": "VanEck Rare Earth & Strategic Metals",  "status": "direct"},

    # ── Energy ───────────────────────────────────────────────────────────────
    "XOP": {"ticker": "IOGP", "isin": "IE00B6R51Z18", "exchange": "LSE",   "name": "iShares Oil & Gas E&P UCITS",           "status": "proxy"},
    "OIH": {"ticker": "OIHV", "isin": "IE000NXF88S1", "exchange": "Xetra", "name": "VanEck Oil Services UCITS",             "status": "direct"},

    # ── Green / Clean / Nuclear ──────────────────────────────────────────────
    "ICLN":{"ticker": "INRG", "isin": "IE00B1XNHC34", "exchange": "LSE",   "name": "iShares Global Clean Energy UCITS",     "status": "direct"},
    "URA": {"ticker": "URNU", "isin": "IE000NDWFGA5", "exchange": "LSE",   "name": "Global X Uranium UCITS",                "status": "direct"},

    # ── Defense ──────────────────────────────────────────────────────────────
    "ITA": {"ticker": "ASWC", "isin": "IE000OJ5TQP4", "exchange": "Xetra", "name": "HANetf Future of Defence UCITS",        "status": "proxy"},

    # ── Gold ─────────────────────────────────────────────────────────────────
    "GLD": {"ticker": "SGLN", "isin": "IE00B4ND3602", "exchange": "LSE",   "name": "iShares Physical Gold ETC",             "status": "direct"},

    # ── Biotech ──────────────────────────────────────────────────────────────
    "XBI": {"ticker": None,   "isin": None,           "exchange": None,    "name": "NO MATCH — drop from universe",         "status": "none"},
    "IBB": {"ticker": "BTEC", "isin": "IE00BYXG2H39", "exchange": "LSE",   "name": "iShares Nasdaq US Biotechnology UCITS", "status": "direct"},

    # ── China ────────────────────────────────────────────────────────────────
    "KWEB":{"ticker": "KWEB", "isin": "IE00BFXR7892", "exchange": "LSE",   "name": "KraneShares CSI China Internet UCITS",  "status": "direct"},

    # ── Bonds (iShares UCITS — direct, Irish, TER 0.07–0.50%) ────────────────
    "TLT": {"ticker": "DTLA", "isin": "IE00BSKRJZ44", "exchange": "LSE",   "name": "iShares $ Treasury 20+yr UCITS (Acc)",  "status": "direct"},
    "IEF": {"ticker": "IBTA", "isin": "IE00B3VWN518", "exchange": "LSE",   "name": "iShares $ Treasury 7-10yr UCITS (Acc)", "status": "direct"},
    "HYG": {"ticker": "IHYU", "isin": "IE00B4PY7Y77", "exchange": "LSE",   "name": "iShares $ High Yield Corp Bond UCITS",  "status": "direct"},

    # ── International Equity ─────────────────────────────────────────────────
    "EFA": {"ticker": "XUSE", "isin": "IE00BKBF6H24", "exchange": "Xetra", "name": "iShares MSCI World ex-USA UCITS",       "status": "proxy"},
    "EEM": {"ticker": "EIMI", "isin": "IE00BKM4GZ66", "exchange": "LSE",   "name": "iShares Core MSCI EM IMI UCITS",        "status": "proxy"},
    "EWJ": {"ticker": "SJPA", "isin": "IE00B53QDK08", "exchange": "LSE",   "name": "iShares Core MSCI Japan IMI UCITS",     "status": "proxy"},
    "EWZ": {"ticker": "IBZL", "isin": "IE00B0M63516", "exchange": "LSE",   "name": "iShares MSCI Brazil UCITS",             "status": "direct"},
    "INDA":{"ticker": "NDIA", "isin": "IE00BZCQB185", "exchange": "LSE",   "name": "iShares MSCI India UCITS (Acc)",        "status": "direct"},

    # ── Broad Commodities ────────────────────────────────────────────────────
    "PDBC":{"ticker": "CMOD", "isin": "IE00BD6FTQ80", "exchange": "LSE",   "name": "Invesco Bloomberg Commodity UCITS",     "status": "proxy"},

    # ── Cash equivalent ──────────────────────────────────────────────────────
    "SHY": {"ticker": "IBTS", "isin": "IE00B14X4S71", "exchange": "LSE",   "name": "iShares $ Treasury 1-3yr UCITS (Dist)", "status": "direct"},

    # ── Reference / S&P 500 ──────────────────────────────────────────────────
    "SPY": {"ticker": "CSPX", "isin": "IE00B5BMR087", "exchange": "LSE",   "name": "iShares Core S&P 500 UCITS (Acc)",      "status": "direct"},
}


def to_ucits(us_ticker: str) -> dict | None:
    """Return UCITS info for a US ticker, or None if no acceptable match."""
    info = UCITS_MAP.get(us_ticker)
    if info is None or info["status"] == "none":
        return None
    return info


def has_match(us_ticker: str) -> bool:
    """True if the US ticker has an acceptable UCITS equivalent."""
    return to_ucits(us_ticker) is not None
