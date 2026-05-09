import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.ERROR)

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

THEMATIC = ["LIT", "COPX", "XOP", "ICLN", "TAN", "BOTZ", "URNM", "NLR", "REMX", "MOO", "FUELC", "HYDR", "PLUG", "BE", "CPER"]
BROAD_SECTORS = ["XLE", "XLK", "XLV", "XLF", "XLI", "XLY", "XLP", "XLU", "XLB", "VNQ", "SMH", "IBB"]
COMMODITIES = ["GLD", "SLV", "PDBC", "DBA", "UNG", "USO"]

ALL_TICKERS = THEMATIC + BROAD_SECTORS + COMMODITIES

START_DATE = "2010-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
WINDOW = 63  # 3-month rolling window for trough/peak detection
MIN_CYCLE_GAIN = 0.20  # 20% minimum gain for a valid cycle

def classify_cycle(gain_pct, duration_days):
    if duration_days > 540 or gain_pct > 150:
        return "Supercycle"
    elif duration_days >= 180 or gain_pct >= 60:
        return "Cyclical"
    else:
        return "Tactical"

def find_cycles(prices):
    """Find all major rally cycles using rolling window trough/peak detection."""
    cycles = []
    n = len(prices)
    if n < WINDOW * 2:
        return cycles

    # Compute rolling min (troughs) and rolling max (peaks)
    roll_min = prices.rolling(window=WINDOW, center=True).min()
    roll_max = prices.rolling(window=WINDOW, center=True).max()

    # A point is a local trough if it equals the rolling min
    is_trough = (prices == roll_min)
    # A point is a local peak if it equals the rolling max
    is_peak = (prices == roll_max)

    trough_dates = prices.index[is_trough].tolist()
    peak_dates = prices.index[is_peak].tolist()

    # Walk through: find trough -> next peak after trough
    used_peaks = set()
    for t_date in trough_dates:
        t_price = prices.loc[t_date]
        # Find peaks after this trough
        candidate_peaks = [p for p in peak_dates if p > t_date and p not in used_peaks]
        if not candidate_peaks:
            continue
        # Find the highest peak after this trough (greedy: take max gain)
        best_peak = None
        best_gain = 0
        for p_date in candidate_peaks:
            p_price = prices.loc[p_date]
            gain = (p_price - t_price) / t_price * 100
            if gain > best_gain:
                best_gain = gain
                best_peak = p_date
        if best_peak is not None and best_gain >= MIN_CYCLE_GAIN * 100:
            duration = (best_peak - t_date).days
            classification = classify_cycle(best_gain, duration)
            cycles.append({
                "trough_date": t_date,
                "peak_date": best_peak,
                "trough_price": t_price,
                "peak_price": prices.loc[best_peak],
                "gain_pct": best_gain,
                "duration_days": duration,
                "classification": classification,
            })
            used_peaks.add(best_peak)

    # Remove overlapping/dominated cycles: keep non-overlapping by greedy gain
    # Sort by gain descending
    cycles.sort(key=lambda x: -x["gain_pct"])
    non_overlapping = []
    occupied = set()
    for c in cycles:
        # Check if trough or peak already used
        t = c["trough_date"]
        p = c["peak_date"]
        overlap = False
        for (ot, op) in occupied:
            # Overlap if intervals intersect
            if t <= op and p >= ot:
                overlap = True
                break
        if not overlap:
            non_overlapping.append(c)
            occupied.add((t, p))

    non_overlapping.sort(key=lambda x: x["trough_date"])
    return non_overlapping

def get_current_status(ticker, prices, cycles):
    """Compute current status metrics."""
    today = prices.index[-1]
    current_price = prices.iloc[-1]

    def ret(days):
        target = today - pd.Timedelta(days=days)
        # Find nearest available date
        idx = prices.index.searchsorted(target)
        if idx >= len(prices):
            return np.nan
        past_price = prices.iloc[idx]
        return (current_price - past_price) / past_price * 100

    r1m = ret(30)
    r3m = ret(91)
    r6m = ret(182)
    r12m = ret(365)
    r24m = ret(730)

    # 52-week high
    w52 = prices.last("252D") if len(prices) > 252 else prices
    high_52w = w52.max()
    pct_below_52w_high = (current_price - high_52w) / high_52w * 100

    # Momentum direction
    if not np.isnan(r3m) and not np.isnan(r6m):
        momentum = "↑ accelerating" if r3m > r6m else "↓ decelerating"
    else:
        momentum = "N/A"

    # Current cycle: trough of most recent cycle up to today (or ongoing)
    # Find the last trough (from all cycles or raw troughs)
    current_trough_date = None
    current_gain = np.nan
    days_in_cycle = np.nan

    if cycles:
        # Last cycle's trough
        last_cycle = cycles[-1]
        current_trough_date = last_cycle["trough_date"]
        trough_price = last_cycle["trough_price"]
        current_gain = (current_price - trough_price) / trough_price * 100
        days_in_cycle = (today - current_trough_date).days

    return {
        "ticker": ticker,
        "current_price": current_price,
        "pct_below_52w_high": pct_below_52w_high,
        "r1m": r1m,
        "r3m": r3m,
        "r6m": r6m,
        "r12m": r12m,
        "r24m": r24m,
        "momentum": momentum,
        "current_trough_date": current_trough_date,
        "current_gain_from_trough": current_gain,
        "days_in_current_cycle": days_in_cycle,
        "last_cycle_type": cycles[-1]["classification"] if cycles else "N/A",
    }

# ──────────────────────────────────────────────
# DOWNLOAD DATA
# ──────────────────────────────────────────────
print("Downloading ETF data...")
raw = yf.download(ALL_TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
if isinstance(raw.columns, pd.MultiIndex):
    prices_all = raw["Close"]
else:
    prices_all = raw[["Close"]]

print(f"Downloaded {len(prices_all.columns)} tickers, {len(prices_all)} trading days\n")

# ──────────────────────────────────────────────
# PROCESS EACH ETF
# ──────────────────────────────────────────────
valid_tickers = []
all_cycles = []
status_rows = []

for ticker in ALL_TICKERS:
    if ticker not in prices_all.columns:
        print(f"  SKIP {ticker}: not in download")
        continue
    series = prices_all[ticker].dropna()
    # Require >= 2 years of data (504 trading days)
    if len(series) < 504:
        print(f"  SKIP {ticker}: only {len(series)} days of data")
        continue
    valid_tickers.append(ticker)
    cycles = find_cycles(series)
    for c in cycles:
        c["ticker"] = ticker
    all_cycles.extend(cycles)
    status = get_current_status(ticker, series, cycles)
    status_rows.append(status)

print(f"\nValid tickers analyzed: {len(valid_tickers)}")
print(f"Total cycles found: {len(all_cycles)}\n")

# ──────────────────────────────────────────────
# SUMMARY TABLE sorted by 12M return
# ──────────────────────────────────────────────
df_status = pd.DataFrame(status_rows)
df_status = df_status.sort_values("r12m", ascending=False)

def fmt_pct(v):
    if pd.isna(v):
        return "   N/A"
    return f"{v:+7.1f}%"

def fmt_days(v):
    if pd.isna(v):
        return "  N/A"
    return f"{int(v):5d}d"

def fmt_gain(v):
    if pd.isna(v):
        return "   N/A"
    return f"{v:+7.1f}%"

print("=" * 130)
print("SUMMARY TABLE — sorted by 12-month return")
print("=" * 130)
header = (
    f"{'Ticker':<6}  {'1M':>8}  {'3M':>8}  {'6M':>8}  {'12M':>8}  {'24M':>8}  "
    f"{'52W vs Hi':>10}  {'LastCycleType':<12}  {'CurGain':>8}  {'DaysInCyc':>10}  {'Momentum':<18}"
)
print(header)
print("-" * 130)
for _, row in df_status.iterrows():
    line = (
        f"{row['ticker']:<6}  "
        f"{fmt_pct(row['r1m']):>8}  "
        f"{fmt_pct(row['r3m']):>8}  "
        f"{fmt_pct(row['r6m']):>8}  "
        f"{fmt_pct(row['r12m']):>8}  "
        f"{fmt_pct(row['r24m']):>8}  "
        f"{fmt_pct(row['pct_below_52w_high']):>10}  "
        f"{row['last_cycle_type']:<12}  "
        f"{fmt_gain(row['current_gain_from_trough']):>8}  "
        f"{fmt_days(row['days_in_current_cycle']):>10}  "
        f"{row['momentum']:<18}"
    )
    print(line)
print("=" * 130)

# ──────────────────────────────────────────────
# TOP 10 HISTORICAL SUPERCYCLES / BIGGEST CYCLES
# ──────────────────────────────────────────────
df_cycles = pd.DataFrame(all_cycles)
df_cycles = df_cycles.sort_values("gain_pct", ascending=False)

print("\n")
print("=" * 100)
print("TOP 20 HISTORICAL CYCLES (by gain %) across all ETFs")
print("=" * 100)
header2 = (
    f"{'Rank':<5}  {'Ticker':<6}  {'Trough Date':<12}  {'Peak Date':<12}  "
    f"{'Gain %':>8}  {'Duration':>10}  {'Classification':<14}"
)
print(header2)
print("-" * 100)
for i, (_, row) in enumerate(df_cycles.head(20).iterrows(), 1):
    line = (
        f"{i:<5}  "
        f"{row['ticker']:<6}  "
        f"{str(row['trough_date'].date()):<12}  "
        f"{str(row['peak_date'].date()):<12}  "
        f"{row['gain_pct']:>8.1f}%  "
        f"{row['duration_days']:>8}d  "
        f"{row['classification']:<14}"
    )
    print(line)
print("=" * 100)

# ──────────────────────────────────────────────
# SUPERCYCLE-PRONE ETFs (ETFs that have had at least 1 supercycle)
# ──────────────────────────────────────────────
print("\n")
print("=" * 80)
print("SUPERCYCLE-PRONE ETFs (had at least one Supercycle)")
print("=" * 80)
supercycle_df = df_cycles[df_cycles["classification"] == "Supercycle"]
supercycle_counts = supercycle_df.groupby("ticker").agg(
    num_supercycles=("gain_pct", "count"),
    max_gain=("gain_pct", "max"),
    avg_duration=("duration_days", "mean"),
).sort_values("max_gain", ascending=False)
print(supercycle_counts.to_string())

# ──────────────────────────────────────────────
# CYCLE COUNT BREAKDOWN per ETF
# ──────────────────────────────────────────────
print("\n")
print("=" * 80)
print("CYCLE TYPE BREAKDOWN per ETF")
print("=" * 80)
breakdown = df_cycles.groupby(["ticker", "classification"]).size().unstack(fill_value=0)
for col in ["Tactical", "Cyclical", "Supercycle"]:
    if col not in breakdown.columns:
        breakdown[col] = 0
breakdown = breakdown[["Tactical", "Cyclical", "Supercycle"]]
breakdown["Total"] = breakdown.sum(axis=1)
breakdown = breakdown.sort_values("Supercycle", ascending=False)
print(breakdown.to_string())

# ──────────────────────────────────────────────
# CURRENT CYCLE CONTEXT: Who is early/mid/late?
# ──────────────────────────────────────────────
print("\n")
print("=" * 90)
print("CURRENT CYCLE CONTEXT — ETFs with active cycles, sorted by current gain from trough")
print("=" * 90)
df_active = df_status[df_status["current_gain_from_trough"].notna()].copy()
df_active = df_active.sort_values("current_gain_from_trough", ascending=False)

header3 = (
    f"{'Ticker':<6}  {'Trough Date':<13}  {'DaysIn':>7}  {'CurGain':>9}  "
    f"{'LastCycType':<13}  {'12M':>8}  {'Momentum':<18}"
)
print(header3)
print("-" * 90)
for _, row in df_active.iterrows():
    trough_str = str(row["current_trough_date"].date()) if row["current_trough_date"] is not None else "N/A"
    line = (
        f"{row['ticker']:<6}  "
        f"{trough_str:<13}  "
        f"{fmt_days(row['days_in_current_cycle']):>7}  "
        f"{fmt_gain(row['current_gain_from_trough']):>9}  "
        f"{row['last_cycle_type']:<13}  "
        f"{fmt_pct(row['r12m']):>8}  "
        f"{row['momentum']:<18}"
    )
    print(line)
print("=" * 90)

print("\nAnalysis complete.")
