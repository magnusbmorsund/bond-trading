# CLAUDE.md — Developer Guide for Claude Code

## Architecture

All parameters flow through `configs/bond_v1.py` (and the equivalent file for each strategy variant). Optuna patches config attributes at runtime via `setattr(config, k, v)` — this is why every module does `import configs.bond_v1 as config` and reads `config.X` at call time (never `from configs.bond_v1 import X` for any tunable parameter).

## Key Design Decisions

**Why trailing stops are in `backtest.py`, not `portfolio.py`**
`portfolio.py` builds monthly target weights from macro signals. `backtest.py` applies daily trailing stops on top. This separation lets you see signal intent vs. executed positions. `effective_weights()` in `backtest.py` is the live equivalent.

**Why the commodity budget is not normalized after trailing stops**
When a commodity ETF is stopped out, its weight moves to SHY rather than being redistributed to other commodities. This avoids concentrating in the one commodity that happens to be above its stop — if the whole basket is breaking down, we want cash, not the last-survivor.

**Why all sector strategies earn 0% on cash (SHY)**
All sector strategies are executed via Nordnet with weekly/monthly rebalancing and trailing stop loss orders. When a stop triggers, proceeds sit as uninvested cash — earning nothing. All sector backtest files zero out SHY's daily return before computing `raw_daily`: `daily_ret_no_cash[config.CASH_ETF] = 0.0`. Do not remove this — crediting SHY's ~4–5% annualized return would overstate live performance. This applies to sector_v1 through sector_v2e.

**Why `DD_SCALE=0.0` (full exit)**
The drawdown overlay exits fully (not partially) when in distress. Partial scaling (e.g., 0.3×) was tested but empirically worse — it keeps you exposed during continued drawdowns. Binary exit + full re-entry on the first green day captures upswings cleanly.

**Weight normalization (portfolio.py step 8)**
During severe rate-hike environments (e.g., 2022), ALL duration/credit ETFs can have negative momentum and get zeroed by the momentum filter. Without step 8, the portfolio would be under-invested (e.g., 7% weight). Step 8 parks the residual in SHY so the portfolio is always fully invested — the drawdown overlay then handles cash/exposure scaling.

## Module Responsibilities

| Module | Owns | Does NOT own |
|--------|------|-------------|
| `configs/bond_v1.py` | All tunable parameters | Logic |
| `strategies/bond_v1/signals.py` | FRED data → z-scores (daily) | Weight decisions |
| `strategies/bond_v1/portfolio.py` | Z-scores + momentum → monthly weights | Daily adjustments |
| `strategies/bond_v1/backtest.py` | Daily returns, trailing stops, vol target, DD overlay | Signal computation |
| `strategies/backtest_core.py` | Shared backtest building blocks (vol_scale, DD overlay, trailing stops) | Strategy-specific logic |
| `optimize.py` | Optuna search loop | Strategy logic |
| `main.py` | CLI parsing, orchestration | Business logic |

## Running the Strategy

```bash
export FRED_API_KEY=your_key_here
python main.py weights v1        # → bond V1 positions (display only)
python main.py weights v2        # → bond V2 positions
python main.py weights sector2e  # → sector V2e positions (current best, weekly)
python main.py trade v1          # → execute rebalance via IBKR Gateway
```

Strategy is always a positional argument (not a flag). Default is `v1` if omitted.
Available strategies: `v1`, `v2`, `v3`, `sector`, `sector2`, `sector2b`, `sector2c`, `sector2d`, `sector2e`.

## IBKR Gateway Integration

`python main.py trade [STRATEGY] [--dry-run]` connects to a running IBKR Gateway,
fetches live account equity and positions, and submits `MarketOrder`s to rebalance
to the strategy's effective weights (trailing-stop adjusted).

**Prerequisites:** IBKR Gateway must be running and logged in before calling `trade`.

**Ports:** 4002 = paper trading (default), 4001 = live account.

**Environment variables (all optional):**

| Variable | Default | Purpose |
|----------|---------|---------|
| `IBKR_HOST` | `127.0.0.1` | Gateway hostname |
| `IBKR_PORT` | `4002` | 4002 = paper, 4001 = live |
| `IBKR_CLIENT_ID` | `1` | API client ID (must be unique per active connection) |
| `IBKR_MIN_ORDER_USD` | `50` | Skip orders smaller than this value |

**Workflow:**
1. Run `python main.py trade --dry-run` to preview orders without submitting
2. Review the order table (current % → target % → Δ shares → est. $)
3. Run `python main.py trade` and type `y` to submit

**Note:** Trailing stops are computed in software via `effective_weights()` — the same
values shown by `python main.py weights`. No native IBKR trailing-stop orders are placed.

## Data Caching

- ETF prices and VIX: `data/cache/etf_prices.csv`, `data/cache/vix.csv` — refreshed when behind the last trading day
- FRED series: `data/cache/fred_<SERIES>.csv` — daily series refresh every 2 days, monthly series (CPI, FEDFUNDS, UNRATE, INDPRO) every 35 days, weekly (WALCL, TEDRATE) every 10 days
- Cache is committed to `.gitignore` (not tracked) — `python main.py fetch` to populate

## Optuna Optimization

```bash
python main.py optimize v1 --trials 300      # bond V1
python main.py optimize sector2b --trials 300
```

Searches over ~25 parameters (lookbacks, allocation caps, signal weights, trailing stop parameters). 70/30 train/test split. Objective: `Sharpe × ann_return × 10 − drawdown_penalty − return_penalty − worst_month_penalty`. Heavy penalty (20×) for max DD > 10%. Saves `best_params<suffix>.json` (e.g. `best_params.json` for V1, `best_params_sector2b.json` for V2b) which `main.py weights` loads automatically.

## Extending the Strategy

**Adding a new FRED signal:**
1. Add series to `FRED_SERIES` in `configs/bond_v1.py`
2. Add signal function in `strategies/bond_v1/signals.py` following the `_zscore()` pattern
3. Add weight constant to `configs/bond_v1.py` (e.g., `W_DURATION_NEWVAR = 0.10`)
4. Add to the composite in `compute_all_macro()`
5. Add to `PARAM_SPACE` in `optimize.py` to make it tunable

**Adding a new ETF:**
1. Add ticker to the appropriate list in `configs/bond_v1.py` (e.g., `CREDIT_ETFS`)
2. Delete `data/cache/etf_prices.csv` and run `python main.py fetch` to re-download
3. The momentum filter, inverse-vol weighting, and blending apply automatically

## Known Limitations

- **Backtest starts 2005** — limited pre-GFC history for some ETFs (PDBC launched 2012, others later). Weights are NaN for missing ETFs and fall back to the available subset.
- **No transaction cost model** — turnover is ~15-25%/month on active periods; real slippage/commission will reduce returns modestly.
- **FRED data lag** — CPI, UNRATE, INDPRO publish with 2-4 week lag. The strategy only uses month-end values, so this is correctly handled by the monthly rebalance.
- **TED spread discontinued 2023** — TEDRATE from FRED has no data after 2023-01-31. The `ted_stress_signal` returns 0 after that date, effectively dropping that sub-signal. This is handled gracefully.

## Sector Rotation Strategies

A separate strategy family runs pure momentum on a broad ETF universe with no FRED signals. It lives in its own modules to keep Optuna config patching isolated from the bond strategy.

**Current best strategy: `sector2e`** — V2d liquid universe + 24m/36m supercycle lookbacks. CAGR 43.6%, Sharpe 3.40, Max DD -7.3% (2005–2026, best params, no cash yield). Zero negative years 2005–2026. Full period (2000–2026): 38.2% CAGR — diluted by sparse ETF universe pre-2005.

### Strategy variants

| Key | Config | Description |
|-----|--------|-------------|
| `sector` | `configs/sector_v1.py` | XL-series, single-lookback momentum |
| `sector2` | `configs/sector_v2.py` | 35 ETFs, multi-timescale, adaptive stops, monthly rebalance |
| `sector2b` | `configs/sector_v2b.py` | Weekly rebalance, expanded 37-ETF universe |
| `sector2c` | `configs/sector_v2c.py` | Cross-asset + correlation filter + cluster caps |
| `sector2d` | `configs/sector_v2d.py` | Liquid ETFs only (≥$100M/day ADV filter) |
| `sector2e` | `configs/sector_v2e.py` | V2d universe + 24m/36m supercycle momentum lookbacks — **current best / production** |

### V2e-specific design: `_weighted_blend` NaN-safe helper

V2e adds 24m (504 days) and 36m (756 days) momentum lookbacks. During the warmup period these return NaN. The `_weighted_blend(pairs)` function in `strategies/sector_v2e/signals.py` computes the weighted average while ignoring NaN per cell — shorter lookbacks carry full weight until longer history is available. This replaces the simple division formula used in V2c/V2d `composite_score`. Do not simplify this back to a plain division or NaN will propagate and the strategy goes to cash for the first 3 years.

### Nordnet live execution (glidende stop loss)

Nordnet's "Glidende stop loss" is confirmed available for US-listed ETFs. It trails from the highest price since order placement.

**Do not reset stops every Friday.** The backtest uses an 86-day rolling peak. Resetting weekly makes the effective window 1 week — far too tight, will trigger on normal volatility. Instead:
- Let stops trail naturally from original placement
- Only cancel and reset when: (a) position is fully closed at rebalancing, (b) adaptive stop% changes by >3pp, or (c) order nears the 30-day Nordnet validity limit
- `python main.py weights sector2e` prints the exact stop% and stop price ready to enter in Nordnet

### Why each sector strategy has its own config module

Each `configs/sector_v*.py` is standalone. Optuna patches it via `setattr(cfg, k, v)` at runtime, identical to the bond-strategy pattern. Separate files prevent trial state from bleeding across strategies and ensure `import configs.bond_v1 as config` in bond modules is never shadowed.

### Module Responsibilities (all sector variants follow this pattern)

| Module | Owns | Does NOT own |
|--------|------|-------------|
| `configs/sector_v*.py` | All tunable parameters | Logic |
| `data/pipelines/sector_v*.py` | Price download + period resampling | Signal computation |
| `strategies/sector_v*/portfolio.py` | Multi-timescale momentum → weights | Daily adjustments |
| `strategies/sector_v*/backtest.py` | Daily returns, adaptive trailing stops, vol target, DD overlay | Signal computation |

### Weekly Resampling (V2b and above)

`data/pipelines/sector_v2b.py` calls `resample_to_period_end("W")` to convert daily prices to weekly. This uses `.resample("W").last()` which anchors on **Sunday** by calendar — but then `.values` is assigned back to the actual last trading day index (Friday, or Thursday on short weeks). The result is that rebalance dates are always real trading days, never calendar Sundays.

### Running Sector strategies

```bash
# Production (Sector V2e — current best)
python main.py weights sector2e       # today's positions + Nordnet stop prices
python main.py backtest sector2e --best  # full backtest + charts
python main.py optimize sector2e --trials 300

# Other variants
python main.py weights sector2b
python main.py backtest sector2b --best
python main.py backtest sector2c --best
python main.py backtest sector2d --best

# Comparison charts
python main.py compare-sector        # V2 / V2b / V2c → sector_comparison.png
python main.py v2c-long              # V2c + V2d extended history → v2c_extended.png
python main.py v2d-v2e               # V2d vs V2e → v2d_v2e.png
python main.py v2c-v2d-v2e          # three-way V2c/V2d/V2e → v2c_v2d_v2e.png
```

Optimised params load automatically from `best_params_sector2e.json` (and equivalent files for other variants). Key V2b params (for reference): `N_POSITIONS=4`, `MAX_WEIGHT=15.9%`, `STOP_TACTICAL=4%`, `STOP_SUPERCYCLE=14%`, `TRAILING_STOP_WINDOW=108d`, `DD_THRESHOLD=-14.6%`, `VOL_TARGET=17.0%`, `SPY_MA_WINDOW=154`.

### Testing Sector strategies

After any change to sector portfolio or backtest logic, run:

```bash
python - <<'EOF'
import warnings; warnings.filterwarnings("ignore")
from data.pipelines.sector_v2b import load_all
from strategies.sector_v2b.backtest import run
from analysis.performance import summary

prices = load_all()
res = run(prices)

s = summary(res["daily_returns"], res["nav"], "Sector V2b")
print(s)
EOF
```

Targets (Sector V2b, 2010-2026, optimised params, **no cash yield — SHY earns 0%**): CAGR > 42%, Sharpe > 2.8, Max Drawdown better than -8%.

For V2e use:
```bash
python - <<'EOF'
import warnings; warnings.filterwarnings("ignore")
from data.pipelines.sector_v2e import load_all
from strategies.sector_v2e.backtest import run
from analysis.performance import summary

prices = load_all()
res = run(prices)
s = summary(res["daily_returns"], res["nav"], "Sector V2e")
print(s)
EOF
```

Targets (Sector V2e, 2005-2026, optimised params, **no cash yield — SHY earns 0%**): CAGR > 42%, Sharpe > 3.2, Max Drawdown better than -10%.

## Testing Changes

After any change to signals, portfolio, or backtest logic, run:

```bash
FRED_API_KEY=<key> python - <<'EOF'
import warnings; warnings.filterwarnings("ignore")
from data.pipelines.bond_v1 import load_all
from strategies.bond_v1.backtest import run
from analysis.performance import summary
import pandas as pd

macro, prices = load_all()
res = run(macro, prices)

# Weight sanity check
sums = res["weights"].sum(axis=1)
bad = sums[abs(sums - 1.0) > 0.01]
assert len(bad) == 0, f"Weight sums broken: {bad}"

s = summary(res["daily_returns"], res["nav"], "Strategy")
print(s)
EOF
```

Targets (bond strategy, optimised params, 2011-2026, cash on stop-outs): CAGR > 19%, Sharpe (CAGR/vol) > 2.9, Max Drawdown better than -11%. Primary return metric is CAGR (geometric), not arithmetic mean×252.
