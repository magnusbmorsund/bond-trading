# CLAUDE.md — Developer Guide for Claude Code

## Architecture

All parameters flow through `config.py`. Optuna patches `config` attributes at runtime via `setattr(config, k, v)` — this is why every module does `import config` and reads `config.X` at call time (never `from config import X` for any tunable parameter).

## Key Design Decisions

**Why trailing stops are in `backtest.py`, not `portfolio.py`**
`portfolio.py` builds monthly target weights from macro signals. `backtest.py` applies daily trailing stops on top. This separation lets you see signal intent vs. executed positions. `effective_weights()` in `backtest.py` is the live equivalent.

**Why the commodity budget is not normalized after trailing stops**
When a commodity ETF is stopped out, its weight moves to SHY rather than being redistributed to other commodities. This avoids concentrating in the one commodity that happens to be above its stop — if the whole basket is breaking down, we want cash, not the last-survivor.

**Why `DD_SCALE=0.0` (full exit)**
The drawdown overlay exits fully (not partially) when in distress. Partial scaling (e.g., 0.3×) was tested but empirically worse — it keeps you exposed during continued drawdowns. Binary exit + full re-entry on the first green day captures upswings cleanly.

**Weight normalization (portfolio.py step 8)**
During severe rate-hike environments (e.g., 2022), ALL duration/credit ETFs can have negative momentum and get zeroed by the momentum filter. Without step 8, the portfolio would be under-invested (e.g., 7% weight). Step 8 parks the residual in SHY so the portfolio is always fully invested — the drawdown overlay then handles cash/exposure scaling.

## Module Responsibilities

| Module | Owns | Does NOT own |
|--------|------|-------------|
| `config.py` | All tunable parameters | Logic |
| `signals.py` | FRED data → z-scores (daily) | Weight decisions |
| `portfolio.py` | Z-scores + momentum → monthly weights | Daily adjustments |
| `backtest.py` | Daily returns, trailing stops, vol target, DD overlay | Signal computation |
| `optimize.py` | Optuna search loop | Strategy logic |
| `main.py` | CLI parsing, orchestration | Business logic |

## Running the Strategy

```bash
export FRED_API_KEY=your_key_here
python main.py weights   # → get today's IBKR positions (display only)
python main.py trade     # → execute rebalance via IBKR Gateway
```

## IBKR Gateway Integration

`python main.py trade [--v2|--v3] [--dry-run]` connects to a running IBKR Gateway,
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
python main.py optimize --trials 300
```

Searches over ~25 parameters (lookbacks, allocation caps, signal weights, trailing stop parameters). 70/30 train/test split. Objective: `Sharpe × ann_return × 10 − drawdown_penalty − return_penalty − worst_month_penalty`. Heavy penalty (20×) for max DD > 10%. Saves `best_params.json` which `main.py weights` loads automatically.

## Extending the Strategy

**Adding a new FRED signal:**
1. Add series to `FRED_SERIES` in `config.py`
2. Add signal function in `signals.py` following the `_zscore()` pattern
3. Add weight constant to `config.py` (e.g., `W_DURATION_NEWVAR = 0.10`)
4. Add to the composite in `compute_all_macro()`
5. Add to `PARAM_SPACE` in `optimize.py` to make it tunable

**Adding a new ETF:**
1. Add ticker to the appropriate list in `config.py` (e.g., `CREDIT_ETFS`)
2. Delete `data/cache/etf_prices.csv` and run `python main.py fetch` to re-download
3. The momentum filter, inverse-vol weighting, and blending apply automatically

## Known Limitations

- **Backtest starts 2005** — limited pre-GFC history for some ETFs (PDBC launched 2012, others later). Weights are NaN for missing ETFs and fall back to the available subset.
- **No transaction cost model** — turnover is ~15-25%/month on active periods; real slippage/commission will reduce returns modestly.
- **FRED data lag** — CPI, UNRATE, INDPRO publish with 2-4 week lag. The strategy only uses month-end values, so this is correctly handled by the monthly rebalance.
- **TED spread discontinued 2023** — TEDRATE from FRED has no data after 2023-01-31. The `ted_stress_signal` returns 0 after that date, effectively dropping that sub-signal. This is handled gracefully.

## Sector Rotation Strategies

A separate strategy family (Sector V1, V2, V2b) runs pure momentum on a broad ETF universe with no FRED signals. It lives in its own modules to keep Optuna config patching isolated from the bond strategy.

### Why `strategy_sector_v2b/` is a separate package

`config_sector_v2b.py` is a standalone module — not `config.py`. Optuna patches it via `setattr(config_sector_v2b, k, v)` at runtime, identical to the bond-strategy pattern. Keeping sector configs in a separate file means `import config` in bond modules is never shadowed and trial state never bleeds across strategies.

### Module Responsibilities (Sector V2b)

| Module | Owns | Does NOT own |
|--------|------|-------------|
| `config_sector_v2b.py` | All sector V2b tunable parameters | Logic |
| `data/pipeline_sector_v2b.py` | Price download + weekly resampling | Signal computation |
| `strategy_sector_v2b/portfolio.py` | Multi-timescale momentum → weekly weights | Daily adjustments |
| `strategy_sector_v2b/backtest.py` | Daily returns, adaptive trailing stops, vol target, DD overlay | Signal computation |
| `strategy_sector_v2b/optimize.py` | Optuna search loop for sector V2b | Strategy logic |

### Weekly Resampling

`pipeline_sector_v2b.py` calls `resample_to_period_end("W")` to convert daily prices to weekly. This uses `.resample("W").last()` which anchors on **Sunday** by calendar — but then `.values` is assigned back to the actual last trading day index (Friday, or Thursday on short weeks). The result is that rebalance dates are always real trading days, never calendar Sundays.

### Running Sector V2b

```bash
python main.py weights --sector2b --best      # today's IBKR positions
python main.py backtest --sector2b --best     # full 2010-2026 backtest
python main.py optimize --sector2b --trials 300
```

Optimised params live in `best_params_sector2b.json` and are loaded automatically by `--best`. Key params: `N_POSITIONS=4`, `MAX_WEIGHT=15.9%`, `STOP_TACTICAL=4%`, `STOP_SUPERCYCLE=14%`, `TRAILING_STOP_WINDOW=108d`, `DD_THRESHOLD=-14.6%`, `VOL_TARGET=17.0%`, `SPY_MA_WINDOW=154`.

### Testing Sector V2b

After any change to sector portfolio or backtest logic, run:

```bash
python - <<'EOF'
import warnings; warnings.filterwarnings("ignore")
from data.pipeline_sector_v2b import load_all
from strategy_sector_v2b.backtest import run
from analysis.performance import summary

prices = load_all()
res = run(prices)

s = summary(res["daily_returns"], res["nav"], "Sector V2b")
print(s)
EOF
```

Targets (Sector V2b, 2010-2026, optimised params): CAGR > 44%, Sharpe > 2.9, Max Drawdown better than -8%.

## Testing Changes

After any change to signals, portfolio, or backtest logic, run:

```bash
FRED_API_KEY=<key> python - <<'EOF'
import warnings; warnings.filterwarnings("ignore")
from data.pipeline import load_all
from strategy.backtest import run
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
