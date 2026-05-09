# Multi-Strategy Systematic Trading

Two independent strategy families: a **macro-driven bond + commodities rotation** (V1/V2/V3) and a **pure-momentum sector rotation** (Sector V1/V2/V2b). Both run daily trailing stops and share the same backtesting and optimisation infrastructure.

---

## Sector Rotation Strategy (V2b)

Pure multi-timescale momentum across 37 ETFs. No FRED signals — price alone drives all decisions. Weekly rebalancing with daily adaptive trailing stops placed natively at the broker.

### Performance (2010–2026 backtest, optimised params)

| Metric               | Sector V1 | Sector V2 | Sector V2b |
|---------------------|-----------|-----------|------------|
| **CAGR**            | —         | **43.9%** | **45.0%**  |
| Sharpe (CAGR / vol) | —         | 2.93      | **3.03**   |
| Max Drawdown        | —         | -8.2%     | **-7.0%**  |
| Total Return        | —         | —         | **42,815%**|
| ETF universe        | —         | 32        | **37**     |
| Rebalance frequency | —         | Monthly   | **Weekly** |
| OOS CAGR (2021-26)  | —         | —         | **42.2%**  |
| OOS Sharpe (2021-26)| —         | —         | **3.06**   |

All 16 years 2011-2026 positive. OOS figures confirm no overfitting (in-sample ends 2020).

<details>
<summary>Year-by-year returns (Sector V2b)</summary>

| Year | Return |
|------|--------|
| 2011 | +2.1%  |
| 2012 | +30.8% |
| 2013 | +69.9% |
| 2014 | +42.0% |
| 2015 | +35.5% |
| 2016 | +76.8% |
| 2017 | +57.6% |
| 2018 | +32.7% |
| 2019 | +37.2% |
| 2020 | +122.2%|
| 2021 | +51.8% |
| 2022 | +2.6%  |
| 2023 | +39.5% |
| 2024 | +61.4% |
| 2025 | +93.1% |
| 2026 (partial) | +29.0% |

</details>

### How It Works

**Multi-timescale momentum** — each ETF is scored on short, medium, and long lookbacks simultaneously. Only the top `N_POSITIONS` ETFs by composite score enter the portfolio; the rest move to cash (or SHY). Inverse-vol weighting within the selected set, capped at `MAX_WEIGHT` per position.

**Adaptive trailing stops** — two stop layers: a tactical stop (`STOP_TACTICAL=4%`) for near-term drawdown protection and a supercycle stop (`STOP_SUPERCYCLE=14%`) that allows riding longer trends. Stops are placed as native broker orders (not soft computed), so intraday execution is possible.

**Supercycle riding** — the `ALPHA_SLOW` parameter (0.72) gives heavy weight to the long-horizon momentum score, which keeps the strategy in structural winners (e.g., GLD 2020, tech ETFs 2023-24) through normal volatility while the tactical stop handles crashes.

**Drawdown overlay** — exits to cash when portfolio drawdown exceeds `DD_THRESHOLD=-14.6%`. Re-enters fully on the first positive day.

**SPY market regime filter** — when SPY is below its `SPY_MA_WINDOW=154` day moving average, position sizing scales down.

### ETF Universe (V2b — 37 ETFs)

V2b adds **GLD, ITA, XBI, TAN, KWEB** relative to V2's 32-ETF universe, capturing gold, aerospace/defence, biotech, clean energy, and Chinese tech as additional momentum sources.

### Sector V2b CLI

```bash
python main.py weights --sector2b --best      # today's positions
python main.py backtest --sector2b --best     # full backtest
python main.py optimize --sector2b --trials 300
```

---

## Bond + Commodities Rotation Strategy (V1/V2/V3)

Systematic macro-driven rotation across fixed income, commodity, and satellite ETFs. Targets strong risk-adjusted returns with max drawdown below 10%. Runs monthly with daily trailing stops.

Three strategy versions share the same codebase — add `--v2` or `--v3` to any command to switch.

## Performance (2011–2026 backtest, optimised params, cash on stop-outs)

Stop-triggered positions hold cash at 0% (not SHY) until the next monthly rebalance. CAGR is the primary return metric — it compounds daily returns geometrically and is the number a buy-and-hold investor actually experiences.

| Metric                | V1       | V2       | V3       |
|----------------------|----------|----------|----------|
| **CAGR**             | **19.0%**| **21.3%**| **19.4%**|
| Arith. Ann. Return   | 17.6%    | 19.6%    | 17.9%    |
| Volatility           | 6.4%     | 7.1%     | 5.3%     |
| Sharpe (CAGR / vol)  | 2.98     | 2.99     | **3.68** |
| Max Drawdown         | -6.6%    | -10.1%   | **-6.6%**|
| Calmar (CAGR)        | 2.89     | 2.10     | **2.94** |
| Worst Month          | -4.6%    | -5.7%    | **-1.9%**|
| Final NAV            | 14.3x    | **19.2x**| 15.2x    |
| Turnover (avg)       | 17%/mo   | 22%/mo   | 13%/mo   |

**V3** — best risk-adjusted: Sharpe 3.68, MaxDD -6.6%, worst month only -1.9%.  
**V2** — highest absolute return: 21.3% CAGR, 19.2x final NAV, at the cost of deeper drawdowns.  
**V1** — conservative middle ground: 19.0% CAGR, MaxDD -6.6%, lower turnover.

Exact figures and year-by-year breakdown stored in [`backtest_results.json`](backtest_results.json).

<details>
<summary>Year-by-year returns</summary>

| Year | V1    | V2     | V3    |
|------|-------|--------|-------|
| 2011 | 33.0% |  54.9% | 40.4% |
| 2012 | 10.7% |  11.6% | 11.4% |
| 2013 |  0.4% |  -2.3% | -3.2% |
| 2014 |  2.3% |  13.3% |  5.6% |
| 2015 |  9.2% |   6.8% |  7.1% |
| 2016 | 27.7% |  33.9% | 22.1% |
| 2017 | 14.7% |  20.7% | 15.3% |
| 2018 |  5.4% |   2.2% |  5.1% |
| 2019 | 22.9% |  19.5% | 16.0% |
| 2020 | 39.3% |  55.8% | 44.8% |
| 2021 | 16.9% |  20.4% | 24.2% |
| 2022 |  1.3% |  -7.7% |  5.2% |
| 2023 | 20.1% |  12.1% | 13.4% |
| 2024 | 41.4% |  46.3% | 39.1% |
| 2025 | 34.5% |  43.7% | 44.4% |
| 2026 | 21.5% |  16.7% | 19.8% |

</details>

## ETF Universe

| Bucket       | V1 ETFs                                | V2 additions       | Role |
|-------------|----------------------------------------|-------------------|------|
| Duration     | TLT, IEF, SHY                          | —                 | Defensive anchor / cash pool |
| Inflation    | TIP                                    | VTIP              | Inflation hedge (V2 splits by duration risk) |
| Credit       | LQD, HYG, ANGL, SJNK, BKLN, EMB, PFF  | —                 | Spread income |
| Commodities  | GLD, PDBC, DBA                         | SLV               | Primary alpha source |
| Real Assets  | —                                      | VNQ               | V2 REIT satellite |
| Equity       | —                                      | SPY               | V2 growth regime satellite |

## How It Works

### Signals (FRED data, updated daily)

Three composite z-scores drive all allocation decisions:

**`duration_z`** — positive → favour TLT/IEF (long bonds)
- 2s10s yield curve slope (20%)
- 10Y-3M spread, better recession predictor (20%)
- Fed funds rate direction (15%)
- 10Y real yield (DFII10) — the key 2022 signal (25%)
- Unemployment trend / Sahm rule (10%)
- ISM Manufacturing PMI / Industrial production deceleration (10%)

**`credit_z`** — positive → favour credit ETFs
- HY OAS level, inverted (35%)
- IG spread momentum / widening speed (15%)
- VIX regime (20%)
- Fed balance sheet QE/QT (15%)
- TED spread financial stress (15%)

**`inflation_z`** — positive → favour TIP/VTIP
- 10Y breakeven inflation ROC (50%)
- CPI YoY momentum (50%)

**`usd_z`** (V2 only) — rising USD dampens commodity allocation
- Nominal broad trade-weighted dollar index (DTWEXBGS), 3-month momentum

### Allocation

**V1 buckets (in order):**
1. **Commodity basket** — size grows with `0.5×inflation_z + 0.5×duration_z`; max 40%. Inverse-vol weighted within bucket, gated by 12-1 month momentum.
2. **Credit** — scales with `credit_z`, hard-capped at 50% and further capped when VIX > 25.
3. **TIP** — scales with `inflation_z`, max 15%.
4. **Duration** — remainder. TLT/IEF/SHY proportions set by softmax on `duration_z`.
5. **Momentum filter** — any ETF with negative 12-1 month momentum is zeroed; freed weight parks in SHY.

**V2 additions:**
- **SLV** in commodity basket — silver complements GLD (higher beta, industrial exposure); inverse-vol weighted automatically.
- **VTIP alongside TIP** — when `duration_z` is negative (rates rising), allocation shifts toward VTIP (2.5yr duration) to avoid duration bleed. VTIP split was -16% in 2022 vs TIP's -16%.
- **SPY equity satellite** — active only when VIX < 10 + credit spreads tight + SPY momentum positive. Max 15% of portfolio.
- **VNQ real estate satellite** — active when inflation and credit are both positive. Max 5% of portfolio.
- **USD signal** dampens commodity budget when dollar is rising (commodities priced in USD).

### Risk Management

**Per-position trailing stops (daily)** — commodity and satellite ETFs exit if price drops >3% below the 21-day rolling peak. Freed weight moves to SHY. Key edge: macro signals drive monthly entry; price-driven stops drive daily exit.

**Drawdown overlay** — when portfolio drawdown exceeds threshold AND yesterday was negative, exposure scales down. Re-enters fully the next positive day. Cash earns the fed funds rate.
- V1: threshold -3%, scale to 30%
- V2: threshold -2%, scale to 45%
- V3: threshold -14%, scale to 10% (effectively ride-through with light trim)

**Volatility targeting** — daily scaling so realised vol tracks target, leverage capped at maximum.
- V1: 15% vol target, 1.75× leverage cap
- V2: 12% vol target, 1.75× leverage cap
- V3: 12% vol target, 1.50× leverage cap

## Setup

```bash
pip install -r requirements.txt
export FRED_API_KEY=your_key_here   # free key at fred.stlouisfed.org/docs/api/api_key.html
```

## Usage

```bash
# V1 strategy
python main.py fetch
python main.py backtest --best
python main.py weights
python main.py optimize --trials 300

# V2 strategy — append --v2 to any command
python main.py fetch --v2
python main.py backtest --v2 --best
python main.py weights --v2
python main.py optimize --v2 --trials 300

# V3 strategy — append --v3 to any command
python main.py fetch --v3
python main.py backtest --v3 --best
python main.py weights --v3
python main.py optimize --v3 --trials 300
```

The `weights` command is the production entry point. Run it each month-end to get exact position sizes for IBKR. It shows both raw signal weights and trailing-stop-adjusted effective positions.

## Monthly Workflow

1. `python main.py fetch [--v2]` — refresh FRED + price data
2. `python main.py weights [--v2]` — read the "EFFECTIVE POSITIONS" table
3. Set each ETF to the shown % of total portfolio value in IBKR
4. Note any `[STOPPED OUT]` ETFs — these should be flat

## Re-optimising

```bash
python main.py optimize --trials 500          # re-optimise V1
python main.py optimize --v2 --trials 300     # re-optimise V2
python main.py optimize --v3 --trials 300     # re-optimise V3
```

Best parameters save to `best_params.json` (V1), `best_params_v2.json` (V2), and `best_params_v3.json` (V3), loaded automatically by `weights`. Re-run when macro regime shifts substantially or after 12+ months of live trading.

## Project Structure

```
bond-trading/
├── config.py                   # Bond V1 parameters
├── config_v2.py                # Bond V2 parameters (SLV, VTIP, VNQ, SPY, USD signal)
├── config_v3.py                # Bond V3 parameters (managed futures DBMF/CTA)
├── config_v2b.py               # Bond V2b experiment (3-month duration momentum gate)
├── config_sector_v2.py         # Sector V2 parameters (32 ETFs, monthly)
├── config_sector_v2b.py        # Sector V2b parameters (37 ETFs, weekly) ← production
├── main.py                     # CLI entry point (--v2/--v3/--sector2b flags)
├── optimize.py                 # Optuna optimisation (supports all strategy flags)
├── best_params.json            # Bond V1 production parameters
├── best_params_v2.json         # Bond V2 production parameters
├── best_params_v3.json         # Bond V3 production parameters
├── best_params_sector2.json    # Sector V2 production parameters
├── best_params_sector2b.json   # Sector V2b production parameters ← primary
├── daily_weights.py            # GitHub Actions: compute + email weights daily
├── strategy/                   # Bond V1 strategy modules
│   ├── signals.py
│   ├── portfolio.py
│   └── backtest.py
├── strategy_v2/                # Bond V2 strategy modules
│   ├── signals.py              # + USD signal (DTWEXBGS), ISM fallback
│   ├── portfolio.py            # + equity satellite, VNQ, VTIP/TIP split
│   └── backtest.py
├── strategy_v3/                # Bond V3 strategy modules (+ managed futures)
│   ├── signals.py
│   ├── portfolio.py
│   └── backtest.py
├── strategy_v2b/               # Bond V2b experiment (not in production)
│   ├── signals.py
│   ├── portfolio.py
│   └── backtest.py
├── strategy_sector_v2/         # Sector V2 modules (32 ETFs, monthly)
│   ├── portfolio.py
│   └── backtest.py
├── strategy_sector_v2b/        # Sector V2b modules (37 ETFs, weekly) ← production
│   ├── portfolio.py
│   └── backtest.py
├── data/
│   ├── pipeline.py             # Bond V1 data loading
│   ├── pipeline_v2.py          # Bond V2/V2b data loading (etf_prices_v2.csv cache)
│   ├── pipeline_v3.py          # Bond V3 data loading (etf_prices_v3.csv cache)
│   ├── pipeline_sector_v2.py   # Sector V2 data loading (monthly resampling)
│   ├── pipeline_sector_v2b.py  # Sector V2b data loading (weekly resampling)
│   ├── fred_client.py          # FRED API + caching
│   └── price_client.py         # Yahoo Finance ETF prices + caching
└── analysis/
    └── performance.py          # Metrics + charts
```

## Key Config Parameters

| Parameter | V1 (optimised) | V2 (optimised) | V3 (optimised) | Description |
|-----------|---------------|---------------|---------------|-------------|
| `MAX_ALT_ALLOC` | 60% | 60% | 60% | Max commodity basket allocation |
| `TRAILING_STOP_PCT` | 3% | 3% | 3% | Exit if price drops this far from rolling peak |
| `TRAILING_STOP_WINDOW` | 21 days | 21 days | 21 days | Rolling peak lookback for trailing stop |
| `VOL_TARGET` | 15% | 12% | 12% | Portfolio volatility target |
| `MAX_LEVERAGE` | 1.75× | 1.75× | 1.50× | Max daily vol-scaling leverage |
| `DD_THRESHOLD` | -3% | -2% | -14% | Drawdown level that triggers exposure scaling |
| `DD_SCALE` | 30% | 45% | 10% | Exposure kept during drawdown event |
| `MAX_CREDIT_ALLOC` | 80% | 80% | 75% | Max credit bucket allocation |
| `MAX_EQUITY_ALLOC` | — | 5% | 15% | Equity satellite cap |
| `MAX_REALESTATE_ALLOC` | — | 10% | 10% | VNQ allocation cap |
| `W_COMMODITY_USD` | — | 0.40 | 0.05 | USD drag weight on commodity budget |
| `VTIP_DURATION_SCALE` | — | 1.1 | 1.1 | Sensitivity of TIP/VTIP split to duration_z |
