# Multi-Strategy Systematic Trading

Two independent strategy families: a **macro-driven bond + commodities rotation** (V1/V2/V3) and a **pure-momentum sector rotation** (Sector V1/V2/V2b/V2c/V2d/V2e). Both run daily trailing stops and share the same backtesting and optimisation infrastructure.

---

## Sector Rotation Strategy — V2e (current best)

Pure multi-timescale momentum across 37 liquid ETFs (all ≥$100M avg daily dollar volume). No FRED signals — price alone drives all decisions. Weekly rebalancing with adaptive trailing stops managed manually in Nordnet (glidende stop loss). Cash from stopped-out positions earns 0% — both in live trading and in the backtest.

**Key innovation vs earlier variants:** V2e adds 24m and 36m momentum components to the slow-momentum blend, giving the ranking signal memory of multi-year economic supercycles (commodity cycles, tech booms, rate cycles). A NaN-safe blending helper handles the warmup period gracefully so shorter lookbacks carry full weight until longer history is available.

### Performance (optimised params, no cash yield — SHY earns 0%)

| Metric          | Sector V1  | Sector V2  | Sector V2b | Sector V2c | Sector V2d | **Sector V2e** |
|----------------|------------|------------|------------|------------|------------|----------------|
| **Period**     | 2007–2026  | 2010–2026  | 2010–2026  | 2005–2026  | 2005–2026  | **2005–2026**  |
| **CAGR**       | 16.0%      | 42.5%      | 43.9%      | 40.9%      | 39.0%      | **43.6%**      |
| Sharpe         | 1.54       | 2.86       | 2.97       | **3.76**   | **3.82**   | 3.40           |
| Max Drawdown   | -10.4%     | -8.3%      | -6.2%      | -6.9%      | **-4.6%**  | -7.3%          |
| Worst Month    | -4.7%      | -6.4%      | -5.2%      | -2.7%      | **-2.1%**  | -4.0%          |
| ETF universe   | 18         | 35         | 37         | 46         | 38         | **37**         |
| Rebalance      | Monthly    | Monthly    | Weekly     | Weekly     | Weekly     | **Weekly**     |
| Negative years | —          | —          | 0/16       | —          | 0/21       | **0/21**       |

All figures assume SHY earns 0% — reflecting Nordnet behaviour where stopped-out cash sits uninvested. Period starts differ because many ETFs in the extended universe launched post-2005.

<details>
<summary>V2e year-by-year returns (2005–2026, no negative years)</summary>

| Year | Return  | Driver |
|------|---------|--------|
| 2005 | +44.4%  | Commodity supercycle |
| 2006 | +45.6%  | Commodity + energy |
| 2007 | +71.7%  | Commodity supercycle peak |
| 2008 | +6.0%   | DD overlay + stops absorbed GFC |
| 2009 | +23.6%  | Recovery rotation |
| 2010 | +40.4%  | Metals + EM |
| 2011 | +26.5%  | Gold + defensive |
| 2012 | +30.3%  | Risk-on rotation |
| 2013 | +42.9%  | Tech + biotech |
| 2014 | +30.6%  | Tech momentum |
| 2015 | +22.3%  | Healthcare + tech |
| 2016 | +54.7%  | Energy recovery + metals |
| 2017 | +59.6%  | Tech supercycle |
| 2018 | +33.7%  | Defensive rotation |
| 2019 | +33.4%  | Tech + gold |
| 2020 | +124.6% | Tech/growth surge post-COVID |
| 2021 | +55.5%  | Commodities + tech |
| 2022 | +14.7%  | Energy + commodity hedge |
| 2023 | +41.2%  | AI/semis cycle |
| 2024 | +57.4%  | AI/semis + gold supercycle |
| 2025 | +112.2% | Commodity + defense supercycle |
| 2026 | +37.7%  | Partial year |

</details>

### How It Works

**Multi-timescale momentum** — each ETF is scored on 1m, 3m, 12m, 18m, 24m, and 36m lookbacks simultaneously. The 24m/36m components are NaN-safe (gracefully excluded during warmup). Only the top `N_POSITIONS` ETFs by composite score enter the portfolio; the rest move to SHY. Inverse-vol weighting within the selected set, capped at `MAX_WEIGHT` per position.

**Adaptive trailing stops** — two stop layers: a tactical stop (`STOP_TACTICAL≈4%`) for near-term protection and a supercycle stop (`STOP_SUPERCYCLE≈14.5%`) that allows riding longer trends. The stop percentage adapts based on each ETF's 12m momentum — low momentum gets a tight stop, established supercycle gets a wide stop.

**Cluster caps** — prevents concentration within correlated groups (e.g., max 1 precious miner from GDX/GDXJ/SIL, max 2 from bonds). Ensures genuine diversification even when an entire cluster is trending.

**Correlation filter** — at each weekly rebalance, skips any candidate whose rolling 60-day correlation with an already-selected ETF exceeds `CORR_THRESHOLD≈0.59`.

**Drawdown overlay** — exits to cash when portfolio drawdown exceeds `DD_THRESHOLD≈-14.7%`. Re-enters fully on the first positive day.

**SPY market regime filter** — when SPY is below its 237-day moving average, positions scale to cash.

**Volatility targeting** — daily scaling so realised vol tracks `VOL_TARGET≈20.6%`, leverage capped at 1.0× (no margin).

### ETF Universe (V2e / V2d — 37 ETFs + SHY)

All ETFs ≥$100M average daily dollar volume (liquid-only filter applied in V2d):

| Group | ETFs | Avg ADV |
|-------|------|---------|
| Sector core | XLE, XLK, XLV, XLF, XLI, XLY, XLP, XLU, XLB, VNQ | $1.1B–$3.1B |
| Compute/AI | SMH, ARKK, IGV | $0.8B–$4.9B |
| Precious miners | GDX, GDXJ, SIL | $0.2B–$2.3B |
| Base miners | XME, COPX, REMX | $0.1B–$0.4B |
| Energy | XOP, OIH | $0.2B–$0.9B |
| Green energy | ICLN, URA | $0.1B–$0.2B |
| Defense | ITA | $0.3B |
| Gold | GLD | $5.0B |
| Biotech | XBI, IBB | $0.3B–$1.2B |
| China tech | KWEB | $0.7B |
| Bonds | TLT, IEF, HYG | $1.0B–$4.3B |
| International | EFA, EEM, EWJ, EWZ, INDA | $0.4B–$2.6B |
| Commodities | PDBC | $0.2B |
| Cash | SHY | — |

### Nordnet Live Execution

**Glidende stop loss** (trailing stop loss) is supported for US-listed ETFs on Nordnet.

Each weekly rebalance (`python main.py weights sector2e`):
1. Execute buy/sell orders to reach target weights
2. For each new or changed position: set a glidende stop loss at the adaptive stop% shown in output
3. **Do not reset stops weekly** — let them trail naturally from original placement. The backtest uses an 86-day rolling peak; resetting weekly creates a 1-week effective window, which is far too tight and will cause false exits on normal volatility
4. Only cancel and reset a stop when: (a) position is closed, (b) stop% changes by >3pp, or (c) order approaches 30-day validity limit

The `python main.py weights sector2e` output prints the exact stop% and stop price for each position.

### Sector CLI

```bash
# V2e (current best)
python main.py weights sector2e           # today's positions + stop prices
python main.py backtest sector2e --best   # full backtest + charts
python main.py optimize sector2e --trials 300

# Other variants
python main.py weights sector2b
python main.py weights sector2d

# Comparison charts
python main.py compare-sector             # V2 / V2b / V2c → sector_comparison.png
python main.py v2c-long                   # V2c + V2d extended history → v2c_extended.png
python main.py v2d-v2e                    # V2d vs V2e → v2d_v2e.png
python main.py v2c-v2d-v2e               # three-way comparison → v2c_v2d_v2e.png
```

---

## Bond + Commodities Rotation Strategy (V1/V2/V3)

Systematic macro-driven rotation across fixed income, commodity, and satellite ETFs. Targets strong risk-adjusted returns with max drawdown below 10%. Runs monthly with daily trailing stops.

Three strategy versions share the same codebase — pass `v2` or `v3` as the strategy argument to switch.

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

<details>
<summary>Year-by-year returns (Bond strategies)</summary>

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

## ETF Universe (Bond strategies)

| Bucket       | V1 ETFs                                | V2 additions       | Role |
|-------------|----------------------------------------|-------------------|------|
| Duration     | TLT, IEF, SHY                          | —                 | Defensive anchor / cash pool |
| Inflation    | TIP                                    | VTIP              | Inflation hedge |
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

### Risk Management

**Per-position trailing stops (daily)** — commodity and satellite ETFs exit if price drops >3% below the 21-day rolling peak. Freed weight moves to SHY.

**Drawdown overlay** — when portfolio drawdown exceeds threshold AND yesterday was negative, exposure scales down. Re-enters fully the next positive day.

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

Strategy is always a positional argument. Default is `v1` when omitted.

```bash
# Sector strategies (no FRED key needed)
python main.py weights sector2e           # today's positions + stop prices
python main.py backtest sector2e --best   # full backtest + charts
python main.py optimize sector2e --trials 300

# Bond strategies
python main.py fetch v1
python main.py backtest v1 --best
python main.py weights v1
python main.py weights v2
python main.py weights v3

# Comparison charts
python main.py compare                    # Bond V1/V2/V3 → backtest_comparison.png
python main.py compare-sector             # Sector V2/V2b/V2c → sector_comparison.png
python main.py v2c-long                   # V2c + V2d extended history → v2c_extended.png
python main.py v2d-v2e                    # V2d vs V2e → v2d_v2e.png
python main.py v2c-v2d-v2e               # three-way comparison → v2c_v2d_v2e.png
```

The `weights` command is the production entry point. Run it each week-end (sector) or month-end (bond) to get exact position sizes. It shows both raw signal weights and trailing-stop-adjusted effective positions, plus the adaptive stop% and stop price for each held position.

## Weekly Sector Workflow (V2e)

1. `python main.py weights sector2e` — get target weights and stop prices
2. Execute buy/sell orders in Nordnet to reach target weights
3. For new/changed positions: set glidende stop loss at the stop% shown in output
4. Leave existing stops running (do not reset weekly — backtest uses 86-day trailing window)
5. Only reset a stop when: position closed, stop% changes >3pp, or order nears 30-day limit

## Re-optimising

```bash
python main.py optimize sector2e --trials 300
python main.py optimize sector2d --trials 300
python main.py optimize v1 --trials 500
python main.py optimize v2 --trials 300
```

Best parameters save to `best_params_sector2e.json` etc., loaded automatically by `weights`.

## Project Structure

```
bond-trading/
├── main.py                     # CLI entry point — strategy registry + all subcommands
├── optimize.py                 # Optuna optimisation (all strategies)
├── configs/
│   ├── bond_v1.py              # Bond V1 parameters
│   ├── bond_v2.py              # Bond V2 parameters
│   ├── bond_v3.py              # Bond V3 parameters
│   ├── sector_base.py          # Shared sector defaults + BASE_PARAM_SPACE
│   ├── sector_v1.py            # Sector V1 parameters
│   ├── sector_v2.py            # Sector V2 parameters (35 ETFs, monthly)
│   ├── sector_v2b.py           # Sector V2b parameters (37 ETFs, weekly)
│   ├── sector_v2c.py           # Sector V2c parameters (cross-asset + cluster caps)
│   ├── sector_v2d.py           # Sector V2d parameters (liquid ETFs ≥$100M/day)
│   └── sector_v2e.py           # Sector V2e parameters (V2d + 24m/36m supercycle) ← current best
├── strategies/
│   ├── backtest_core.py        # Shared utilities: vol_scale, DD overlay, trailing stops
│   ├── bond_v1/                # signals.py, portfolio.py, backtest.py
│   ├── bond_v2/
│   ├── bond_v3/
│   ├── sector_v1/
│   ├── sector_v2/
│   ├── sector_v2b/
│   ├── sector_v2c/
│   ├── sector_v2d/
│   └── sector_v2e/             # ← current best
├── data/
│   ├── fred_client.py          # FRED API + caching
│   ├── price_client.py         # Yahoo Finance ETF prices + caching
│   ├── pipelines/              # One module per strategy: load_all() → prices
│   └── cache/                  # git-ignored; populate with python main.py fetch
├── analysis/
│   └── performance.py          # Metrics + all chart functions
├── broker/
│   └── ibkr_client.py          # IBKR Gateway API client
├── best_params_sector2e.json   # Sector V2e optimised parameters ← primary
├── best_params_sector2d.json
├── best_params_sector2b.json
├── best_params.json            # Bond V1
├── best_params_v2.json
├── best_params_v3.json
└── logs/                       # Per-strategy log files
```
