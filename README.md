# Multi-Strategy Systematic Trading

Two independent strategy families: a **macro-driven bond + commodities rotation** (V1/V2/V3) and a **pure-momentum sector rotation** (Sector V1/V2/V2b…V2h). Both run daily trailing stops and share the same backtesting and optimisation infrastructure.

> **⚠️ Read this before trusting any number below.** Earlier versions of this README reported spectacular returns (Sector V2e at "43.6% CAGR / Sharpe 3.40", bond strategies at ~20% CAGR). **Those numbers were a backtest artifact.** The trailing-stop logic contained a look-ahead bug (the position dodged the loss on the day the stop triggered) and an unrealistic fill assumption. After fixing both (`backtest_core.py` now lags the stop trigger by one day, PRs #6/#8), the honest figures are far more modest: roughly **3–6% CAGR, Sharpe 0.4–0.9, double-digit drawdowns**. The tables below are the corrected, post-fix figures. See [`stop-lookahead-bias`](#integrity-notes) and the `quant-backtest-integrity` skill for the full story.

---

## Honest Performance Summary

All figures below are produced by the **current** (look-ahead-fixed) engine with optimised params (`best_params*.json`), over each strategy's full available history through 2026-06-05. Sector strategies assume stopped-out cash earns 0% (Nordnet behaviour); bond strategies hold cash at 0% on stop-outs until the next monthly rebalance. CAGR is geometric.

| Strategy | Period | CAGR | Sharpe | Max DD | Worst Month | Final NAV |
|----------|--------|------|--------|--------|-------------|-----------|
| Bond V1 | 2003–2026 | 2.7% | 0.85 | -12.7% | -5.3% | 1.87× |
| Bond V2 | 2003–2026 | 2.9% | 0.77 | -12.9% | -4.6% | 1.96× |
| Bond V3 | 2003–2026 | 2.9% | 0.88 | -11.1% | -2.3% | 1.97× |
| Sector V1 | 2007–2026 | 1.6% | 0.26 | -15.1% | -5.7% | 1.36× |
| Sector V2b | 2010–2026 | 3.9% | 0.49 | -26.0% | -4.7% | 1.88× |
| Sector V2c | 2000–2026 | 4.3% | 0.56 | -16.5% | -6.8% | 3.06× |
| Sector V2d | 2000–2026 | 5.5% | 0.67 | -18.1% | -4.5% | 4.05× |
| Sector V2e | 2000–2026 | 4.5% | 0.62 | -20.0% | -7.8% | 3.19× |
| Sector V2f | 2000–2026 | 4.7% | 0.64 | -18.2% | -6.3% | 3.32× |
| **Sector V2g** | 2000–2026 | **5.8%** | **0.65** | -20.8% | -7.7% | 4.47× |

Periods differ because ETFs in the extended universes launched at different dates. These are honest momentum-strategy returns — comparable to or modestly below buy-and-hold equity on a risk-adjusted basis, not the "spectacular alpha" the broken backtest suggested.

> **Reproduce:** `python main.py backtest sector2e --best` (and equivalent for each key). Any change to signals/portfolio/backtest/stop logic must leave these figures essentially unchanged — a sudden jump back to 20–40% CAGR means a look-ahead or fill bug has been reintroduced.

---

## Sector Rotation Strategy

Pure multi-timescale momentum across a broad liquid-ETF universe. No FRED signals — price alone drives all decisions. Weekly rebalancing with trailing stops managed manually in Nordnet (glidende stop loss). Cash from stopped-out positions earns 0% — both in live trading and in the backtest.

### Variants

| Key | Config | Description |
|-----|--------|-------------|
| `sector` | `configs/sector_v1.py` | XL-series, single-lookback momentum, monthly |
| `sector2` | `configs/sector_v2.py` | 35 ETFs, multi-timescale, adaptive stops, monthly (supports `--stop-freq`) |
| `sector2b` | `configs/sector_v2b.py` | Weekly rebalance, expanded 37-ETF universe |
| `sector2c` | `configs/sector_v2c.py` | Cross-asset + correlation filter + cluster caps |
| `sector2d` | `configs/sector_v2d.py` | Liquid ETFs only (≥$100M/day ADV filter) |
| `sector2e` | `configs/sector_v2e.py` | V2d universe + 24m/36m supercycle momentum lookbacks |
| `sector2f` | `configs/sector_v2f.py` | V2e minus XBI and IGV — **UCITS-tradeable subset for Nordnet retail** |
| `sector2g` | `configs/sector_v2g.py` | **Honest rebuild**: MA-break exit + concentrated (N=3). Best Sharpe on the corrected engine |
| `sector2h` | `configs/sector_v2h.py` | Tight fixed trailing stop **filled at the stop price**, gap-aware execution model |

**Why V2g and V2h exist.** Once the look-ahead bug was fixed, the original adaptive %-off-peak trailing stop turned out to be the *worst* exit on this universe — it whipsaws volatile thematic/supercycle ETFs. Two honest responses:
- **V2g** drops the trailing stop entirely in favour of a slow 200-day moving-average trend break (`EXIT_MODE="ma_break"`), concentrating into 3 high-conviction winners. "Ride winners, cut losers slowly."
- **V2h** keeps a tight ~10% trailing stop but models the realistic *resting-stop fill* (fill at the stop level minus slippage, with a deterministic fat-tail gap), consistent with Kaminski & Lo on momentum stops. The value of a stop hinges almost entirely on the fill price assumption.

### How It Works

**Multi-timescale momentum** — each ETF is scored on 1m, 3m, 12m, 18m (and for V2e+, 24m, 36m) lookbacks simultaneously. The long lookbacks are NaN-safe via `_weighted_blend` (gracefully excluded during warmup so shorter lookbacks carry full weight). Only the top `N_POSITIONS` ETFs by composite score enter; the rest move to SHY. Inverse-vol weighting within the selected set, capped at `MAX_WEIGHT`.

**Exit logic** — depends on the variant: adaptive %-off-peak trailing stop (V2/…/V2f), 200-day MA trend break (V2g), or fixed trailing stop filled at the stop level (V2h). The trigger is **lagged one day** in the backtest (`backtest_core.py`) — the position is observed below its stop at the day-T close and exits from T+1, holding *through* day-T's loss. Removing this lag reintroduces the look-ahead bug.

**Cluster caps** — prevents concentration within correlated groups (e.g. max 1 precious miner from GDX/GDXJ/SIL).

**Correlation filter** — at each rebalance, skips a candidate whose rolling correlation with an already-selected ETF exceeds `CORR_THRESHOLD`.

**Drawdown overlay** — exits to cash when portfolio drawdown exceeds `DD_THRESHOLD`; re-enters fully on the first positive day.

**SPY market regime filter** — scales to cash when SPY is below its long moving average.

**Volatility targeting** — daily scaling so realised vol tracks `VOL_TARGET`, leverage capped at 1.0× (no margin) for sector variants.

### ETF Universe (V2e / V2d — ~37 ETFs + SHY)

All ETFs ≥$100M average daily dollar volume (liquid-only filter applied in V2d):

| Group | ETFs |
|-------|------|
| Sector core | XLE, XLK, XLV, XLF, XLI, XLY, XLP, XLU, XLB, VNQ |
| Compute/AI | SMH, ARKK, IGV |
| Precious miners | GDX, GDXJ, SIL |
| Base miners | XME, COPX, REMX |
| Energy | XOP, OIH |
| Green energy | ICLN, URA |
| Defense | ITA |
| Gold | GLD |
| Biotech | XBI, IBB |
| China tech | KWEB |
| Bonds | TLT, IEF, HYG |
| International | EFA, EEM, EWJ, EWZ, INDA |
| Commodities | PDBC |
| Cash | SHY |

V2f drops **XBI** and **IGV** (no acceptable UCITS equivalent on Nordnet); V2g/V2h inherit the V2f universe.

### Honest year-by-year returns (Sector, optimised params, no cash yield)

| Year | V2e | V2f | V2g |
|------|------|------|------|
| 2003 |  9.5% | 17.9% | 25.1% |
| 2004 |  4.6% |  6.1% |  7.0% |
| 2005 |  5.3% | 10.8% | 17.9% |
| 2006 | 14.1% | 15.3% | 12.2% |
| 2007 |  6.0% | 13.4% |  2.7% |
| 2008 | -1.0% |  0.1% | -0.3% |
| 2009 |  6.5% |  6.7% |  8.2% |
| 2010 |  8.2% | 10.4% | 11.4% |
| 2011 | -5.6% | -6.3% | -2.6% |
| 2012 |  9.4% |  6.8% |  6.8% |
| 2013 | 16.7% |  9.9% | 28.3% |
| 2014 |  6.3% |  4.3% |  2.2% |
| 2015 | -4.0% | -3.7% | -4.3% |
| 2016 | -0.5% | -1.5% |  2.7% |
| 2017 | 21.7% | 17.7% | 25.9% |
| 2018 | -9.8% | -8.8% | -9.7% |
| 2019 |  1.7% |  7.3% |  6.0% |
| 2020 |  8.0% |  4.0% | -0.2% |
| 2021 |  5.0% |  9.7% |  5.5% |
| 2022 | -0.9% |  1.1% |  6.1% |
| 2023 |  2.5% | -1.1% | -2.5% |
| 2024 |  2.3% | -0.0% | -5.8% |
| 2025 | 17.2% |  7.0% | 22.8% |
| 2026 |  2.0% |  2.0% |  0.8% |

Note the corrected backtest has **multiple negative years** — the previous "zero negative years" claim was part of the same artifact.

### Nordnet Live Execution

**Glidende stop loss** (trailing stop loss) is supported for US-listed ETFs on Nordnet.

Each weekly rebalance:
1. Execute buy/sell orders to reach target weights.
2. For each new or changed position, set a glidende stop loss at the stop% shown in output.
3. **Do not reset stops weekly** — let them trail naturally from original placement. The backtest uses an ~86-day rolling peak; resetting weekly creates a 1-week effective window, far too tight, causing false exits.
4. Only cancel and reset a stop when: (a) position is closed, (b) stop% changes by >3pp, or (c) order approaches the 30-day validity limit.

`python main.py weights sector2e` prints the exact stop% and stop price for each held position.

**UCITS retail (Norway):** US ETFs are blocked for retail under PRIIPs. Use the UCITS-tradeable subset:

```bash
python main.py weights sector2f --ucits   # positions + UCITS tickers + ISINs for Nordnet
```

The US→UCITS mapping lives in `configs/ucits_mapping.py` (direct / proxy / none status per ticker).

### Sector CLI

```bash
python main.py weights sector2e            # today's positions + stop prices
python main.py weights sector2f --ucits    # UCITS tickers + ISINs
python main.py backtest sector2e --best    # full backtest + charts
python main.py backtest sector2 --best --stop-freq weekly   # stop-frequency sweep (sector2 only)
python main.py optimize sector2e --trials 300

# Comparison charts
python main.py compare-sector              # V2 / V2b / V2c → sector_comparison.png
python main.py v2c-long                    # V2c + V2d extended history → v2c_extended.png
python main.py v2d-v2e                     # V2d vs V2e → v2d_v2e.png
python main.py v2c-v2d-v2e                 # three-way comparison → v2c_v2d_v2e.png
```

---

## Bond + Commodities Rotation Strategy (V1/V2/V3)

Systematic macro-driven rotation across fixed income, commodity, and satellite ETFs. Runs monthly with daily trailing stops. Three versions share the same codebase — pass `v2` or `v3` as the strategy argument to switch.

Honest performance is in the [summary table above](#honest-performance-summary): all three deliver ~2.7–2.9% CAGR, Sharpe 0.77–0.88, max drawdown around -11% to -13% over 2003–2026. V3 has the best risk-adjusted profile (Sharpe 0.88, worst month only -2.3%).

### Honest year-by-year returns (Bond, optimised params, cash on stop-outs)

| Year | V1 | V2 | V3 |
|------|------|------|------|
| 2008 |  7.4% | 10.8% |  —    |
| 2009 |  1.7% |  0.4% |  —    |
| 2010 |  9.5% | 10.1% | 10.9% |
| 2011 |  6.0% |  5.4% | 11.5% |
| 2012 |  3.2% |  4.4% |  3.0% |
| 2013 | -0.9% |  1.9% |  1.7% |
| 2014 |  2.9% |  4.7% |  4.6% |
| 2015 | -1.6% | -2.2% | -2.3% |
| 2016 |  6.1% |  4.7% |  5.2% |
| 2017 |  5.3% |  5.1% |  6.0% |
| 2018 | -0.3% | -1.1% | -1.7% |
| 2019 |  7.6% |  9.1% |  8.2% |
| 2020 |  0.4% |  1.4% |  5.0% |
| 2021 |  1.3% |  4.1% | -1.4% |
| 2022 | -9.8% |-12.4% | -2.3% |
| 2023 |  8.5% |  6.2% |  4.5% |
| 2024 |  8.7% |  7.7% |  7.6% |
| 2025 |  8.1% |  9.1% |  7.8% |
| 2026 |  1.7% |  1.8% |  2.2% |

(V3 has insufficient history before 2010; `—` marks warmup.)

### ETF Universe (Bond strategies)

| Bucket | V1 ETFs | V2 additions | Role |
|--------|---------|--------------|------|
| Duration | TLT, IEF, SHY | — | Defensive anchor / cash pool |
| Inflation | TIP | VTIP | Inflation hedge |
| Credit | LQD, HYG, ANGL, SJNK, BKLN, EMB, PFF | — | Spread income |
| Commodities | GLD, PDBC, DBA | SLV | Primary alpha source |
| Real Assets | — | VNQ | V2 REIT satellite |
| Equity | — | SPY | V2 growth regime satellite |

### How It Works

**Signals (FRED data, updated daily)** — three composite z-scores drive allocation:

**`duration_z`** — positive → favour TLT/IEF: 2s10s slope (20%), 10Y-3M spread (20%), Fed funds direction (15%), 10Y real yield DFII10 (25%), unemployment/Sahm (10%), ISM PMI / industrial production (10%).

**`credit_z`** — positive → favour credit ETFs: HY OAS inverted (35%), IG spread momentum (15%), VIX regime (20%), Fed balance sheet QE/QT (15%), TED spread (15%, discontinued 2023).

**`inflation_z`** — positive → favour TIP/VTIP: 10Y breakeven ROC (50%), CPI YoY momentum (50%).

**`usd_z`** (V2 only) — rising USD dampens commodity allocation.

### Risk Management

**Per-position trailing stops (daily)** — commodity/satellite ETFs exit when price drops below the rolling peak by the configured stop%; freed weight moves to SHY. Trigger is lagged one day (no look-ahead).

**Drawdown overlay** — when portfolio drawdown exceeds threshold AND yesterday was negative, exposure scales down; re-enters fully the next positive day.

**Volatility targeting** — daily scaling to a target vol, leverage capped:
- V1: 15% vol target, 1.75× cap
- V2: 12% vol target, 1.75× cap
- V3: 12% vol target, 1.50× cap

---

## Setup

```bash
pip install -r requirements.txt
export FRED_API_KEY=your_key_here   # free key at fred.stlouisfed.org/docs/api/api_key.html
```

Optionally set `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` to read/write market data from the `magnus-trading` Supabase project instead of yfinance/FRED (see CLAUDE.md → Data Pipeline).

## Usage

Strategy is always a positional argument. Default is `v1` when omitted. Choices: `v1`, `v2`, `v3`, `sector`, `sector2`, `sector2b`, `sector2c`, `sector2d`, `sector2e`, `sector2f`, `sector2g`, `sector2h`.

```bash
# Sector strategies (no FRED key needed)
python main.py weights sector2e            # today's positions + stop prices
python main.py weights sector2f --ucits    # UCITS tickers + ISINs
python main.py backtest sector2e --best    # full backtest + charts
python main.py optimize sector2e --trials 300

# Bond strategies
python main.py fetch v1
python main.py backtest v1 --best
python main.py weights v1
python main.py weights v2
python main.py weights v3

# Optional: stop-frequency sweep (sector2 only)
python main.py backtest sector2 --best --stop-freq weekly   # daily | weekly | monthly

# Comparison charts
python main.py compare                      # Bond V1/V2/V3 → backtest_comparison.png
python main.py compare-sector               # Sector V2/V2b/V2c → sector_comparison.png
python main.py v2c-long                     # V2c + V2d extended history → v2c_extended.png
python main.py v2d-v2e                      # V2d vs V2e → v2d_v2e.png
python main.py v2c-v2d-v2e                  # three-way comparison → v2c_v2d_v2e.png
```

The `weights` command is the production entry point. It shows both raw signal weights and trailing-stop-adjusted effective positions, plus the stop% and stop price for each held position.

## Re-optimising

```bash
python main.py optimize sector2e --trials 300
python main.py optimize sector2g --trials 300
python main.py optimize v1 --trials 500
```

Best parameters save to `best_params_sector2e.json` etc., loaded automatically by `weights` and `backtest --best`.

## Project Structure

```
bond-trading/
├── main.py                     # CLI entry point — strategy registry + all subcommands
├── optimize.py                 # Optuna optimisation (all strategies)
├── configs/
│   ├── bond_base.py            # Shared bond defaults + BASE_PARAM_SPACE
│   ├── bond_v1.py / bond_v2.py / bond_v3.py
│   ├── sector_base.py          # Shared sector defaults + BASE_PARAM_SPACE
│   ├── sector_v1.py … sector_v2e.py
│   ├── sector_v2f.py           # UCITS-tradeable subset of V2e
│   ├── sector_v2g.py           # Honest rebuild: MA-break exit, concentrated
│   ├── sector_v2h.py           # Fixed trailing stop, filled at stop price (gap-aware)
│   └── ucits_mapping.py        # US ETF → UCITS ticker/ISIN map for Nordnet
├── strategies/
│   ├── backtest_core.py        # Shared: vol_scale, DD overlay, trailing stops (one-day-lagged trigger)
│   ├── bond_shared.py          # Shared bond building blocks
│   ├── bond_v1/ bond_v2/ bond_v3/
│   ├── sector_shared/          # Shared sector building blocks
│   └── sector_v1/ … sector_v2h/
├── data/
│   ├── fred_client.py          # FRED API + caching
│   ├── price_client.py         # Yahoo Finance / Supabase ETF prices + caching
│   ├── backfill.py             # One-time historical backfill → Supabase
│   ├── pipelines/              # One module per strategy: load_all()
│   └── cache/                  # git-ignored; populate with python main.py fetch
├── analysis/
│   ├── metrics.py              # summary() + ratio helpers
│   └── performance.py          # Chart functions
├── scripts/                    # data_pipeline.py, UCITS what-if backtests, optimisers
├── db/schema.sql               # Supabase schema
├── best_params*.json           # Optimised parameters per strategy
└── logs/                       # Per-strategy log files
```

## Integrity Notes

This project shipped, then corrected, two classic backtest bugs. Both are documented in the `quant-backtest-integrity` skill and enforced as a review gate:

- **Look-ahead in the trailing stop** (PR #6) — the position was zeroed on the day the stop triggered, dodging that day's loss. Fixed by lagging the trigger one day (`backtest_core.py`). This alone accounted for the bulk of the inflated CAGR.
- **Unrealistic fill assumption** (PR #8) — stop value depends almost entirely on the assumed fill price. The fill-at-stop execution model (with slippage + gap risk) is the realistic basis; fill-at-close makes stops look useless, clean-fill-at-stop overstates them.

**Any reported performance number must come from the current engine.** If a change makes CAGR jump back toward 20–40% or Sharpe above ~1, assume a bug has been reintroduced and investigate before trusting it.
