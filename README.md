# Bond + Commodities Rotation Strategy

Systematic macro-driven rotation across fixed income, commodity, and satellite ETFs. Targets strong risk-adjusted returns with max drawdown below 10%. Runs monthly with daily trailing stops.

Two strategy versions share the same codebase — add `--v2` to any command to switch.

## Performance (2005–2026 backtest, optimised params)

| Metric          | V1       | V2       |
|----------------|----------|----------|
| Ann. Return     | 23.6%    | 24.9%    |
| Ann. Volatility | 7.0%     | 6.8%     |
| Sharpe Ratio    | 3.10     | 3.32     |
| Max Drawdown    | -7.2%    | -9.4%    |
| Calmar Ratio    | 3.27     | 2.66     |
| Worst Month     | -4.6%    | -4.2%    |
| Turnover (avg)  | 16%/mo   | 21%/mo   |

V1 has the edge on drawdown control (Calmar 3.27). V2 has the edge on raw return and Sharpe. Both ran on the same 2005–2026 period with a 70/30 train/test split; OOS Sharpe was 3.38 for V2.

## ETF Universe

| Bucket       | V1 ETFs                                | V2 additions       | Role |
|-------------|----------------------------------------|-------------------|------|
| Duration     | TLT, IEF, SHY                          | —                 | Defensive anchor / cash pool |
| Inflation    | TIP                                    | VTIP              | Inflation hedge (V2 splits by duration risk) |
| Credit       | LQD, HYG, ANGL, SJNK, BKLN, EMB, PFF | —                 | Spread income |
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

**Per-position trailing stops (daily)** — each commodity ETF exits if its price drops >3% below its 21-day rolling peak. Freed weight moves to SHY. Key edge: macro signals drive monthly entry; price-driven stops drive daily exit.

**Drawdown overlay** — when portfolio drawdown exceeds -10% AND yesterday was negative, exposure scales to 45% (V2) / 0% (V1 default). Re-enters fully the next positive day. Cash earns the fed funds rate.

**Volatility targeting** — daily scaling so realised vol tracks 11% (V2) / 8% (V1) target, leverage capped at 1.75×.

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
```

Best parameters save to `best_params.json` (V1) and `best_params_v2.json` (V2), loaded automatically by `weights`. Re-run when macro regime shifts substantially or after 12+ months of live trading.

## Project Structure

```
bond-trading/
├── config.py               # V1 parameters
├── config_v2.py            # V2 parameters (SLV, VTIP, VNQ, SPY, USD signal)
├── main.py                 # CLI entry point (--v2 flag switches strategy)
├── optimize.py             # Optuna optimisation (supports --v2)
├── best_params.json        # V1 production parameters
├── best_params_v2.json     # V2 production parameters
├── strategy/               # V1 strategy modules
│   ├── signals.py
│   ├── portfolio.py
│   └── backtest.py
├── strategy_v2/            # V2 strategy modules
│   ├── signals.py          # + USD signal (DTWEXBGS), ISM fallback
│   ├── portfolio.py        # + equity satellite, VNQ, VTIP/TIP split
│   └── backtest.py
├── data/
│   ├── pipeline.py         # V1 data loading
│   ├── pipeline_v2.py      # V2 data loading (etf_prices_v2.csv cache)
│   ├── fred_client.py      # FRED API + caching
│   └── price_client.py     # Yahoo Finance ETF prices + caching
└── analysis/
    └── performance.py      # Metrics + charts
```

## Key Config Parameters

| Parameter | V1 (optimised) | V2 (optimised) | Description |
|-----------|---------------|---------------|-------------|
| `MAX_ALT_ALLOC` | 60% | 60% | Max commodity basket allocation |
| `TRAILING_STOP_PCT` | 3% | 3% | Exit commodity if price drops this far from peak |
| `TRAILING_STOP_WINDOW` | 21 days | 21 days | Rolling peak lookback for trailing stop |
| `VOL_TARGET` | 11% | 11% | Portfolio volatility target |
| `MAX_LEVERAGE` | 1.75× | 1.75× | Max daily vol-scaling leverage |
| `DD_THRESHOLD` | -10% | -10% | Drawdown level that triggers scaling |
| `DD_SCALE` | 45% | 45% | Exposure kept during drawdown |
| `MAX_EQUITY_ALLOC` | — | 15% | V2 equity satellite cap |
| `MAX_REALESTATE_ALLOC` | — | 5% | V2 VNQ allocation cap |
| `W_COMMODITY_USD` | — | 0.30 | V2 USD drag weight on commodity budget |
| `VTIP_DURATION_SCALE` | — | 1.0 | V2 sensitivity of TIP/VTIP split |
