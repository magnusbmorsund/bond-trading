# Bond + Commodities Rotation Strategy

Systematic macro-driven rotation across fixed income and commodity ETFs, targeting 10%+ annual return with max drawdown below 10%. Runs monthly with daily trailing stops.

## Performance (2005–2026 backtest)

| Metric          | Strategy |
|----------------|----------|
| Ann. Return     | 11.3%    |
| Ann. Volatility | 5.2%     |
| Sharpe Ratio    | 2.08     |
| Max Drawdown    | -8.6%    |
| Calmar Ratio    | 1.32     |
| Worst Month     | -4.3%    |
| Winning Years   | 21 / 22  |

Notable years: 2008 +30.8%, 2020 +20.8%, 2022 -5.0% (vs TLT -31%), 2024 +20.2%.

## ETF Universe

| Bucket     | ETFs                                      | Role |
|-----------|-------------------------------------------|------|
| Duration   | TLT, IEF, SHY                             | Defensive anchor / cash pool |
| Inflation  | TIP                                        | Inflation hedge |
| Credit     | LQD, HYG, ANGL, SJNK, BKLN, EMB, PFF    | Spread income |
| Commodities| GLD, PDBC, DBA, DBB                       | Primary alpha source |

## How It Works

### Signals (FRED data, updated daily)

Three composite z-scores drive all allocation decisions:

**`duration_z`** — positive → favour TLT/IEF (long bonds)
- 2s10s yield curve slope (20%)
- 10Y-3M spread, better recession predictor (20%)
- Fed funds rate direction (15%)
- 10Y real yield (DFII10) — the key 2022 signal (25%)
- Unemployment trend / Sahm rule (10%)
- Industrial production deceleration (10%)

**`credit_z`** — positive → favour LQD/HYG (credit)
- HY OAS level, inverted (35%)
- IG spread momentum / widening speed (15%)
- VIX regime (20%)
- Fed balance sheet QE/QT (15%)
- TED spread financial stress (15%)

**`inflation_z`** — positive → favour TIP
- 10Y breakeven inflation ROC (50%)
- CPI YoY momentum (50%)

### Allocation

1. **Commodity basket** — size grows with `0.5×inflation_z + 0.5×duration_z`; max 40% of portfolio. Within budget, each ETF is inverse-vol weighted, gated by 12-1 month momentum.
2. **Credit** — scales with `credit_z`, hard-capped at 50% and further capped when VIX > 25.
3. **TIP** — scales with `inflation_z`, max 15%.
4. **Duration** — remainder. TLT/IEF/SHY proportions set by softmax on `duration_z`.
5. **Momentum filter** — any ETF with negative 12-1 month momentum gets zeroed; freed weight goes to SHY.

### Risk Management

**Per-position trailing stops (daily)** — each commodity ETF exits if its price drops >4% below its 21-day rolling peak. Freed weight moves to SHY. This is the key edge: macro signals drive monthly entry; price-driven stops drive daily exit.

**Drawdown overlay** — when the portfolio is in drawdown beyond -5% AND yesterday was negative, exposure scales to 0% (full cash). Re-enters fully the next positive day (rides upswings). Cash earns the fed funds rate.

**Volatility targeting** — daily scaling so realised vol tracks 8% target, leverage capped at 1.5×.

## Setup

```bash
pip install -r requirements.txt
export FRED_API_KEY=your_key_here   # get free key at fred.stlouisfed.org/docs/api/api_key.html
```

## Usage

```bash
# Refresh all data
python main.py fetch

# Run backtest with default params
python main.py backtest

# Run backtest with saved best params
python main.py backtest --best

# Get today's IBKR positions (trailing-stop adjusted, production config)
python main.py weights

# Run Optuna hyperparameter optimisation (300 trials by default)
python main.py optimize --trials 300
```

The `weights` command is the production entry point. Run it each month-end (or intraday) to get the exact position sizes to set in IBKR. It shows both the raw signal weights and the trailing-stop-adjusted effective positions.

## Monthly Workflow

1. `python main.py fetch` — refresh FRED + price data
2. `python main.py weights` — read the "EFFECTIVE POSITIONS" table
3. Set each ETF to the shown % of total portfolio value in IBKR
4. Note any `[STOPPED OUT]` ETFs — these positions should be flat

## Re-optimising

```bash
python main.py optimize --trials 500
```

Best parameters are saved to `best_params.json` and loaded automatically by `main.py weights`. Re-run when macro regime shifts substantially (e.g., new Fed cycle) or after 12+ months of live trading.

## Project Structure

```
bond-trading/
├── config.py               # All parameters — edit here or via best_params.json
├── main.py                 # CLI entry point
├── optimize.py             # Optuna hyperparameter search
├── best_params.json        # Production parameters (loaded by 'weights' command)
├── strategy/
│   ├── signals.py          # Macro signal computation (FRED data → z-scores)
│   ├── portfolio.py        # Signal → target weights
│   └── backtest.py         # Backtest engine + trailing stops + vol targeting
├── data/
│   ├── pipeline.py         # Orchestrates all data loading
│   ├── fred_client.py      # FRED API + caching
│   └── price_client.py     # Yahoo Finance ETF prices + caching
└── analysis/
    └── performance.py      # Metrics + charts
```

## Key Config Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_ALT_ALLOC` | 40% | Max commodity basket allocation |
| `TRAILING_STOP_PCT` | 4% | Exit commodity if price drops this far from peak |
| `TRAILING_STOP_WINDOW` | 21 days | Rolling peak lookback for trailing stop |
| `VOL_TARGET` | 8% | Portfolio volatility target |
| `MAX_LEVERAGE` | 1.5× | Max daily vol-scaling leverage |
| `DD_THRESHOLD` | -5% | Drawdown level that triggers cash-out |
| `DD_SCALE` | 0% | Exposure kept during drawdown (0 = full exit) |
