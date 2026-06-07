---
name: quant-backtest-integrity
description: >
  MANDATORY review gate when building, modifying, or reporting any trading
  strategy / backtest in this repo (signals, portfolio, stops/exits, vol
  targeting, optimization, or quoting CAGR/Sharpe/DD). Use whenever editing
  strategies/**, configs/sector_v*.py, configs/bond_v*.py, optimize.py,
  daily_weights.py, or before stating any performance number. Encodes the
  look-ahead / fill-assumption / overfit failures that made sector2f look like
  38% CAGR / Sharpe 3.25 when the honest figure is ~4.7% / 0.64.
---

# Quant backtest integrity

A backtest bug here is not a cosmetic error — it sizes real capital. `sector2f`
reported **38% CAGR / Sharpe 3.25**; the honest figure after fixing one
look-ahead line was **~4.7% / 0.64**. Trading on the inflated number would have
meant over-sizing into a strategy that delivers a fraction of the return with
**deeper** drawdowns than the backtest showed. Treat every rule below as
capital-preservation, not academic nicety.

## The smell test — STOP and audit if you see any of these
On liquid ETFs / 2000–2026, honest results live around **Sharpe 0.4–0.9, CAGR
3–8%, max DD −15% to −35%**. If a backtest shows any of:
- **Sharpe > ~1.5**, **CAGR > ~20%**, or **max DD better than ~−10%**
- an equity curve that **dodges almost every down month**
- OOS performance **≥** in-sample
…assume a bug (look-ahead, fill assumption, survivorship, or overfit) until
proven otherwise. Do **not** report it as real.

## 1. Look-ahead in stops/exits (the sector2f bug — check FIRST)
A trigger computed from day-T's close must **never** affect day-T's return.
- The exit/stop mask must be **lagged**: `triggered = triggered.shift(1)`. The
  position is held *through* the breach day and goes flat from T+1.
- Rolling peaks/MAs used in triggers must be `.shift(1)` (peak through T-1).
- Signal weights must be `.shift(1)` (trade on yesterday's signal).
- **Verify**: zeroing a weight on the same bar whose return you then multiply by
  it = look-ahead. If `daily_w[T]` depends on `price[T]` and earns
  `daily_ret[T]`, it's wrong.
- Quick test: lag the trigger and re-run. If CAGR/Sharpe collapse, the headline
  was the bug.

## 2. Fill assumptions are first-class — model them, conservatively
Where an exit is *booked* swings Sharpe from ~0.6 to ~1.9 on the same strategy:
- `fill=close` (exit at the close below the stop) = pessimistic.
- `fill=stop` (exit at the stop level) = realistic only for liquid names, ex-gaps.
- Never assume frictionless fills. Always include **slippage** and a **gap
  model** (a fraction of triggers fill materially worse than the stop).
- Backtest must mirror live execution: Nordnet *glidende* stops are **not
  guaranteed**, fill **intraday** (not at the prior close), and gap on opens.
- Validate the fill assumption against **real fills** (see analysis/stop_tracker.py
  + `python main.py stop-report`) before trusting tight-stop results.

## 3. Don't let the optimizer game an assumption
Optuna will drive any knob that exploits a modeling assumption to its corner and
print a fake Sharpe 3+. Seen this session: tight stop → floor + max leverage
under `fill=stop`.
- **Never optimize execution-realism knobs** (slippage, gap, fill mode) or
  **leverage** — fix them as conservative assumptions.
- Treat any best-param **hitting a search-space bound** as a red flag, not a win.
- Keep parameter ranges tight and few; ~7 risk params over ~6–8 independent
  crashes is overfitting territory.

## 4. Always report out-of-sample, never full-period-optimized
- Optimize on train only; report the **held-out** test window separately
  (`optimize.py` does 70/30 — use it). A number that exists only in-sample is
  fiction.
- Quote the OOS figure as the headline. State train vs OOS explicitly.

## 5. When refactoring a shared engine, prove legacy is unchanged
- New behaviour must be **config-gated and default-off**; existing strategies
  must be byte-identical. Verify (e.g. sector2f stays 4.7%/0.64) before/after.

## 6. Re-derive performance after ANY change
After touching signals / portfolio / stops / vol-target / costs, re-run the
backtest and compare before/after. Update CLAUDE.md / memory if headline numbers
move. Never quote a stale or pre-change number.

## 7. Live-vs-backtest parity gaps to check
- `best_params*.json` are **git-ignored** → CI/live `daily_weights.py` may run on
  **config defaults**, not optimized params. Confirm which params actually trade.
- The live `effective_weights` path and the backtest must use the same stop
  logic and lag.

## Reference (this repo)
- Look-ahead bug + fix and honest numbers: memory `stop-lookahead-bias`.
- Fill-model research, corner-solution warning, slippage/gap sensitivity:
  memory `stop-fill-model`.
- Engine knobs: `EXIT_MODE`, `STOP_FILL`, `STOP_PERSIST`, `STOP_SLIP_BPS`,
  `GAP_FRAC`/`GAP_EXTRA` in strategies/sector_shared/backtest.py.
