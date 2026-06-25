# Commodity Supercycle — design & honest numbers

A commodity-complex momentum-rotation variant (`commodity`) that rides multi-year
commodity up-legs ("supercycles") and scales exposure with a FRED macro overlay.

```bash
python main.py backtest commodity --best     # full backtest (after optimize)
python main.py optimize commodity --trials 250
python main.py weights  commodity             # today's positions
```

## Design

- **Universe (commodity complexes only, 20 ETFs):** broad baskets (PDBC, DBC, GSG);
  energy (USO, UNG, XOP, OIH); precious (GLD, SLV, GDX, GDXJ, SIL); base/industrial
  (DBB, CPER, XME, COPX); agriculture (DBA); energy-transition (LIT, URA, REMX).
  Cluster caps stop the book loading several correlated sub-baskets at once.
- **Signal:** the V2e multi-timescale momentum composite with the 24m/36m
  "supercycle" lookbacks active (NaN-safe blend). `ONLY_POSITIVE_COMPOSITE` ⇒ a
  complex is held only while its own trend is up (TS-momentum filter); otherwise cash.
- **Equity-decoupled:** the SPY-trend regime gate is **neutralised**
  (`signals.spy_regime` returns 1.0) so the sleeve stays a genuine equity
  diversifier — commodities can rally when stocks fall (2022).
- **Macro exposure overlay (the new piece):** a daily multiplier in
  `[MACRO_FLOOR, 1.0]` from a small FRED z-score composite — **−USD momentum**
  (DTWEXBGS), **−real-yield momentum** (DFII10), **+inflation/industrial momentum**
  (CPIAUCSL, INDPRO). Positive ⇒ commodity-friendly backdrop. Applied as the
  outermost exposure scaler, publication-lagged + shifted so day-T uses only prior
  macro state. With `MACRO_CEIL = 1.0` it can only **de-risk** — it cannot
  manufacture return, so it cannot inflate the Sharpe. `MACRO_FLOOR = 1.0` would
  disable it (the optimiser is free to find it adds nothing — and it does).
- **No per-position trailing stops** (`TRAILING_STOP_ETFS = []`); defense is the
  momentum re-rank + macro overlay + vol-target + drawdown overlay.

## Honest numbers (2010-01 → 2026-06, net of the Saxo cost model, SHY earns 0%)

> **Read this before citing any figure.** The 70/30 split is **2010–2021 train /
> 2021–2026 test**, and that test window is a single commodity regime (the
> 2021–22 spike + 2023–25 recovery). It is a *peeked* validation window, not clean
> forward evidence — treat its Sharpe as an optimistic upper bound, exactly as the
> repo's other sector variants are flagged.

| Cut | Sharpe | Ann. ret | Max DD | Note |
|---|---|---|---|---|
| **Train 2010–2021** (optimised) | **0.08** | 0.3% | −16.7% | **Dead** — the 2000s supercycle ended ~2011; there was no supercycle in-sample to catch |
| Test 2021–26 (optimised, isolated slice) | 0.56 | 3.7% | −13.9% | optimiser re-runs on the slice → 3y of NaN momentum warmup drags it |
| Test 2021–26 (full-warmup slice) | ~1.0 | 7–10% | −7…12% | representative of live (always has history); **one regime only** |
| Full 2010–26, default params | 0.31 | 2.5% | −43.6% | vs broad-commodity buy&hold **0.12 / 0.6% / −70.9%** |

**Per-year (default params, full period):** positive in 8/17 years. It **lagged the
actual 2021–22 spike badly** (+7.0% / +6.8% vs buy&hold +40.6% / +21.0%) — momentum
is laggy — but was genuinely strong in **2023 +20.3% / 2024 +10.9% / 2025 +22.5%**
while buy&hold was flat/negative. Returns are **not** concentrated in 2021–22
(ex-2021/22 total +31%).

## Verdict — research-grade, NOT promoted

1. **It is a sound *defensive* commodity sleeve, not a proven alpha engine.** It
   roughly halves the drawdown of holding commodities (−44% vs −71%) and doubles
   buy&hold's Sharpe — but in absolute terms ~0.3 full-period Sharpe is weak, in
   line with the repo's honest sector-momentum band (0.4–0.9) and the related
   `magnus-trading` graveyard (supercycle/sector rotation ≈ 0.15).
2. **There was essentially no supercycle in the sample.** On the 2010–2021 train it
   earns ~0% (Sharpe 0.08). The only strong stretch is the post-2021 commodity
   regime, i.e. the peeked window.
3. **Optimisation overfits.** The un-optimised defaults *beat* the optimised params
   out-of-sample — the classic signature of no robust edge to fit (the dead decade
   gave the optimiser only noise to chase).
4. **The macro overlay is inert** here (+~0.02 Sharpe; the optimiser damped it to
   near-off, `MACRO_GAIN≈0.13`, `MACRO_FLOOR≈0.44`). Consistent with the literature:
   factor/macro timing is "deceptively difficult."

**Conclusion:** keep as a documented research variant. Don't promote to a live
sleeve on the strength of the 2021–26 window, and don't read the ~1.0 OOS Sharpe as
forward-looking. The defensive structure is the real, repeatable part; the return
engine is waiting for a supercycle that didn't occur in the sample.
