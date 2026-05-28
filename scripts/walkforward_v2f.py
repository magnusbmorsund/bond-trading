"""
Walk-forward validation for sector_v2f.

Approach: run the FULL backtest 2000-2026 with each Optuna trial's params,
then mask the daily returns into training (most years) and 4 disjoint
out-of-sample test windows:

  Test windows:    2003-2005, 2008-2010, 2015-2017, 2022-2026
  Training years:  everything else  (2000-2002, 2006-2007, 2011-2014, 2018-2021)

Optuna optimizes on training-year returns only. After convergence, we report
performance on each test window separately to check the strategy holds up
out-of-sample. Best params saved to best_params_sector2f_walkforward.json
(separate from the V2e-derived best_params_sector2f.json — does not overwrite).

Why mask rather than slice the input prices: momentum lookbacks (12m/18m/24m/36m)
need continuous price history. Slicing breaks the warmup. We run the strategy
on the full timeline and just measure performance on the chosen segments.

Run:
    python scripts/walkforward_v2f.py [--trials 300]
"""
import argparse
import json
import warnings
import sys
import logging
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import configs.sector_v2f as cfg
from data.pipelines.sector_v2f import load_all
from strategies.sector_v2f.backtest import run as run_backtest
from analysis.performance import sharpe, max_drawdown, summary
from optimize import _suggest_params, _apply_params

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")

# ── Test windows (inclusive year ranges) ─────────────────────────────────────
TEST_WINDOWS = [
    ("2003-2005", 2003, 2005),
    ("2008-2010", 2008, 2010),
    ("2015-2017", 2015, 2017),
    ("2022-2026", 2022, 2026),
]

# ── Penalties (same shape as optimize.py _optimize_sector for sector2e) ──────
DD_THRESH  = 0.12
RET_THRESH = 0.15
WM_THRESH  = -0.07

BEST_PATH = ROOT / "best_params_sector2f_walkforward.json"


def is_test_year(year: int) -> bool:
    return any(lo <= year <= hi for _, lo, hi in TEST_WINDOWS)


def build_masks(index: pd.DatetimeIndex) -> tuple[pd.Series, dict[str, pd.Series]]:
    """Return (train_mask, {window_name: test_mask}) — bool Series aligned to index."""
    years = index.year
    train_mask = pd.Series([not is_test_year(y) for y in years], index=index)
    window_masks = {
        name: pd.Series([lo <= y <= hi for y in years], index=index)
        for name, lo, hi in TEST_WINDOWS
    }
    return train_mask, window_masks


def objective_from(ret: pd.Series) -> float:
    """Sharpe × ann_return × 10 − penalties (same shape as _optimize_sector)."""
    if ret.empty or len(ret) < 252:
        return -10.0
    nav = (1 + ret).cumprod()
    sr  = sharpe(ret)
    mdd = max_drawdown(nav)
    ann_ret = float(nav.iloc[-1] ** (252 / len(ret)) - 1)

    dd_penalty     = max(0.0, abs(mdd) - DD_THRESH) * 20.0
    return_penalty = max(0.0, RET_THRESH - ann_ret) * 4.0
    monthly_ret    = (1 + ret).resample("ME").prod() - 1
    wm_penalty     = max(0.0, WM_THRESH - float(monthly_ret.min())) * 8.0

    return sr * ann_ret * 10 - dd_penalty - return_penalty - wm_penalty


def run_with_params(params: dict, prices: pd.DataFrame) -> pd.Series | None:
    """Apply params, run backtest, return daily_returns (or None on failure)."""
    _apply_params(params, cfg)
    try:
        res = run_backtest(prices)
        return res["daily_returns"].dropna()
    except Exception as exc:
        logger.debug("Trial failed: %s", exc)
        return None


def main(n_trials: int = 300):
    print("=" * 78)
    print(f"WALK-FORWARD VALIDATION — sector V2f")
    print("=" * 78)
    print("Test windows (OOS):  " + ", ".join(name for name, *_ in TEST_WINDOWS))
    print("Training years:      everything else")
    print(f"Trials:              {n_trials}")
    print("=" * 78)

    print("\nLoading V2f prices...")
    prices = load_all()
    train_mask, window_masks = build_masks(prices.index)
    print(f"Price range:  {prices.index[0].date()} → {prices.index[-1].date()}")

    # ── Optuna study on training-year returns only ────────────────────────────
    def trial_fn(trial):
        params = _suggest_params(trial, cfg.PARAM_SPACE)
        ret = run_with_params(params, prices)
        if ret is None:
            return -10.0
        train_ret = ret[train_mask.reindex(ret.index, fill_value=False)]
        return objective_from(train_ret)

    print(f"\nRunning {n_trials} Optuna trials (training-year objective)...")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(trial_fn, n_trials=n_trials, show_progress_bar=True)
    best = study.best_params
    print(f"\nBest training objective: {study.best_value:.3f}")

    # ── Evaluate best params over full timeline ───────────────────────────────
    final_ret = run_with_params(best, prices)
    assert final_ret is not None, "Best params should produce valid returns"

    train_ret = final_ret[train_mask.reindex(final_ret.index, fill_value=False)]
    train_nav = (1 + train_ret).cumprod()

    print("\n" + "=" * 78)
    print("TRAINING PERFORMANCE  (objective Optuna saw)")
    print("=" * 78)
    print(summary(train_ret, train_nav, "Train"))

    # ── Per-window OOS performance ────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("OUT-OF-SAMPLE PERFORMANCE  (held out from optimization)")
    print("=" * 78)
    oos_rows = []
    for name, _, _ in TEST_WINDOWS:
        mask = window_masks[name].reindex(final_ret.index, fill_value=False)
        win_ret = final_ret[mask]
        if win_ret.empty:
            print(f"\n--- {name} (no data) ---")
            continue
        win_nav = (1 + win_ret).cumprod()
        n = len(win_ret)
        cagr = float(win_nav.iloc[-1] ** (252 / n) - 1)
        sr   = sharpe(win_ret)
        mdd  = max_drawdown(win_nav)
        total = float(win_nav.iloc[-1] - 1)
        oos_rows.append({
            "Window": name,
            "Days":    n,
            "CAGR":   f"{cagr:.1%}",
            "Sharpe": f"{sr:.2f}",
            "MaxDD":  f"{mdd:.1%}",
            "Total":  f"{total:.1%}",
        })

    oos_df = pd.DataFrame(oos_rows)
    print(oos_df.to_string(index=False))

    # ── Combined OOS across all test windows ──────────────────────────────────
    all_test_mask = pd.Series(False, index=final_ret.index)
    for m in window_masks.values():
        all_test_mask |= m.reindex(final_ret.index, fill_value=False)
    combined_ret = final_ret[all_test_mask]
    combined_nav = (1 + combined_ret).cumprod()
    print("\n--- Combined across all 4 OOS windows ---")
    print(summary(combined_ret, combined_nav, "OOS combined"))

    # ── Sanity: compare to V2e-inherited best params on same windows ──────────
    print("\n" + "=" * 78)
    print("COMPARISON: V2e-inherited params vs walk-forward-optimised params")
    print("=" * 78)
    inherited = json.loads((ROOT / "best_params_sector2f.json").read_text())
    inh_ret = run_with_params(inherited, prices)

    inh_rows = []
    for name, _, _ in TEST_WINDOWS:
        mask = window_masks[name].reindex(inh_ret.index, fill_value=False)
        wr = inh_ret[mask]
        if wr.empty:
            continue
        wn = (1 + wr).cumprod()
        n = len(wr)
        inh_rows.append({
            "Window": name,
            "CAGR":   f"{float(wn.iloc[-1] ** (252/n) - 1):.1%}",
            "Sharpe": f"{sharpe(wr):.2f}",
            "MaxDD":  f"{max_drawdown(wn):.1%}",
        })
    inh_df = pd.DataFrame(inh_rows)

    side = pd.DataFrame({
        "Window":      [r["Window"] for r in oos_rows],
        "WF CAGR":     [r["CAGR"]   for r in oos_rows],
        "Inh CAGR":    inh_df["CAGR"].tolist(),
        "WF Sharpe":   [r["Sharpe"] for r in oos_rows],
        "Inh Sharpe":  inh_df["Sharpe"].tolist(),
        "WF MaxDD":    [r["MaxDD"]  for r in oos_rows],
        "Inh MaxDD":   inh_df["MaxDD"].tolist(),
    })
    print("\nPer-window OOS  (WF=walk-forward params, Inh=V2e-inherited params):")
    print(side.to_string(index=False))

    inh_test_mask = pd.Series(False, index=inh_ret.index)
    for m in window_masks.values():
        inh_test_mask |= m.reindex(inh_ret.index, fill_value=False)
    inh_combined = inh_ret[inh_test_mask]
    inh_nav = (1 + inh_combined).cumprod()
    print("\nCombined OOS across all 4 windows:")
    print(pd.concat([
        summary(combined_ret, combined_nav, "WalkFwd"),
        summary(inh_combined, inh_nav,      "Inherited"),
    ], axis=1).to_string())

    # ── Save walk-forward best params ─────────────────────────────────────────
    BEST_PATH.write_text(json.dumps(best, indent=2))
    print(f"\nWalk-forward best params saved → {BEST_PATH}")
    print("\nWalk-forward params:")
    for k, v in best.items():
        inh_v = inherited.get(k, "—")
        print(f"  {k:<28s} {v!s:>14}   (V2e-inherited: {inh_v})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=300)
    args = parser.parse_args()
    main(args.trials)
