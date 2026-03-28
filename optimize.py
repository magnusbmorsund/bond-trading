"""
Optuna hyperparameter optimization for the bond rotation strategy.

Splits data 70/30 (train/test). Optimises Sharpe on the training window,
then evaluates best params on the held-out test window.

Usage:
    python optimize.py [--trials N]   (default 300)
"""
import argparse
import warnings
import json
import os

import optuna
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

import config
from data.pipeline       import load_all
from strategy.backtest   import run
from analysis.performance import sharpe, max_drawdown, summary


# ---------------------------------------------------------------------------
# Parameter space
# ---------------------------------------------------------------------------

PARAM_SPACE = {
    # Lookback windows
    "LOOKBACK_SIGNAL":  ("int",   126, 504, 21),
    "LOOKBACK_VOL":     ("int",    21, 126, 21),
    "MOMENTUM_WINDOW":  ("int",   126, 504, 21),
    "MOMENTUM_SKIP":    ("int",     0,  42,  5),
    # Allocation caps — credit can go higher with expanded universe
    "MAX_CREDIT_ALLOC": ("float", 0.20, 0.70, 0.05),
    "MAX_TIP_ALLOC":    ("float", 0.00, 0.30, 0.05),
    "SIGNAL_BLEND":     ("float", 0.00, 1.00, 0.10),
    # Volatility targeting — wider range to reach 5%+ returns
    "VOL_TARGET":       ("float", 0.05, 0.12, 0.01),
    "MAX_LEVERAGE":     ("float", 1.00, 2.00, 0.25),
    # VIX thresholds
    "VIX_RISK_OFF":     ("float", 18.0, 35.0, 1.0),
    "VIX_RISK_ON":      ("float", 10.0, 20.0, 1.0),
    # Composite signal weights
    "W_DURATION_2S10S": ("float", 0.10, 0.60, 0.10),
    "W_DURATION_10Y3M": ("float", 0.10, 0.60, 0.10),
    "W_DURATION_FED":   ("float", 0.10, 0.60, 0.10),
    "W_CREDIT_HYOAS":   ("float", 0.30, 0.80, 0.10),
    "W_CREDIT_VIX":     ("float", 0.20, 0.70, 0.10),
    "W_INFLATION_BEI":  ("float", 0.20, 0.80, 0.10),
    "W_INFLATION_CPI":  ("float", 0.20, 0.80, 0.10),
}

BEST_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "best_params.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_params(params: dict):
    """Monkey-patch config module so signals/portfolio pick up new values."""
    for k, v in params.items():
        setattr(config, k, v)


def _restore_defaults():
    """Restore config to module defaults (re-import)."""
    import importlib
    importlib.reload(config)


def _suggest_params(trial) -> dict:
    params = {}
    for name, spec in PARAM_SPACE.items():
        kind = spec[0]
        if kind == "int":
            _, lo, hi, step = spec
            params[name] = trial.suggest_int(name, lo, hi, step=step)
        else:
            _, lo, hi, step = spec
            params[name] = trial.suggest_float(name, lo, hi, step=step)
    return params


def _run_on_slice(macro: pd.DataFrame, prices: pd.DataFrame) -> dict:
    """Run backtest on a data slice; return None on failure."""
    try:
        return run(macro, prices)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

RETURN_TARGET = 0.05   # 5% annualised — penalise strategies below this

def make_objective(macro_train, prices_train):
    def objective(trial):
        params = _suggest_params(trial)
        _apply_params(params)

        results = _run_on_slice(macro_train, prices_train)
        if results is None or len(results["daily_returns"].dropna()) < 252:
            return -10.0

        ret = results["daily_returns"].dropna()
        nav = results["nav"]
        sr  = sharpe(ret)
        mdd = max_drawdown(nav)

        n       = len(ret)
        ann_ret = float(nav.iloc[-1] ** (252 / n) - 1)

        # Penalise drawdowns beyond 15%
        dd_penalty     = max(0.0, abs(mdd) - 0.15) * 4.0
        # Penalise annualised returns below 5% target
        return_penalty = max(0.0, RETURN_TARGET - ann_ret) * 5.0

        return sr - dd_penalty - return_penalty

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_optimization(n_trials: int = 300):
    print("Loading data...")
    macro, prices = load_all()

    # ── Train / test split ─────────────────────────────────────────────────
    split = int(len(macro) * 0.70)
    macro_train,  prices_train  = macro.iloc[:split],  prices.iloc[:split]
    macro_test,   prices_test   = macro.iloc[split:],  prices.iloc[split:]

    train_end = macro_train.index[-1].date()
    test_start = macro_test.index[0].date()
    print(f"Train: {macro_train.index[0].date()} → {train_end}")
    print(f"Test : {test_start} → {macro_test.index[-1].date()}")
    print(f"Running {n_trials} Optuna trials...\n")

    # ── Optimise ───────────────────────────────────────────────────────────
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(make_objective(macro_train, prices_train),
                   n_trials=n_trials,
                   show_progress_bar=True)

    best_params = study.best_params
    best_val    = study.best_value
    print(f"\nBest train objective (Sharpe - penalty): {best_val:.3f}")

    # ── Evaluate on test set ───────────────────────────────────────────────
    _apply_params(best_params)
    res_train = run(macro_train, prices_train)
    res_test  = run(macro_test,  prices_test)
    _restore_defaults()

    s_train = summary(res_train["daily_returns"], res_train["nav"], "Train")
    s_test  = summary(res_test["daily_returns"],  res_test["nav"],  "Test (OOS)")

    tbl = pd.concat([s_train, s_test], axis=1)
    print("\n" + "=" * 52)
    print("OPTIMISED STRATEGY — TRAIN vs OUT-OF-SAMPLE")
    print("=" * 52)
    print(tbl.to_string())
    print("=" * 52)

    # ── Baseline comparison ────────────────────────────────────────────────
    _restore_defaults()
    res_base_test = run(macro_test, prices_test)
    s_base = summary(res_base_test["daily_returns"], res_base_test["nav"], "Default (OOS)")
    print("\nDefault params on same test period:")
    print(pd.concat([s_test, s_base], axis=1).to_string())

    # ── Print and save best params ─────────────────────────────────────────
    print("\nBest parameters:")
    for k, v in best_params.items():
        default_val = getattr(config, k)
        print(f"  {k:<22s} {v!s:>8}   (default: {default_val})")

    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\nSaved → {BEST_PARAMS_PATH}")
    print("To use: set these values in config.py, or load via load_best_params().")

    return best_params


def load_best_params() -> dict:
    """Load previously saved best params (for use in main.py)."""
    if not os.path.exists(BEST_PARAMS_PATH):
        raise FileNotFoundError("No best_params.json found — run optimize first.")
    with open(BEST_PARAMS_PATH) as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=300)
    args = parser.parse_args()
    run_optimization(n_trials=args.trials)
