"""
Optuna hyperparameter optimization for the bond rotation strategy.

Splits data 70/30 (train/test). Optimises Sharpe on the training window,
then evaluates best params on the held-out test window.

Usage:
    python optimize.py [--trials N] [--v2]   (default 300 trials, v1 strategy)
"""
import argparse
import warnings
import logging
import json
import os
import importlib

import optuna
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

import config
from data.pipeline       import load_all
from strategy.backtest   import run
from analysis.performance import sharpe, max_drawdown, summary

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parameter spaces
# ---------------------------------------------------------------------------

PARAM_SPACE = {
    # Lookback windows
    "LOOKBACK_SIGNAL":  ("int",    84, 504, 21),
    "LOOKBACK_VOL":     ("int",    21, 126, 21),
    "MOMENTUM_WINDOW":  ("int",   126, 504, 21),
    "MOMENTUM_SKIP":    ("int",     0,  42,  5),
    # Allocation caps
    "MAX_CREDIT_ALLOC": ("float", 0.30, 0.80, 0.05),
    "MAX_TIP_ALLOC":    ("float", 0.00, 0.30, 0.05),
    "SIGNAL_BLEND":     ("float", 0.00, 1.00, 0.10),
    # Vol/leverage
    "VOL_TARGET":       ("float", 0.05, 0.15, 0.01),
    "MAX_LEVERAGE":     ("float", 1.00, 1.75, 0.25),
    # VIX thresholds
    "VIX_RISK_OFF":     ("float", 18.0, 40.0, 1.0),
    "VIX_RISK_ON":      ("float", 10.0, 22.0, 1.0),
    # Drawdown overlay
    "DD_THRESHOLD":     ("float", -0.15, -0.02, 0.01),
    "DD_SCALE":         ("float",  0.00,  0.50, 0.05),
    # Per-position trailing stops
    "TRAILING_STOP_PCT":    ("float", 0.03, 0.15, 0.01),
    "TRAILING_STOP_WINDOW": ("int",   21,  126,  21),
    # Commodity allocation budget
    "MAX_ALT_ALLOC":      ("float", 0.20, 0.60, 0.05),
    # Duration composite weights
    "W_DURATION_2S10S":   ("float", 0.05, 0.40, 0.05),
    "W_DURATION_10Y3M":   ("float", 0.05, 0.40, 0.05),
    "W_DURATION_FED":     ("float", 0.05, 0.30, 0.05),
    "W_DURATION_REALYLD": ("float", 0.10, 0.50, 0.05),
    "W_DURATION_LABOR":   ("float", 0.00, 0.25, 0.05),
    "W_DURATION_ISM":     ("float", 0.00, 0.25, 0.05),
    # Credit composite weights
    "W_CREDIT_HYOAS":     ("float", 0.15, 0.60, 0.05),
    "W_CREDIT_IGMOM":     ("float", 0.05, 0.35, 0.05),
    "W_CREDIT_VIX":       ("float", 0.10, 0.50, 0.05),
    "W_CREDIT_FEDQT":     ("float", 0.05, 0.35, 0.05),
    "W_CREDIT_TED":       ("float", 0.05, 0.35, 0.05),
    # Inflation composite weights
    "W_INFLATION_BEI":    ("float", 0.20, 0.80, 0.10),
    "W_INFLATION_CPI":    ("float", 0.20, 0.80, 0.10),
}

# Additional parameters only in v2 space
V2_PARAM_ADDITIONS = {
    "MAX_EQUITY_ALLOC":      ("float", 0.00, 0.30, 0.05),
    "MAX_REALESTATE_ALLOC":  ("float", 0.00, 0.15, 0.05),
    "W_COMMODITY_USD":       ("float", 0.00, 0.50, 0.05),
    "VTIP_DURATION_SCALE":   ("float", 0.10, 1.50, 0.10),
}

BEST_PARAMS_PATH    = os.path.join(os.path.dirname(__file__), "best_params.json")
BEST_PARAMS_V2_PATH = os.path.join(os.path.dirname(__file__), "best_params_v2.json")

_RETURN_TARGET = 0.10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_params(params: dict, cfg):
    for k, v in params.items():
        setattr(cfg, k, v)


def _restore_defaults(cfg):
    importlib.reload(cfg)


def _suggest_params(trial, param_space: dict) -> dict:
    params = {}
    for name, spec in param_space.items():
        kind = spec[0]
        if kind == "int":
            _, lo, hi, step = spec
            params[name] = trial.suggest_int(name, lo, hi, step=step)
        else:
            _, lo, hi, step = spec
            params[name] = trial.suggest_float(name, lo, hi, step=step)
    return params


def _run_on_slice(run_fn, macro: pd.DataFrame, prices: pd.DataFrame):
    try:
        return run_fn(macro, prices)
    except Exception as exc:
        logger.debug("Trial backtest failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def make_objective(macro_train, prices_train, run_fn, cfg, param_space):
    def objective(trial):
        params = _suggest_params(trial, param_space)
        _apply_params(params, cfg)

        results = _run_on_slice(run_fn, macro_train, prices_train)
        if results is None or len(results["daily_returns"].dropna()) < 252:
            return -10.0

        ret = results["daily_returns"].dropna()
        nav = results["nav"]
        sr  = sharpe(ret)
        mdd = max_drawdown(nav)

        n       = len(ret)
        ann_ret = float(nav.iloc[-1] ** (252 / n) - 1)

        dd_penalty     = max(0.0, abs(mdd) - 0.10) * 20.0
        return_penalty = max(0.0, _RETURN_TARGET - ann_ret) * 4.0
        monthly_ret    = (1 + ret).resample("ME").prod() - 1
        wm_penalty     = max(0.0, -0.04 - float(monthly_ret.min())) * 8.0

        return sr * ann_ret * 10 - dd_penalty - return_penalty - wm_penalty

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_optimization(n_trials: int = 300, v2: bool = False):
    if v2:
        import config_v2 as cfg_mod
        from data.pipeline_v2    import load_all as _load_all
        from strategy_v2.backtest import run as _run
        param_space = {**PARAM_SPACE, **V2_PARAM_ADDITIONS}
        best_path   = BEST_PARAMS_V2_PATH
        label       = "V2"
    else:
        cfg_mod     = config
        _load_all   = load_all
        _run        = run
        param_space = PARAM_SPACE
        best_path   = BEST_PARAMS_PATH
        label       = "V1"

    logger.info("Loading data for %s optimization...", label)
    macro, prices = _load_all()

    split = int(len(macro) * 0.70)
    macro_train, prices_train = macro.iloc[:split],  prices.iloc[:split]
    macro_test,  prices_test  = macro.iloc[split:],  prices.iloc[split:]

    logger.info(
        "Train: %s → %s  |  Test: %s → %s",
        macro_train.index[0].date(), macro_train.index[-1].date(),
        macro_test.index[0].date(),  macro_test.index[-1].date(),
    )
    print(f"[{label}] Train: {macro_train.index[0].date()} → {macro_train.index[-1].date()}")
    print(f"[{label}] Test : {macro_test.index[0].date()}  → {macro_test.index[-1].date()}")
    print(f"Running {n_trials} Optuna trials ({label})...\n")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        make_objective(macro_train, prices_train, _run, cfg_mod, param_space),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_val    = study.best_value
    print(f"\nBest train objective ({label}): {best_val:.3f}")

    _apply_params(best_params, cfg_mod)
    res_train = _run(macro_train, prices_train)
    res_test  = _run(macro_test,  prices_test)
    _restore_defaults(cfg_mod)

    s_train = summary(res_train["daily_returns"], res_train["nav"], "Train")
    s_test  = summary(res_test["daily_returns"],  res_test["nav"],  "Test (OOS)")

    print("\n" + "=" * 52)
    print(f"OPTIMISED {label} — TRAIN vs OUT-OF-SAMPLE")
    print("=" * 52)
    print(pd.concat([s_train, s_test], axis=1).to_string())
    print("=" * 52)

    _restore_defaults(cfg_mod)
    res_base = _run(macro_test, prices_test)
    s_base   = summary(res_base["daily_returns"], res_base["nav"], f"Default {label} (OOS)")
    print(f"\nDefault {label} params on same test period:")
    print(pd.concat([s_test, s_base], axis=1).to_string())

    print("\nBest parameters:")
    for k, v in best_params.items():
        default_val = getattr(cfg_mod, k, "N/A")
        print(f"  {k:<26s} {v!s:>8}   (default: {default_val})")

    with open(best_path, "w") as f:
        json.dump(best_params, f, indent=2)
    logger.info("Best %s params saved → %s", label, best_path)
    print(f"\nSaved → {best_path}")

    return best_params


def load_best_params(suffix: str = "") -> dict:
    """Load saved best params. suffix='' for v1, '_v2' for v2."""
    path = os.path.join(os.path.dirname(__file__), f"best_params{suffix}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No {os.path.basename(path)} found — run optimize first.")
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s  %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--v2", action="store_true", help="Optimise v2 strategy")
    args = parser.parse_args()
    run_optimization(n_trials=args.trials, v2=args.v2)
