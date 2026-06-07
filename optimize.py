"""
Optuna hyperparameter optimization for all bond and sector rotation strategies.

Splits data 70/30 (train/test). Optimises on the training window, then evaluates
best params on the held-out test window.

Usage (via main.py — preferred):
    python main.py optimize v1 --trials 300
    python main.py optimize sector2e --trials 300

Usage (direct):
    python optimize.py [--trials N] [--v2] [--v3]
    python optimize.py [--trials N] [--sector] [--sector2] [--sector2b]
    python optimize.py [--trials N] [--sector2c] [--sector2d] [--sector2e]
    python optimize.py [--trials N] [--sector2 --stop-freq weekly]
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

import configs.bond_v1 as config
from data.pipelines.bond_v1       import load_all
from strategies.bond_v1.backtest   import run
from analysis.performance import sharpe, max_drawdown, summary

logger = logging.getLogger(__name__)

_BASE = os.path.dirname(__file__)

BEST_PARAMS_PATH                  = os.path.join(_BASE, "best_params.json")
BEST_PARAMS_V2_PATH               = os.path.join(_BASE, "best_params_v2.json")
BEST_PARAMS_V3_PATH               = os.path.join(_BASE, "best_params_v3.json")
BEST_PARAMS_SECTOR_PATH           = os.path.join(_BASE, "best_params_sector.json")
BEST_PARAMS_SECTOR2_PATH          = os.path.join(_BASE, "best_params_sector2.json")
BEST_PARAMS_SECTOR2_WEEKLY_PATH   = os.path.join(_BASE, "best_params_sector2_weekly.json")
BEST_PARAMS_SECTOR2_MONTHLY_PATH  = os.path.join(_BASE, "best_params_sector2_monthly.json")
BEST_PARAMS_SECTOR2B_PATH         = os.path.join(_BASE, "best_params_sector2b.json")
BEST_PARAMS_SECTOR2C_PATH         = os.path.join(_BASE, "best_params_sector2c.json")
BEST_PARAMS_SECTOR2D_PATH         = os.path.join(_BASE, "best_params_sector2d.json")
BEST_PARAMS_SECTOR2E_PATH         = os.path.join(_BASE, "best_params_sector2e.json")
BEST_PARAMS_SECTOR2F_PATH         = os.path.join(_BASE, "best_params_sector2f.json")
BEST_PARAMS_SECTOR2G_PATH         = os.path.join(_BASE, "best_params_sector2g.json")

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
    """Handle both 4-tuple specs (with step) and 3-tuple specs (no step)."""
    params = {}
    for name, spec in param_space.items():
        kind = spec[0]
        if kind == "int":
            if len(spec) == 4:
                _, lo, hi, step = spec
                params[name] = trial.suggest_int(name, lo, hi, step=step)
            else:
                _, lo, hi = spec
                params[name] = trial.suggest_int(name, lo, hi)
        else:
            if len(spec) == 4:
                _, lo, hi, step = spec
                params[name] = trial.suggest_float(name, lo, hi, step=step)
            else:
                _, lo, hi = spec
                params[name] = trial.suggest_float(name, lo, hi)
    return params


def _run_on_slice(run_fn, macro, prices: pd.DataFrame):
    """Run backtest on a slice. macro may be None for sector strategy."""
    try:
        return run_fn(macro, prices)
    except Exception as exc:
        logger.debug("Trial backtest failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Objective (bond strategies)
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
# Generic sector optimizer (prices-only strategies: V1, V2b–V2e)
# ---------------------------------------------------------------------------

def _optimize_sector(
    cfg_mod, load_fn, run_fn, label: str, best_path: str, n_trials: int,
    dd_thresh: float = 0.12, ret_thresh: float = 0.15, wm_thresh: float = -0.07,
    run_kwargs: dict | None = None,
) -> dict:
    """Run a full Optuna study for a sector strategy, print train/OOS summary, save params."""
    run_kwargs = run_kwargs or {}
    param_space = cfg_mod.PARAM_SPACE

    logger.info("Loading %s price data...", label)
    prices_all   = load_fn()
    split        = int(len(prices_all) * 0.70)
    prices_train = prices_all.iloc[:split]
    prices_test  = prices_all.iloc[split:]

    logger.info(
        "Train: %s → %s  |  Test: %s → %s",
        prices_train.index[0].date(), prices_train.index[-1].date(),
        prices_test.index[0].date(),  prices_test.index[-1].date(),
    )
    print(f"[{label}] Train: {prices_train.index[0].date()} → {prices_train.index[-1].date()}")
    print(f"[{label}] Test : {prices_test.index[0].date()}  → {prices_test.index[-1].date()}")
    print(f"Running {n_trials} Optuna trials ({label})...\n")

    def objective(trial):
        params = _suggest_params(trial, param_space)
        _apply_params(params, cfg_mod)
        try:
            results = run_fn(prices_train, **run_kwargs)
        except Exception as exc:
            logger.debug("%s trial failed: %s", label, exc)
            return -10.0
        if results is None or len(results["daily_returns"].dropna()) < 252:
            return -10.0
        ret = results["daily_returns"].dropna()
        nav = results["nav"]
        sr  = sharpe(ret)
        mdd = max_drawdown(nav)
        n   = len(ret)
        ann_ret        = float(nav.iloc[-1] ** (252 / n) - 1)
        dd_penalty     = max(0.0, abs(mdd) - dd_thresh) * 20.0
        return_penalty = max(0.0, ret_thresh - ann_ret) * 4.0
        monthly_ret    = (1 + ret).resample("ME").prod() - 1
        wm_penalty     = max(0.0, wm_thresh - float(monthly_ret.min())) * 8.0
        return sr * ann_ret * 10 - dd_penalty - return_penalty - wm_penalty

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    print(f"\nBest train objective ({label}): {study.best_value:.3f}")

    _apply_params(best_params, cfg_mod)
    res_train = run_fn(prices_train, **run_kwargs)
    res_test  = run_fn(prices_test,  **run_kwargs)
    _restore_defaults(cfg_mod)

    s_train = summary(res_train["daily_returns"], res_train["nav"], "Train")
    s_test  = summary(res_test["daily_returns"],  res_test["nav"],  "Test (OOS)")

    print("\n" + "=" * 52)
    print(f"OPTIMISED {label} — TRAIN vs OUT-OF-SAMPLE")
    print("=" * 52)
    print(pd.concat([s_train, s_test], axis=1).to_string())
    print("=" * 52)

    _restore_defaults(cfg_mod)
    res_base = run_fn(prices_test, **run_kwargs)
    s_base   = summary(res_base["daily_returns"], res_base["nav"], f"Default {label} (OOS)")
    print(f"\nDefault {label} params on same test period:")
    print(pd.concat([s_test, s_base], axis=1).to_string())

    print("\nBest parameters:")
    for k, v in best_params.items():
        default_val = getattr(cfg_mod, k, "N/A")
        print(f"  {k:<30s} {v!s:>10}   (default: {default_val})")

    with open(best_path, "w") as f:
        json.dump(best_params, f, indent=2)
    logger.info("Best %s params saved → %s", label, best_path)
    print(f"\nSaved → {best_path}")
    return best_params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_optimization(
    n_trials: int = 300,
    v2: bool = False, v3: bool = False,
    sector: bool = False,
    sector2: bool = False, sector2b: bool = False, sector2c: bool = False,
    sector2d: bool = False, sector2e: bool = False,
    sector2f: bool = False, sector2g: bool = False,
    stop_freq: str = "daily",
):
    # ── Sector V2 (with stop_freq variants) ───────────────────────────────
    if sector2:
        import configs.sector_v2 as cfg_mod
        from data.pipelines.sector_v2     import load_all as _load_prices
        from strategies.sector_v2.backtest import run as _run_sector2

        if stop_freq == "weekly":
            best_path  = BEST_PARAMS_SECTOR2_WEEKLY_PATH
            label      = "SECTOR2-WEEKLY"
            dd_thresh, ret_thresh, wm_thresh = 0.14, 0.12, -0.08
        elif stop_freq == "monthly":
            best_path  = BEST_PARAMS_SECTOR2_MONTHLY_PATH
            label      = "SECTOR2-MONTHLY"
            dd_thresh, ret_thresh, wm_thresh = 0.16, 0.10, -0.09
        else:
            best_path  = BEST_PARAMS_SECTOR2_PATH
            label      = "SECTOR2"
            dd_thresh, ret_thresh, wm_thresh = 0.12, 0.15, -0.07

        return _optimize_sector(
            cfg_mod, _load_prices, _run_sector2, label, best_path, n_trials,
            dd_thresh=dd_thresh, ret_thresh=ret_thresh, wm_thresh=wm_thresh,
            run_kwargs={"stop_freq": stop_freq},
        )

    # ── Sector V2b ────────────────────────────────────────────────────────
    elif sector2b:
        import configs.sector_v2b as cfg_mod
        from data.pipelines.sector_v2b     import load_all as _load_prices
        from strategies.sector_v2b.backtest import run as _run
        return _optimize_sector(cfg_mod, _load_prices, _run, "SECTOR2B", BEST_PARAMS_SECTOR2B_PATH, n_trials)

    # ── Sector V2c ────────────────────────────────────────────────────────
    elif sector2c:
        import configs.sector_v2c as cfg_mod
        from data.pipelines.sector_v2c     import load_all as _load_prices
        from strategies.sector_v2c.backtest import run as _run
        return _optimize_sector(cfg_mod, _load_prices, _run, "SECTOR2C", BEST_PARAMS_SECTOR2C_PATH, n_trials)

    # ── Sector V2d ────────────────────────────────────────────────────────
    elif sector2d:
        import configs.sector_v2d as cfg_mod
        from data.pipelines.sector_v2d     import load_all as _load_prices
        from strategies.sector_v2d.backtest import run as _run
        return _optimize_sector(cfg_mod, _load_prices, _run, "SECTOR2D", BEST_PARAMS_SECTOR2D_PATH, n_trials)

    # ── Sector V2e ────────────────────────────────────────────────────────
    elif sector2e:
        import configs.sector_v2e as cfg_mod
        from data.pipelines.sector_v2e     import load_all as _load_prices
        from strategies.sector_v2e.backtest import run as _run
        return _optimize_sector(cfg_mod, _load_prices, _run, "SECTOR2E", BEST_PARAMS_SECTOR2E_PATH, n_trials)

    # ── Sector V2f (UCITS-tradeable subset of V2e) ────────────────────────
    elif sector2f:
        import configs.sector_v2f as cfg_mod
        from data.pipelines.sector_v2f     import load_all as _load_prices
        from strategies.sector_v2f.backtest import run as _run
        return _optimize_sector(cfg_mod, _load_prices, _run, "SECTOR2F", BEST_PARAMS_SECTOR2F_PATH, n_trials)

    # ── Sector V2g (honest rebuild: trend-break exit + concentration) ─────
    elif sector2g:
        import configs.sector_v2g as cfg_mod
        from data.pipelines.sector_v2g     import load_all as _load_prices
        from strategies.sector_v2g.backtest import run as _run
        return _optimize_sector(cfg_mod, _load_prices, _run, "SECTOR2G", BEST_PARAMS_SECTOR2G_PATH, n_trials)

    # ── Sector V1 ─────────────────────────────────────────────────────────
    elif sector:
        import configs.sector_v1 as cfg_mod
        from data.pipelines.sector_v1     import load_all as _load_prices
        from strategies.sector_v1.backtest import run as _run
        return _optimize_sector(
            cfg_mod, _load_prices, _run, "SECTOR", BEST_PARAMS_SECTOR_PATH, n_trials,
            dd_thresh=0.12, ret_thresh=0.12, wm_thresh=-0.06,
        )

    # ── Bond V3 ───────────────────────────────────────────────────────────
    elif v3:
        import configs.bond_v3 as cfg_mod
        from data.pipelines.bond_v3    import load_all as _load_all
        from strategies.bond_v3.backtest import run as _run
        param_space = cfg_mod.PARAM_SPACE
        best_path   = BEST_PARAMS_V3_PATH
        label       = "V3"

    # ── Bond V2 ───────────────────────────────────────────────────────────
    elif v2:
        import configs.bond_v2 as cfg_mod
        from data.pipelines.bond_v2    import load_all as _load_all
        from strategies.bond_v2.backtest import run as _run
        param_space = cfg_mod.PARAM_SPACE
        best_path   = BEST_PARAMS_V2_PATH
        label       = "V2"

    # ── Bond V1 (default) ─────────────────────────────────────────────────
    else:
        cfg_mod     = config
        _load_all   = load_all
        _run        = run
        param_space = config.PARAM_SPACE
        best_path   = BEST_PARAMS_PATH
        label       = "V1"

    # ── Shared bond optimization path ─────────────────────────────────────
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

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        make_objective(macro_train, prices_train, _run, cfg_mod, param_space),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_params = study.best_params
    print(f"\nBest train objective ({label}): {study.best_value:.3f}")

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
    path = os.path.join(_BASE, f"best_params{suffix}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No {os.path.basename(path)} found — run optimize first.")
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s  %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials",    type=int, default=300)
    parser.add_argument("--v2",        action="store_true")
    parser.add_argument("--v3",        action="store_true")
    parser.add_argument("--sector",    action="store_true")
    parser.add_argument("--sector2",   action="store_true")
    parser.add_argument("--sector2b",  action="store_true")
    parser.add_argument("--sector2c",  action="store_true")
    parser.add_argument("--sector2d",  action="store_true")
    parser.add_argument("--sector2e",  action="store_true")
    parser.add_argument("--sector2f",  action="store_true")
    parser.add_argument("--sector2g",  action="store_true")
    parser.add_argument("--stop-freq", default="daily", choices=["daily", "weekly", "monthly"],
                        dest="stop_freq")
    args = parser.parse_args()
    run_optimization(
        n_trials=args.trials, v2=args.v2, v3=args.v3,
        sector=args.sector, sector2=args.sector2, sector2b=args.sector2b,
        sector2c=args.sector2c, sector2d=args.sector2d, sector2e=args.sector2e,
        sector2f=args.sector2f, sector2g=args.sector2g,
        stop_freq=args.stop_freq,
    )
