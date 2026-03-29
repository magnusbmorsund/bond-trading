"""
Bond Rotation Strategy — CLI entry point.

Commands:
  python main.py fetch                         Fetch / refresh all data
  python main.py backtest [--best] [--v2]      Backtest (--best uses optimised params)
  python main.py weights  [--v2]               Current positions for IBKR
  python main.py optimize [--trials N] [--v2]  Run Optuna optimisation (default 300 trials)

Add --v2 to any command to run the v2 strategy (SLV, VTIP, VNQ, SPY, USD signal, ISM signal).
"""
import sys
import os
import json
import logging
import argparse
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import config
from analysis.performance import print_summary_table, plot_results, plot_annual_stats, plot_annual_allocations, plot_comparison

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(v2: bool = False):
    os.makedirs(config.LOG_DIR, exist_ok=True)
    suffix   = "_v2" if v2 else ""
    log_file = os.path.join(config.LOG_DIR, f"strategy{suffix}.log")

    fmt    = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
        ],
    )
    for noisy in ("yfinance", "urllib3", "peewee", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Strategy module selector
# ---------------------------------------------------------------------------

def _get_modules(v2: bool):
    """Return (cfg, load_all, run, effective_weights) for v1 or v2."""
    if v2:
        import config_v2 as cfg
        from data.pipeline_v2    import load_all
        from strategy_v2.backtest import run, effective_weights
    else:
        cfg = config
        from data.pipeline      import load_all
        from strategy.backtest  import run, effective_weights
    return cfg, load_all, run, effective_weights


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_env(cfg):
    if not cfg.FRED_API_KEY:
        logger.error("FRED_API_KEY is not set. Export it before running.")
        sys.exit(1)
    if len(cfg.FRED_API_KEY) != 32:
        logger.warning(
            "FRED_API_KEY looks malformed (expected 32 chars, got %d). "
            "Fetches will fall back to cached data.",
            len(cfg.FRED_API_KEY),
        )


def _load_best(cfg, v2: bool = False):
    """Monkey-patch cfg with saved optimised params."""
    suffix = "_v2" if v2 else ""
    path   = os.path.join(os.path.dirname(__file__), f"best_params{suffix}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{os.path.basename(path)} not found — run: python main.py optimize{'  --v2' if v2 else ''}"
        )
    with open(path) as f:
        params = json.load(f)
    for k, v in params.items():
        setattr(cfg, k, v)
    logger.info("Loaded %d optimised params from %s", len(params), os.path.basename(path))


def _validate_weights(weights: pd.Series, label: str = "weights") -> pd.Series:
    if weights.isna().any():
        bad = weights[weights.isna()].index.tolist()
        logger.warning("NaN weights replaced with 0 for: %s", bad)
        weights = weights.fillna(0.0)
    if (weights < 0).any():
        bad = weights[weights < 0].index.tolist()
        logger.warning("Negative weights clipped to 0 for: %s", bad)
        weights = weights.clip(lower=0.0)
    total = weights.sum()
    if abs(total - 1.0) > 0.005:
        logger.warning("%s sum=%.4f — renormalising to 1.0", label, total)
        weights = weights / total
    return weights


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_fetch(v2: bool = False):
    cfg, load_all, _, _ = _get_modules(v2)
    _validate_env(cfg)
    label = "v2" if v2 else "v1"
    logger.info("Force-refreshing all %s data...", label)
    macro, prices = load_all(force=True)
    logger.info(
        "Done. macro=%s  prices=%s  range=%s → %s",
        macro.shape, prices.shape,
        macro.index[0].date(), macro.index[-1].date(),
    )


def cmd_backtest(use_best: bool = False, v2: bool = False):
    cfg, load_all, run, _ = _get_modules(v2)
    suffix = "_v2" if v2 else ""

    if use_best:
        _load_best(cfg, v2=v2)

    logger.info("Loading data...")
    macro, prices = load_all()

    logger.info("Running backtest...")
    results = run(macro, prices)

    print_summary_table(results)
    print("\nMonthly turnover (avg):", f"{results['turnover'].mean():.1%}")

    print("\nMost recent signal weights (pre trailing-stop):")
    for etf, w in results["weights"].iloc[-1].sort_values(ascending=False).items():
        if w > 0.001:
            print(f"  {etf:>4s}  {w:5.1%}  {'█' * int(w * 30)}")

    base = os.path.dirname(__file__)
    logger.info("Saving charts...")
    plot_results(results,          save_path=os.path.join(base, f"backtest_results{suffix}.png"))
    plot_annual_stats(results,     save_path=os.path.join(base, f"annual_stats{suffix}.png"))
    plot_annual_allocations(results, save_path=os.path.join(base, f"annual_allocations{suffix}.png"))


def cmd_weights(v2: bool = False):
    cfg, load_all, run, effective_weights = _get_modules(v2)

    _load_best(cfg, v2=v2)
    logger.info("Loading data...")
    macro, prices = load_all()

    results  = run(macro, prices)
    weights  = results["weights"]
    signal_w = weights.iloc[-1]
    as_of    = weights.index[-1].date()

    eff_w = effective_weights(signal_w, prices[cfg.ETF_UNIVERSE])
    eff_w = _validate_weights(eff_w, label="effective weights")

    label = " V2" if v2 else ""
    print(f"\n{'='*45}")
    print(f"SIGNAL WEIGHTS{label}  (model, as of {as_of})")
    print(f"{'='*45}")
    for etf, w in signal_w.sort_values(ascending=False).items():
        if w > 0.001:
            stopped = " [STOPPED OUT]" if eff_w.get(etf, w) < w * 0.5 else ""
            print(f"  {etf:>4s}  {w:6.2%}  {'█' * int(w * 40)}{stopped}")

    print(f"\n{'='*45}")
    print(f"EFFECTIVE POSITIONS{label}  (after trailing stops — trade these)")
    print(f"{'='*45}")
    for etf, w in eff_w.sort_values(ascending=False).items():
        if w > 0.001:
            print(f"  {etf:>4s}  {w:6.2%}  {'█' * int(w * 40)}")
    print(f"  {'─'*40}")
    print(f"  {'Sum':>4s}  {eff_w.sum():6.2%}")
    print(f"\nFor IBKR: set each ETF as % of total portfolio value above.")
    print(f"Trailing stop: {cfg.TRAILING_STOP_PCT:.0%} below {cfg.TRAILING_STOP_WINDOW}-day peak")


def cmd_compare():
    """Run both V1 and V2 with best params and produce a side-by-side comparison chart."""
    import config_v2
    from data.pipeline_v2    import load_all as load_all_v2
    from strategy_v2.backtest import run as run_v2

    logger.info("Loading V1 data...")
    _load_best(config, v2=False)
    macro1, prices1 = __import__("data.pipeline", fromlist=["load_all"]).load_all()
    results_v1 = __import__("strategy.backtest", fromlist=["run"]).run(macro1, prices1)

    logger.info("Loading V2 data...")
    _load_best(config_v2, v2=True)
    macro2, prices2 = load_all_v2()
    results_v2 = run_v2(macro2, prices2)

    save_path = os.path.join(os.path.dirname(__file__), "backtest_comparison.png")
    logger.info("Saving comparison chart...")
    plot_comparison(results_v1, results_v2, save_path=save_path)


def cmd_optimize(n_trials: int = 300, v2: bool = False):
    from optimize import run_optimization
    run_optimization(n_trials=n_trials, v2=v2)


def main():
    parser = argparse.ArgumentParser(prog="main.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd")

    p_fetch = sub.add_parser("fetch")
    p_fetch.add_argument("--v2", action="store_true", help="Use v2 strategy")

    p_bt = sub.add_parser("backtest")
    p_bt.add_argument("--best", action="store_true", help="Use optimised params")
    p_bt.add_argument("--v2",   action="store_true", help="Use v2 strategy")

    p_wt = sub.add_parser("weights")
    p_wt.add_argument("--v2", action="store_true", help="Use v2 strategy")

    sub.add_parser("compare")

    p_opt = sub.add_parser("optimize")
    p_opt.add_argument("--trials", type=int, default=300)
    p_opt.add_argument("--v2", action="store_true", help="Optimise v2 strategy")

    args = parser.parse_args()

    v2  = getattr(args, "v2", False)
    cfg = _get_modules(v2)[0]

    _setup_logging(v2=v2)

    if args.cmd in ("fetch", "weights", "optimize"):
        _validate_env(cfg)

    if   args.cmd == "fetch":    cmd_fetch(v2=v2)
    elif args.cmd == "backtest": cmd_backtest(use_best=args.best, v2=v2)
    elif args.cmd == "weights":  cmd_weights(v2=v2)
    elif args.cmd == "compare":  cmd_compare()
    elif args.cmd == "optimize": cmd_optimize(n_trials=args.trials, v2=v2)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
