"""
Bond Rotation Strategy — CLI entry point.

Commands:
  python main.py fetch                Fetch / refresh all data
  python main.py backtest [--best]    Backtest (--best uses optimised params)
  python main.py weights              Current positions for IBKR (trailing-stop adjusted)
  python main.py optimize [--trials N] Run Optuna optimisation (default 300 trials)
"""
import sys
import os
import logging
import argparse
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import config
from data.pipeline        import load_all
from strategy.backtest    import run, effective_weights
from analysis.performance import print_summary_table, plot_results, plot_annual_stats, plot_annual_allocations

CHART_PATH        = os.path.join(os.path.dirname(__file__), "backtest_results.png")
ANNUAL_STATS_PATH = os.path.join(os.path.dirname(__file__), "annual_stats.png")
ANNUAL_ALLOC_PATH = os.path.join(os.path.dirname(__file__), "annual_allocations.png")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging():
    os.makedirs(config.LOG_DIR, exist_ok=True)
    log_file = os.path.join(config.LOG_DIR, "strategy.log")

    fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
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
    # Quieten noisy third-party loggers
    for noisy in ("yfinance", "urllib3", "peewee", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Startup guard
# ---------------------------------------------------------------------------

def _validate_env():
    """Fail fast if critical env vars are missing."""
    if not config.FRED_API_KEY:
        logger.error(
            "FRED_API_KEY is not set. "
            "Export it before running:  export FRED_API_KEY=<your_key>"
        )
        sys.exit(1)
    if len(config.FRED_API_KEY) != 32:
        logger.warning(
            "FRED_API_KEY looks malformed (expected 32 chars, got %d). "
            "Fetches will fall back to cached data.",
            len(config.FRED_API_KEY),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_best():
    """Always load best_params.json — it holds the production configuration."""
    from optimize import load_best_params
    params = load_best_params()
    for k, v in params.items():
        setattr(config, k, v)
    logger.info("Loaded %d optimised params from best_params.json", len(params))


def _validate_weights(weights: pd.Series, label: str = "weights") -> pd.Series:
    """
    Guard before sending weights to broker:
      - Replace NaN with 0
      - Assert non-negative
      - Renormalise if sum deviates from 1.0 by more than 0.5%
    """
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
        logger.warning(
            "%s sum=%.4f — renormalising to 1.0", label, total
        )
        weights = weights / total

    return weights


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_fetch():
    logger.info("Force-refreshing all data...")
    macro, prices = load_all(force=True)
    logger.info(
        "Done. macro=%s  prices=%s  range=%s → %s",
        macro.shape, prices.shape,
        macro.index[0].date(), macro.index[-1].date(),
    )


def cmd_backtest(use_best: bool = False):
    if use_best:
        _load_best()
    logger.info("Loading data...")
    macro, prices = load_all()

    logger.info("Running backtest...")
    results = run(macro, prices)

    print_summary_table(results)
    print("\nMonthly turnover (avg):", f"{results['turnover'].mean():.1%}")

    print("\nMost recent signal weights (pre trailing-stop):")
    for etf, w in results["weights"].iloc[-1].sort_values(ascending=False).items():
        print(f"  {etf:>4s}  {w:5.1%}  {'█' * int(w * 30)}")

    logger.info("Saving charts...")
    plot_results(results, save_path=CHART_PATH)
    plot_annual_stats(results, save_path=ANNUAL_STATS_PATH)
    plot_annual_allocations(results, save_path=ANNUAL_ALLOC_PATH)


def cmd_weights():
    """
    Show today's effective positions for IBKR, with trailing stops applied.
    Always uses production params from best_params.json.
    """
    _load_best()
    logger.info("Loading data...")
    macro, prices = load_all()

    results   = run(macro, prices)
    weights   = results["weights"]
    signal_w  = weights.iloc[-1]
    as_of     = weights.index[-1].date()

    # Apply trailing stops against the most recent prices
    eff_w = effective_weights(signal_w, prices[config.ETF_UNIVERSE])

    # Validate before printing / sending to broker
    eff_w = _validate_weights(eff_w, label="effective weights")

    print(f"\n{'='*45}")
    print(f"SIGNAL WEIGHTS  (model, as of {as_of})")
    print(f"{'='*45}")
    for etf, w in signal_w.sort_values(ascending=False).items():
        if w > 0.001:
            stopped = " [STOPPED OUT]" if eff_w.get(etf, w) < w * 0.5 else ""
            print(f"  {etf:>4s}  {w:6.2%}  {'█' * int(w * 40)}{stopped}")

    print(f"\n{'='*45}")
    print(f"EFFECTIVE POSITIONS  (after trailing stops — trade these)")
    print(f"{'='*45}")
    for etf, w in eff_w.sort_values(ascending=False).items():
        if w > 0.001:
            print(f"  {etf:>4s}  {w:6.2%}  {'█' * int(w * 40)}")
    print(f"  {'─'*40}")
    print(f"  {'Sum':>4s}  {eff_w.sum():6.2%}")
    print(f"\nFor IBKR: set each ETF as % of total portfolio value above.")
    print(f"Trailing stop: {config.TRAILING_STOP_PCT:.0%} below {config.TRAILING_STOP_WINDOW}-day peak")


def cmd_optimize(n_trials: int = 300):
    from optimize import run_optimization
    run_optimization(n_trials=n_trials)


def main():
    _setup_logging()

    parser = argparse.ArgumentParser(prog="main.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("fetch")

    p_bt = sub.add_parser("backtest")
    p_bt.add_argument("--best", action="store_true", help="Use optimised params from best_params.json")

    sub.add_parser("weights")

    p_opt = sub.add_parser("optimize")
    p_opt.add_argument("--trials", type=int, default=300)

    args = parser.parse_args()

    # Validate env for data-fetching commands (not needed for backtest-only if cache is warm)
    if args.cmd in ("fetch", "weights", "optimize"):
        _validate_env()

    if   args.cmd == "fetch":    cmd_fetch()
    elif args.cmd == "backtest": cmd_backtest(use_best=args.best)
    elif args.cmd == "weights":  cmd_weights()
    elif args.cmd == "optimize": cmd_optimize(n_trials=args.trials)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
