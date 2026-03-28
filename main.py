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
import argparse
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import config
from data.pipeline        import load_all
from strategy.backtest    import run, effective_weights
from analysis.performance import print_summary_table, plot_results

CHART_PATH = os.path.join(os.path.dirname(__file__), "backtest_results.png")


def _load_best():
    """Always load best_params.json — it holds the production configuration."""
    from optimize import load_best_params
    params = load_best_params()
    for k, v in params.items():
        setattr(config, k, v)


def cmd_fetch():
    print("Fetching all data (force refresh)...")
    macro, prices = load_all(force=True)
    print(f"\nDone. macro={macro.shape}  prices={prices.shape}")
    print(f"Date range: {macro.index[0].date()} → {macro.index[-1].date()}")


def cmd_backtest(use_best: bool = False):
    if use_best:
        _load_best()
    print("Loading data...")
    macro, prices = load_all()

    print("Running backtest...")
    results = run(macro, prices)

    print_summary_table(results)
    print("\nMonthly turnover (avg):", f"{results['turnover'].mean():.1%}")

    print("\nMost recent signal weights (pre trailing-stop):")
    for etf, w in results["weights"].iloc[-1].sort_values(ascending=False).items():
        print(f"  {etf:>4s}  {w:5.1%}  {'█' * int(w * 30)}")

    print(f"\nSaving chart → {CHART_PATH}")
    plot_results(results, save_path=CHART_PATH)


def cmd_weights():
    """
    Show today's effective positions for IBKR, with trailing stops applied.
    Always uses production params from best_params.json.
    """
    _load_best()
    print("Loading data...")
    macro, prices = load_all()

    results   = run(macro, prices)
    weights   = results["weights"]
    signal_w  = weights.iloc[-1]   # latest monthly signal weights
    as_of     = weights.index[-1].date()

    # Apply trailing stops against the most recent prices to get actionable positions
    eff_w = effective_weights(signal_w, prices[config.ETF_UNIVERSE])

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
    parser = argparse.ArgumentParser(prog="main.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("fetch")

    p_bt = sub.add_parser("backtest")
    p_bt.add_argument("--best", action="store_true", help="Use optimised params from best_params.json")

    sub.add_parser("weights")   # always uses best_params.json (production config)

    p_opt = sub.add_parser("optimize")
    p_opt.add_argument("--trials", type=int, default=300)

    args = parser.parse_args()

    if   args.cmd == "fetch":    cmd_fetch()
    elif args.cmd == "backtest": cmd_backtest(use_best=args.best)
    elif args.cmd == "weights":  cmd_weights()
    elif args.cmd == "optimize": cmd_optimize(n_trials=args.trials)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
