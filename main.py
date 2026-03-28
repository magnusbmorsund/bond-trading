"""
Bond Rotation Strategy — CLI entry point.

Commands:
  python main.py fetch                Fetch / refresh all data
  python main.py backtest [--best]    Backtest (--best uses optimised params)
  python main.py weights  [--best]    Current weights for IBKR (--best uses optimised params)
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
from strategy.backtest    import run
from analysis.performance import print_summary_table, plot_results

CHART_PATH = os.path.join(os.path.dirname(__file__), "backtest_results.png")


def _maybe_load_best(use_best: bool):
    if not use_best:
        return
    from optimize import load_best_params
    params = load_best_params()
    for k, v in params.items():
        setattr(config, k, v)
    print("Loaded optimised params from best_params.json")


def cmd_fetch():
    print("Fetching all data (force refresh)...")
    macro, prices = load_all(force=True)
    print(f"\nDone. macro={macro.shape}  prices={prices.shape}")
    print(f"Date range: {macro.index[0].date()} → {macro.index[-1].date()}")


def cmd_backtest(use_best: bool = False):
    _maybe_load_best(use_best)
    print("Loading data...")
    macro, prices = load_all()

    print("Running backtest...")
    results = run(macro, prices)

    print_summary_table(results)
    print("\nMonthly turnover (avg):", f"{results['turnover'].mean():.1%}")

    print("\nMost recent weights:")
    for etf, w in results["weights"].iloc[-1].sort_values(ascending=False).items():
        print(f"  {etf:>4s}  {w:5.1%}  {'█' * int(w * 30)}")

    print(f"\nSaving chart → {CHART_PATH}")
    plot_results(results, save_path=CHART_PATH)


def cmd_weights(use_best: bool = False):
    _maybe_load_best(use_best)
    print("Loading data...")
    macro, prices = load_all()

    results  = run(macro, prices)
    weights  = results["weights"]
    latest_w = weights.iloc[-1]

    print(f"\nTarget weights as of {weights.index[-1].date()} (effective next trading day):")
    print("-" * 35)
    for etf, w in latest_w.sort_values(ascending=False).items():
        print(f"  {etf:>4s}  {w:6.2%}  {'█' * int(w * 40)}")
    print("-" * 35)
    print(f"  {'Sum':>4s}  {latest_w.sum():6.2%}")
    print("\nFor IBKR: set each position as % of total portfolio value above.")


def cmd_optimize(n_trials: int = 300):
    from optimize import run_optimization
    run_optimization(n_trials=n_trials)


def main():
    parser = argparse.ArgumentParser(prog="main.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("fetch")

    p_bt = sub.add_parser("backtest")
    p_bt.add_argument("--best", action="store_true", help="Use optimised params")

    p_wt = sub.add_parser("weights")
    p_wt.add_argument("--best", action="store_true", help="Use optimised params")

    p_opt = sub.add_parser("optimize")
    p_opt.add_argument("--trials", type=int, default=300)

    args = parser.parse_args()

    if   args.cmd == "fetch":    cmd_fetch()
    elif args.cmd == "backtest": cmd_backtest(use_best=args.best)
    elif args.cmd == "weights":  cmd_weights(use_best=args.best)
    elif args.cmd == "optimize": cmd_optimize(n_trials=args.trials)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
