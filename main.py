"""
Bond Rotation Strategy — CLI entry point.

Commands:
  python main.py fetch       Fetch / refresh all data (FRED + Yahoo Finance)
  python main.py backtest    Run backtest, print summary, save chart
  python main.py weights     Show current target weights for live trading
"""
import sys
import os
import pandas as pd

from data.pipeline       import load_all
from strategy.backtest   import run
from analysis.performance import print_summary_table, plot_results

CHART_PATH = os.path.join(os.path.dirname(__file__), "backtest_results.png")


def cmd_fetch():
    print("Fetching all data (force refresh)...")
    macro, prices = load_all(force=True)
    print(f"\nDone. macro={macro.shape}  prices={prices.shape}")
    print(f"Date range: {macro.index[0].date()} → {macro.index[-1].date()}")


def cmd_backtest():
    print("Loading data...")
    macro, prices = load_all()

    print("Running backtest...")
    results = run(macro, prices)

    print_summary_table(results)

    print("\nMonthly turnover (avg):", f"{results['turnover'].mean():.1%}")

    print("\nMost recent weights:")
    latest = results["weights"].iloc[-1].sort_values(ascending=False)
    for etf, w in latest.items():
        bar = "█" * int(w * 30)
        print(f"  {etf:>4s}  {w:5.1%}  {bar}")

    print(f"\nSaving chart → {CHART_PATH}")
    plot_results(results, save_path=CHART_PATH)


def cmd_weights():
    """Print current target weights for live IBKR orders."""
    print("Loading data...")
    macro, prices = load_all()

    print("Computing current weights...")
    results = run(macro, prices)

    weights = results["weights"]
    latest_date = weights.index[-1]
    latest_w = weights.iloc[-1]

    print(f"\nTarget weights as of {latest_date.date()} (effective next trading day):")
    print("-" * 35)
    for etf, w in latest_w.sort_values(ascending=False).items():
        bar = "█" * int(w * 40)
        print(f"  {etf:>4s}  {w:6.2%}  {bar}")
    print("-" * 35)
    print(f"  {'Sum':>4s}  {latest_w.sum():6.2%}")

    print("\nFor IBKR: set each position as % of total portfolio value above.")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1].lower()
    if cmd == "fetch":
        cmd_fetch()
    elif cmd == "backtest":
        cmd_backtest()
    elif cmd == "weights":
        cmd_weights()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
