"""
Bond Rotation Strategy — CLI entry point.

Commands:
  python main.py fetch                                              Fetch / refresh all data
  python main.py backtest [--best] [--v2|--v3|--sector|--sector2]  Backtest
  python main.py weights  [--v2|--v3|--sector|--sector2]           Current positions for IBKR
  python main.py trade    [--v2|--v3|--sector|--sector2] [--dry-run]  Execute rebalance
  python main.py optimize [--trials N] [--v2|--v3|--sector|--sector2] Run Optuna optimisation

Add --v2 to any command to run the v2 strategy (SLV, VTIP, VNQ, SPY, USD signal, ISM signal).
Add --v3 to any command to run the v3 strategy (EDV, JPST, DBMF, MTUM, growth composite,
  credit impulse, VIX term structure).
Add --sector to run the V1 sector rotation strategy (XL series + SMH/IBB/NLR/MOO,
  single-lookback momentum, SPY 200d MA regime filter, inverse-vol weighting).
Add --sector2 to run the V2 sector rotation strategy (35 ETFs incl. compute/shipping/metals,
  multi-timescale composite momentum, adaptive trailing stops scaled by supercycle strength).

IBKR Gateway env vars (for the trade command):
  IBKR_HOST          Gateway hostname     (default: 127.0.0.1)
  IBKR_PORT          4002=paper 4001=live (default: 4002)
  IBKR_CLIENT_ID     API client ID        (default: 1)
  IBKR_MIN_ORDER_USD Min order size USD   (default: 50)
"""
import sys
import os
import json
import logging
import argparse
from dataclasses import dataclass
from typing import Any, Callable
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import config
from analysis.performance import print_summary_table, plot_results, plot_annual_stats, plot_annual_allocations, plot_comparison

logger = logging.getLogger(__name__)


@dataclass
class Strategy:
    config: Any
    load: Callable   # (force=False) -> (macro_or_None, prices)
    run: Callable    # (macro, prices) -> results
    eff_weights: Callable


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(v2: bool = False, v3: bool = False, sector2: bool = False):
    os.makedirs(config.LOG_DIR, exist_ok=True)
    suffix   = "_sector2" if sector2 else ("_v3" if v3 else ("_v2" if v2 else ""))
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

def _get_strategy(v2: bool, v3: bool = False, sector: bool = False,
                  sector2: bool = False, sector2b: bool = False,
                  sector2c: bool = False, stop_freq: str = "daily") -> Strategy:
    """Resolve config, pipeline, and backtest for the requested strategy variant."""
    if sector2c:
        import config_sector_v2c as cfg
        from data.pipeline_sector_v2c     import load_all as _load
        from strategy_sector_v2c.backtest import run as _run, effective_weights as _eff
        def _load_wrap(force=False): return None, _load(force=force)
        def _run_wrap(macro, prices): return _run(prices)
        return Strategy(cfg, _load_wrap, _run_wrap, _eff)

    if sector2b:
        import config_sector_v2b as cfg
        from data.pipeline_sector_v2b     import load_all as _load
        from strategy_sector_v2b.backtest import run as _run, effective_weights as _eff
        def _load_wrap(force=False): return None, _load(force=force)
        def _run_wrap(macro, prices): return _run(prices)
        return Strategy(cfg, _load_wrap, _run_wrap, _eff)

    if sector2:
        import config_sector_v2 as cfg
        from data.pipeline_sector_v2      import load_all as _load
        from strategy_sector_v2.backtest  import run as _run, effective_weights as _eff
        _sf = stop_freq
        def _load_wrap(force=False): return None, _load(force=force)
        def _run_wrap(macro, prices): return _run(prices, stop_freq=_sf)
        return Strategy(cfg, _load_wrap, _run_wrap, _eff)

    if sector:
        import config_sector as cfg
        from data.pipeline_sector      import load_all as _load
        from strategy_sector.backtest  import run as _run, effective_weights as _eff
        def _load_wrap(force=False): return None, _load(force=force)
        def _run_wrap(macro, prices): return _run(prices)
        return Strategy(cfg, _load_wrap, _run_wrap, _eff)

    if v3:
        import config_v3 as cfg
        from data.pipeline_v3     import load_all
        from strategy_v3.backtest import run, effective_weights
        return Strategy(cfg, load_all, run, effective_weights)

    if v2:
        import config_v2 as cfg
        from data.pipeline_v2     import load_all
        from strategy_v2.backtest import run, effective_weights
        return Strategy(cfg, load_all, run, effective_weights)

    # Default: V1 bond strategy
    from data.pipeline     import load_all
    from strategy.backtest import run, effective_weights
    return Strategy(config, load_all, run, effective_weights)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_env(cfg):
    # Sector strategy needs no FRED key
    if not hasattr(cfg, "FRED_API_KEY"):
        return
    if not cfg.FRED_API_KEY:
        logger.error("FRED_API_KEY is not set. Export it before running.")
        sys.exit(1)
    if len(cfg.FRED_API_KEY) != 32:
        logger.warning(
            "FRED_API_KEY looks malformed (expected 32 chars, got %d). "
            "Fetches will fall back to cached data.",
            len(cfg.FRED_API_KEY),
        )


def _load_best(cfg, v2: bool = False, v3: bool = False, sector: bool = False, sector2: bool = False, sector2b: bool = False, sector2c: bool = False, stop_freq: str = "daily"):
    """Monkey-patch cfg with saved optimised params."""
    if sector2 and stop_freq in ("weekly", "monthly"):
        suffix = f"_sector2_{stop_freq}"
    else:
        suffix = "_sector2c" if sector2c else ("_sector2b" if sector2b else ("_sector2" if sector2 else ("_sector" if sector else ("_v3" if v3 else ("_v2" if v2 else "")))))
    path   = os.path.join(os.path.dirname(__file__), f"best_params{suffix}.json")
    if not os.path.exists(path):
        flag = "  --sector2c" if sector2c else ("  --sector2" if sector2 else ("  --sector" if sector else ("  --v3" if v3 else ("  --v2" if v2 else ""))))
        raise FileNotFoundError(
            f"{os.path.basename(path)} not found — run: python main.py optimize{flag}"
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

def cmd_fetch(v2: bool = False, v3: bool = False, sector: bool = False, sector2: bool = False, sector2b: bool = False, sector2c: bool = False):
    s = _get_strategy(v2, v3, sector, sector2, sector2b, sector2c)
    _validate_env(s.config)
    label = "sector2c" if sector2c else ("sector2b" if sector2b else ("sector2" if sector2 else ("sector" if sector else ("v3" if v3 else ("v2" if v2 else "v1")))))
    logger.info("Force-refreshing all %s data...", label)
    macro, prices = s.load(force=True)
    if macro is not None:
        logger.info("Done. macro=%s  prices=%s  range=%s → %s",
                    macro.shape, prices.shape,
                    macro.index[0].date(), macro.index[-1].date())
    else:
        logger.info("Done. prices=%s  range=%s → %s",
                    prices.shape, prices.index[0].date(), prices.index[-1].date())


def cmd_backtest(use_best: bool = False, v2: bool = False, v3: bool = False, sector: bool = False, sector2: bool = False, sector2b: bool = False, sector2c: bool = False, stop_freq: str = "daily"):
    s = _get_strategy(v2, v3, sector, sector2, sector2b, sector2c, stop_freq=stop_freq)
    if sector2 and stop_freq in ("weekly", "monthly"):
        suffix = f"_sector2_{stop_freq}"
    else:
        suffix = "_sector2c" if sector2c else ("_sector2b" if sector2b else ("_sector2" if sector2 else ("_sector" if sector else ("_v3" if v3 else ("_v2" if v2 else "")))))

    if use_best:
        _load_best(s.config, v2=v2, v3=v3, sector=sector, sector2=sector2, sector2b=sector2b, sector2c=sector2c, stop_freq=stop_freq)

    logger.info("Loading data...")
    macro, prices = s.load()

    logger.info("Running backtest...")
    results = s.run(macro, prices)

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


def cmd_weights(v2: bool = False, v3: bool = False, sector: bool = False, sector2: bool = False, sector2b: bool = False, sector2c: bool = False, stop_freq: str = "daily"):
    s = _get_strategy(v2, v3, sector, sector2, sector2b, sector2c, stop_freq=stop_freq)

    _load_best(s.config, v2=v2, v3=v3, sector=sector, sector2=sector2, sector2b=sector2b, sector2c=sector2c, stop_freq=stop_freq)
    logger.info("Loading data...")
    macro, prices = s.load()

    results  = s.run(macro, prices)
    weights  = results["weights"]
    signal_w = weights.iloc[-1]
    as_of    = weights.index[-1].date()

    cfg   = s.config
    eff_w = s.eff_weights(signal_w, prices[cfg.ETF_UNIVERSE])
    eff_w = _validate_weights(eff_w, label="effective weights")

    label = " SECTOR2c" if sector2c else (" SECTOR2b" if sector2b else (" SECTOR2" if sector2 else (" SECTOR" if sector else (" V3" if v3 else (" V2" if v2 else "")))))
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
    if sector2c or sector2b or sector2:
        rebal_word = "weekly" if (sector2c or sector2b) else "monthly"
        rebal_freq = getattr(cfg, "REBALANCE_FREQ", "W" if (sector2c or sector2b) else "ME")
        print(f"Trailing stop: adaptive (tactical {cfg.STOP_TACTICAL:.0%} → supercycle {cfg.STOP_SUPERCYCLE:.0%}), {cfg.TRAILING_STOP_WINDOW}-day peak  |  rebalance: {rebal_freq}")
        if sector2c:
            from strategy_sector_v2c.backtest import compute_stop_pcts
        elif sector2b:
            from strategy_sector_v2b.backtest import compute_stop_pcts
        else:
            from strategy_sector_v2.backtest import compute_stop_pcts
        stop_df = compute_stop_pcts(eff_w, prices[cfg.ETF_UNIVERSE])
        if not stop_df.empty:
            print(f"\n{'='*65}")
            print(f"NORDNET TRAILING STOPS  (set these at rebalance — cancel old ones first)")
            print(f"{'='*65}")
            print(f"  {'ETF':>5s}  {'Weight':>7s}  {'12M ret':>8s}  {'Stop%':>6s}  {'Stop price':>11s}  {'Margin':>7s}")
            print(f"  {'─'*60}")
            for etf, row in stop_df.iterrows():
                w = eff_w.get(etf, 0.0)
                print(
                    f"  {etf:>5s}  {w:7.2%}  {row['m12']:>8.1%}  "
                    f"{row['stop_pct']:>6.1%}  "
                    f"${row['stop_price']:>10.2f}  "
                    f"{row['pct_to_stop']:>6.1%}"
                )
            print(f"  {'─'*60}")
            print(f"  Stop price = {cfg.TRAILING_STOP_WINDOW}-day peak × (1 − stop%)")
            print(f"  Margin     = how far today's price is above the stop level")
            print(f"  Use a FIXED stop loss in Nordnet, update at each {rebal_word} rebalance")
    else:
        print(f"Trailing stop: {cfg.TRAILING_STOP_PCT:.0%} below {cfg.TRAILING_STOP_WINDOW}-day peak")


def cmd_compare_sector():
    """Run Sector V2 (monthly) and V2b (weekly) with best params → sector_comparison.png."""
    import config_sector_v2 as cfg2, config_sector_v2b as cfg2b
    from data.pipeline_sector_v2  import load_all as load2
    from data.pipeline_sector_v2b import load_all as load2b
    from strategy_sector_v2.backtest  import run as run2
    from strategy_sector_v2b.backtest import run as run2b
    from analysis.performance import plot_sector_comparison

    _load_best(cfg2,  sector2=True)
    _load_best(cfg2b, sector2b=True)

    logger.info("Loading V2 sector data...")
    r2  = run2(load2())
    logger.info("Loading V2b sector data...")
    r2b = run2b(load2b())

    save_path = os.path.join(os.path.dirname(__file__), "sector_comparison.png")
    logger.info("Saving sector comparison chart...")
    plot_sector_comparison(r2, r2b, save_path=save_path)
    print(f"Saved → {save_path}")


def cmd_compare():
    """Run V1, V2, and V3 with best params and produce a side-by-side comparison chart."""
    import config_v2, config_v3
    from data.pipeline_v2     import load_all as load_all_v2
    from data.pipeline_v3     import load_all as load_all_v3
    from strategy_v2.backtest import run as run_v2
    from strategy_v3.backtest import run as run_v3

    logger.info("Loading V1 data...")
    _load_best(config, v2=False)
    macro1, prices1 = __import__("data.pipeline", fromlist=["load_all"]).load_all()
    results_v1 = __import__("strategy.backtest", fromlist=["run"]).run(macro1, prices1)

    logger.info("Loading V2 data...")
    _load_best(config_v2, v2=True)
    macro2, prices2 = load_all_v2()
    results_v2 = run_v2(macro2, prices2)

    logger.info("Loading V3 data...")
    _load_best(config_v3, v3=True)
    macro3, prices3 = load_all_v3()
    results_v3 = run_v3(macro3, prices3)

    save_path = os.path.join(os.path.dirname(__file__), "backtest_comparison.png")
    logger.info("Saving comparison chart...")
    plot_comparison(results_v1, results_v2, results_v3, save_path=save_path)


def cmd_trade(v2: bool = False, v3: bool = False, sector: bool = False, sector2: bool = False, sector2b: bool = False, sector2c: bool = False, dry_run: bool = False, stop_freq: str = "daily"):
    from broker.ibkr_client import IBKRClient

    s = _get_strategy(v2, v3, sector, sector2, sector2b, sector2c, stop_freq=stop_freq)
    cfg = s.config
    _validate_env(cfg)
    _load_best(cfg, v2=v2, v3=v3, sector=sector, sector2=sector2, sector2b=sector2b, sector2c=sector2c, stop_freq=stop_freq)

    logger.info("Loading data...")
    macro, prices = s.load()

    logger.info("Computing effective weights...")
    results  = s.run(macro, prices)
    signal_w = results["weights"].iloc[-1]
    as_of    = results["weights"].index[-1].date()
    eff_w    = s.eff_weights(signal_w, prices[cfg.ETF_UNIVERSE])
    eff_w    = _validate_weights(eff_w, label="effective weights")

    label = " SECTOR2c" if sector2c else (" SECTOR2b" if sector2b else (" SECTOR2" if sector2 else (" SECTOR" if sector else (" V3" if v3 else (" V2" if v2 else "")))))
    logger.info("Target weights%s as of %s:", label, as_of)
    for etf, w in eff_w.sort_values(ascending=False).items():
        if w > 0.001:
            logger.info("  %s  %.2f%%", etf, w * 100)

    client = IBKRClient()
    client.connect()

    try:
        net_liq = client.get_net_liq()
        logger.info("Net liquidation value: $%s", f"{net_liq:,.0f}")

        current_shares = client.get_positions()
        logger.info("Current positions: %s", current_shares)

        all_tickers = list(set(eff_w.index.tolist()) | set(current_shares.keys()))
        logger.info("Fetching prices for %d tickers...", len(all_tickers))
        live_prices = client.get_prices(all_tickers)

        orders = client.build_rebalance_orders(eff_w, net_liq, current_shares, live_prices)

        if not orders:
            print("\nNo orders needed — portfolio is already at target weights.")
            return

        client.print_preview(orders, net_liq)

        if dry_run:
            print("\n[dry-run] No orders submitted.")
        else:
            confirm = input("\nSubmit orders? [y/N]: ").strip().lower()
            if confirm == "y":
                client.submit_orders(orders)
            else:
                print("Aborted — no orders submitted.")
    finally:
        client.disconnect()


def cmd_optimize(n_trials: int = 300, v2: bool = False, v3: bool = False, sector: bool = False, sector2: bool = False, sector2b: bool = False, sector2c: bool = False, stop_freq: str = "daily"):
    from optimize import run_optimization
    run_optimization(n_trials=n_trials, v2=v2, v3=v3, sector=sector, sector2=sector2, sector2b=sector2b, sector2c=sector2c, stop_freq=stop_freq)


def main():
    parser = argparse.ArgumentParser(prog="main.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd")

    p_fetch = sub.add_parser("fetch")
    p_fetch.add_argument("--v2",       action="store_true", help="Use v2 strategy")
    p_fetch.add_argument("--v3",       action="store_true", help="Use v3 strategy")
    p_fetch.add_argument("--sector",   action="store_true", help="Use sector V1 strategy")
    p_fetch.add_argument("--sector2",  action="store_true", help="Use sector V2 strategy")
    p_fetch.add_argument("--sector2b", action="store_true", help="Use sector V2b strategy")
    p_fetch.add_argument("--sector2c", action="store_true", help="Use sector V2c strategy (cross-asset + cluster caps)")

    _STOP_FREQ_HELP = "Stop-loss cadence for sector2 (daily/weekly/monthly, default: daily)"

    p_bt = sub.add_parser("backtest")
    p_bt.add_argument("--best",       action="store_true", help="Use optimised params")
    p_bt.add_argument("--v2",         action="store_true", help="Use v2 strategy")
    p_bt.add_argument("--v3",         action="store_true", help="Use v3 strategy")
    p_bt.add_argument("--sector",     action="store_true", help="Use sector V1 strategy")
    p_bt.add_argument("--sector2",    action="store_true", help="Use sector V2 strategy")
    p_bt.add_argument("--sector2b",   action="store_true", help="Use sector V2b strategy (weekly, expanded)")
    p_bt.add_argument("--sector2c",   action="store_true", help="Use sector V2c strategy (cross-asset + cluster caps)")
    p_bt.add_argument("--stop-freq",  default="daily", choices=["daily","weekly","monthly"],
                      dest="stop_freq", help=_STOP_FREQ_HELP)

    p_wt = sub.add_parser("weights")
    p_wt.add_argument("--v2",         action="store_true", help="Use v2 strategy")
    p_wt.add_argument("--v3",         action="store_true", help="Use v3 strategy")
    p_wt.add_argument("--sector",     action="store_true", help="Use sector V1 strategy")
    p_wt.add_argument("--sector2",    action="store_true", help="Use sector V2 strategy")
    p_wt.add_argument("--sector2b",   action="store_true", help="Use sector V2b strategy")
    p_wt.add_argument("--sector2c",   action="store_true", help="Use sector V2c strategy")
    p_wt.add_argument("--stop-freq",  default="daily", choices=["daily","weekly","monthly"],
                      dest="stop_freq", help=_STOP_FREQ_HELP)

    p_trade = sub.add_parser("trade", help="Execute rebalance via IBKR Gateway")
    p_trade.add_argument("--v2",        action="store_true", help="Use v2 strategy")
    p_trade.add_argument("--v3",        action="store_true", help="Use v3 strategy")
    p_trade.add_argument("--sector",    action="store_true", help="Use sector V1 strategy")
    p_trade.add_argument("--sector2",   action="store_true", help="Use sector V2 strategy")
    p_trade.add_argument("--sector2b",  action="store_true", help="Use sector V2b strategy")
    p_trade.add_argument("--sector2c",  action="store_true", help="Use sector V2c strategy")
    p_trade.add_argument("--stop-freq", default="daily", choices=["daily","weekly","monthly"],
                         dest="stop_freq", help=_STOP_FREQ_HELP)
    p_trade.add_argument("--dry-run",   action="store_true", dest="dry_run",
                         help="Show order preview without submitting")

    sub.add_parser("compare")
    sub.add_parser("compare-sector", help="Sector V2 vs V2b comparison chart → sector_comparison.png")

    p_opt = sub.add_parser("optimize")
    p_opt.add_argument("--trials",    type=int, default=300)
    p_opt.add_argument("--v2",        action="store_true", help="Optimise v2 strategy")
    p_opt.add_argument("--v3",        action="store_true", help="Optimise v3 strategy")
    p_opt.add_argument("--sector",    action="store_true", help="Optimise sector V1 strategy")
    p_opt.add_argument("--sector2",   action="store_true", help="Optimise sector V2 strategy")
    p_opt.add_argument("--sector2b",  action="store_true", help="Optimise sector V2b strategy")
    p_opt.add_argument("--sector2c",  action="store_true", help="Optimise sector V2c strategy")
    p_opt.add_argument("--stop-freq", default="daily", choices=["daily","weekly","monthly"],
                       dest="stop_freq", help=_STOP_FREQ_HELP)

    args = parser.parse_args()

    v2        = getattr(args, "v2",        False)
    v3        = getattr(args, "v3",        False)
    sector    = getattr(args, "sector",    False)
    sector2   = getattr(args, "sector2",   False)
    sector2b  = getattr(args, "sector2b",  False)
    sector2c  = getattr(args, "sector2c",  False)
    stop_freq = getattr(args, "stop_freq", "daily")
    cfg = _get_strategy(v2, v3, sector, sector2, sector2b, sector2c, stop_freq=stop_freq).config

    _setup_logging(v2=v2, v3=v3, sector2=sector2 or sector2b or sector2c)

    if args.cmd in ("fetch", "weights", "optimize", "trade") and not sector and not sector2 and not sector2b and not sector2c:
        _validate_env(cfg)

    if   args.cmd == "fetch":    cmd_fetch(v2=v2, v3=v3, sector=sector, sector2=sector2, sector2b=sector2b, sector2c=sector2c)
    elif args.cmd == "backtest": cmd_backtest(use_best=args.best, v2=v2, v3=v3, sector=sector, sector2=sector2, sector2b=sector2b, sector2c=sector2c, stop_freq=stop_freq)
    elif args.cmd == "weights":  cmd_weights(v2=v2, v3=v3, sector=sector, sector2=sector2, sector2b=sector2b, sector2c=sector2c, stop_freq=stop_freq)
    elif args.cmd == "trade":    cmd_trade(v2=v2, v3=v3, sector=sector, sector2=sector2, sector2b=sector2b, sector2c=sector2c, dry_run=args.dry_run, stop_freq=stop_freq)
    elif args.cmd == "compare":         cmd_compare()
    elif args.cmd == "compare-sector":  cmd_compare_sector()
    elif args.cmd == "optimize": cmd_optimize(n_trials=args.trials, v2=v2, v3=v3, sector=sector, sector2=sector2, sector2b=sector2b, sector2c=sector2c, stop_freq=stop_freq)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
