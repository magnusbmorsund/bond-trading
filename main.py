"""
Bond / Sector Rotation Strategy — CLI entry point.

Usage:
  python main.py fetch    [STRATEGY]
  python main.py backtest [STRATEGY] [--best] [--stop-freq daily|weekly|monthly]
  python main.py weights  [STRATEGY]
  python main.py optimize [STRATEGY] [--trials N]
  python main.py compare
  python main.py compare-sector
  python main.py v2c-long

Available strategies (default: v1):
  v1        Bond rotation V1  (FRED macro signals)
  v2        Bond rotation V2  (USD/ISM signals added)
  v3        Bond rotation V3  (EDV/DBMF/MTUM, growth composite)
  sector    Sector V1         (XL series, single-lookback momentum)
  sector2   Sector V2         (35 ETFs, multi-timescale, adaptive stops)
  sector2b  Sector V2b        (weekly rebalance, expanded universe)
  sector2c  Sector V2c        (cross-asset + correlation filter + cluster caps)
  sector2d  Sector V2d        (V2c universe, liquid ETFs only ≥$100M/day)
  sector2e  Sector V2e        (V2d + 24m/36m supercycle momentum lookbacks)
"""
import sys
import os
import json
import time
import logging
import argparse
import importlib
from dataclasses import dataclass
from typing import Any, Callable
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from analysis.performance import (
    print_summary_table, plot_results, plot_annual_stats,
    plot_annual_allocations, plot_comparison, plot_v2d_v2e, plot_v2c_v2d_v2e,
)

logger = logging.getLogger(__name__)

_BASE = os.path.dirname(__file__)


# ---------------------------------------------------------------------------
# Strategy registry — add new strategies here and nowhere else in this file
# ---------------------------------------------------------------------------

@dataclass
class StrategySpec:
    label: str
    config_path: str
    pipeline_path: str
    backtest_path: str
    params_suffix: str           # appended to "best_params" to find the JSON file
    needs_fred: bool = False     # requires FRED_API_KEY env var
    has_stop_freq: bool = False  # sector2 only: supports --stop-freq weekly/monthly


REGISTRY: dict[str, StrategySpec] = {
    "v1":       StrategySpec("Bond V1",    "configs.bond_v1",    "data.pipelines.bond_v1",    "strategies.bond_v1.backtest",    "",           needs_fred=True),
    "v2":       StrategySpec("Bond V2",    "configs.bond_v2",    "data.pipelines.bond_v2",    "strategies.bond_v2.backtest",    "_v2",        needs_fred=True),
    "v3":       StrategySpec("Bond V3",    "configs.bond_v3",    "data.pipelines.bond_v3",    "strategies.bond_v3.backtest",    "_v3",        needs_fred=True),
    "sector":   StrategySpec("Sector V1",  "configs.sector_v1",  "data.pipelines.sector_v1",  "strategies.sector_v1.backtest",  "_sector"),
    "sector2":  StrategySpec("Sector V2",  "configs.sector_v2",  "data.pipelines.sector_v2",  "strategies.sector_v2.backtest",  "_sector2",   has_stop_freq=True),
    "sector2b": StrategySpec("Sector V2b", "configs.sector_v2b", "data.pipelines.sector_v2b", "strategies.sector_v2b.backtest", "_sector2b"),
    "sector2c": StrategySpec("Sector V2c", "configs.sector_v2c", "data.pipelines.sector_v2c", "strategies.sector_v2c.backtest", "_sector2c"),
    "sector2d": StrategySpec("Sector V2d", "configs.sector_v2d", "data.pipelines.sector_v2d", "strategies.sector_v2d.backtest", "_sector2d"),
    "sector2e": StrategySpec("Sector V2e", "configs.sector_v2e", "data.pipelines.sector_v2e", "strategies.sector_v2e.backtest", "_sector2e"),
    "sector2f": StrategySpec("Sector V2f", "configs.sector_v2f", "data.pipelines.sector_v2f", "strategies.sector_v2f.backtest", "_sector2f"),
}

STRATEGY_CHOICES = list(REGISTRY.keys())


@dataclass
class Strategy:
    config: Any
    load: Callable        # (force=False) -> (macro_or_None, prices)
    run: Callable         # (macro, prices) -> results dict
    eff_weights: Callable
    spec: StrategySpec


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _resolve(strategy: str, stop_freq: str = "daily") -> Strategy:
    """Import and wire up a strategy from the registry. Single point of assembly."""
    spec = REGISTRY[strategy]
    cfg_mod  = importlib.import_module(spec.config_path)
    pipe_mod = importlib.import_module(spec.pipeline_path)
    bt_mod   = importlib.import_module(spec.backtest_path)

    _load_fn = pipe_mod.load_all
    _run_fn  = bt_mod.run
    _eff_fn  = bt_mod.effective_weights

    if spec.needs_fred:
        # Bond pipelines return (macro, prices)
        def load_wrap(force=False): return _load_fn(force=force)
        def run_wrap(macro, prices): return _run_fn(macro, prices)
    elif spec.has_stop_freq:
        _sf = stop_freq
        def load_wrap(force=False): return None, _load_fn(force=force)
        def run_wrap(macro, prices): return _run_fn(prices, stop_freq=_sf)
    else:
        def load_wrap(force=False): return None, _load_fn(force=force)
        def run_wrap(macro, prices): return _run_fn(prices)

    return Strategy(cfg_mod, load_wrap, run_wrap, _eff_fn, spec)


def _params_path(strategy: str, stop_freq: str = "daily") -> str:
    spec = REGISTRY[strategy]
    if spec.has_stop_freq and stop_freq in ("weekly", "monthly"):
        suffix = f"_sector2_{stop_freq}"
    else:
        suffix = spec.params_suffix
    return os.path.join(_BASE, f"best_params{suffix}.json")


def _load_best(cfg, strategy: str, stop_freq: str = "daily"):
    path = _params_path(strategy, stop_freq)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{os.path.basename(path)} not found — run:\n"
            f"  python main.py optimize {strategy}"
        )
    with open(path) as f:
        params = json.load(f)
    for k, v in params.items():
        setattr(cfg, k, v)
    logger.info("Loaded %d optimised params from %s", len(params), os.path.basename(path))

    # Issue #7: warn if params file is stale (>60 days) — regime may have changed
    age_days = (time.time() - os.path.getmtime(path)) / 86400
    if age_days > 60:
        logger.warning(
            "%s is %.0f days old — consider re-optimising: python main.py optimize %s",
            os.path.basename(path), age_days, strategy,
        )


def _validate_env(cfg):
    key = getattr(cfg, "FRED_API_KEY", None)
    if key is None:
        return
    if not key:
        logger.error("FRED_API_KEY is not set. Export it before running.")
        sys.exit(1)
    if len(key) != 32:
        logger.warning("FRED_API_KEY looks malformed (%d chars). Fetches will fall back to cache.", len(key))


def _validate_weights(weights: pd.Series, label: str = "weights") -> pd.Series:
    if weights.isna().any():
        logger.warning("NaN weights replaced with 0 for: %s", weights[weights.isna()].index.tolist())
        weights = weights.fillna(0.0)
    if (weights < 0).any():
        logger.warning("Negative weights clipped to 0 for: %s", weights[weights < 0].index.tolist())
        weights = weights.clip(lower=0.0)
    total = weights.sum()
    if abs(total - 1.0) > 0.005:
        logger.warning("%s sum=%.4f — renormalising to 1.0", label, total)
        weights = weights / total
    return weights


def _setup_logging(strategy: str):
    spec   = REGISTRY.get(strategy, REGISTRY["v1"])
    log_dir = os.path.join(_BASE, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"strategy{spec.params_suffix}.log")
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
        ],
    )
    for noisy in ("yfinance", "urllib3", "peewee", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_fetch(strategy: str):
    s = _resolve(strategy)
    if s.spec.needs_fred:
        _validate_env(s.config)
    logger.info("Force-refreshing %s data...", s.spec.label)
    macro, prices = s.load(force=True)
    if macro is not None:
        logger.info("Done. macro=%s  prices=%s  range=%s → %s",
                    macro.shape, prices.shape,
                    macro.index[0].date(), macro.index[-1].date())
    else:
        logger.info("Done. prices=%s  range=%s → %s",
                    prices.shape, prices.index[0].date(), prices.index[-1].date())


def cmd_backtest(strategy: str, use_best: bool = False, stop_freq: str = "daily"):
    s      = _resolve(strategy, stop_freq)
    suffix = REGISTRY[strategy].params_suffix
    if REGISTRY[strategy].has_stop_freq and stop_freq in ("weekly", "monthly"):
        suffix = f"_sector2_{stop_freq}"

    if use_best:
        _load_best(s.config, strategy, stop_freq)

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

    logger.info("Saving charts...")
    plot_results(results,            save_path=os.path.join(_BASE, f"backtest_results{suffix}.png"))
    plot_annual_stats(results,       save_path=os.path.join(_BASE, f"annual_stats{suffix}.png"))
    plot_annual_allocations(results, save_path=os.path.join(_BASE, f"annual_allocations{suffix}.png"))


def cmd_weights(strategy: str, stop_freq: str = "daily", ucits: bool = False):
    s = _resolve(strategy, stop_freq)
    _load_best(s.config, strategy, stop_freq)

    logger.info("Loading data...")
    macro, prices = s.load()

    results  = s.run(macro, prices)
    weights  = results["weights"]
    signal_w = weights.iloc[-1]
    as_of    = weights.index[-1].date()
    cfg      = s.config

    etf_cols = getattr(cfg, "ETF_UNIVERSE", list(prices.columns))
    eff_w    = s.eff_weights(signal_w, prices[etf_cols])
    eff_w    = _validate_weights(eff_w, label="effective weights")

    print(f"\n{'='*45}")
    print(f"SIGNAL WEIGHTS [{s.spec.label}]  (model, as of {as_of})")
    print(f"{'='*45}")
    for etf, w in signal_w.sort_values(ascending=False).items():
        if w > 0.001:
            stopped = " [STOPPED OUT]" if eff_w.get(etf, w) < w * 0.5 else ""
            print(f"  {etf:>4s}  {w:6.2%}  {'█' * int(w * 40)}{stopped}")

    print(f"\n{'='*45}")
    print(f"EFFECTIVE POSITIONS [{s.spec.label}]  (after trailing stops — trade these)")
    print(f"{'='*45}")
    for etf, w in eff_w.sort_values(ascending=False).items():
        if w > 0.001:
            print(f"  {etf:>4s}  {w:6.2%}  {'█' * int(w * 40)}")
    print(f"  {'─'*40}")
    print(f"  {'Sum':>4s}  {eff_w.sum():6.2%}")
    print(f"\nSet each ETF as % of total portfolio value above.")

    if hasattr(cfg, "STOP_TACTICAL"):
        rebal_freq = getattr(cfg, "REBALANCE_FREQ", "W")
        rebal_word = "monthly" if rebal_freq in ("ME", "MS") else "weekly"
        print(f"Trailing stop: adaptive (tactical {cfg.STOP_TACTICAL:.0%} → supercycle "
              f"{cfg.STOP_SUPERCYCLE:.0%}), {cfg.TRAILING_STOP_WINDOW}-day peak  |  rebalance: {rebal_freq}")
        bt_mod  = importlib.import_module(s.spec.backtest_path)
        stop_df = bt_mod.compute_stop_pcts(eff_w, prices[etf_cols])
        if not stop_df.empty:
            print(f"\n{'='*65}")
            print(f"NORDNET TRAILING STOPS  (cancel old ones first, then set these)")
            print(f"{'='*65}")
            print(f"  {'ETF':>5s}  {'Weight':>7s}  {'12M ret':>8s}  {'Stop%':>6s}  {'Stop price':>11s}  {'Margin':>7s}")
            print(f"  {'─'*60}")
            for etf, row in stop_df.iterrows():
                w = eff_w.get(etf, 0.0)
                print(f"  {etf:>5s}  {w:7.2%}  {row['m12']:>8.1%}  "
                      f"{row['stop_pct']:>6.1%}  ${row['stop_price']:>10.2f}  {row['pct_to_stop']:>6.1%}")
            print(f"  {'─'*60}")
            print(f"  Stop price = {cfg.TRAILING_STOP_WINDOW}-day peak × (1 − stop%)")
            print(f"  Margin     = how far today's price is above the stop level")
            print(f"  Use a FIXED stop loss in Nordnet, update at each {rebal_word} rebalance")
    else:
        print(f"Trailing stop: {cfg.TRAILING_STOP_PCT:.0%} below {cfg.TRAILING_STOP_WINDOW}-day peak")

    if ucits:
        _print_ucits_translation(eff_w, stop_df if hasattr(cfg, "STOP_TACTICAL") else None)


def _print_ucits_translation(eff_w, stop_df):
    """Translate effective positions to UCITS tickers for Nordnet retail."""
    from configs.ucits_mapping import to_ucits

    held = [(etf, w) for etf, w in eff_w.sort_values(ascending=False).items() if w > 0.001]
    if not held:
        return

    print(f"\n{'='*78}")
    print(f"UCITS EQUIVALENTS  (tradeable on Nordnet — use these, not the US tickers)")
    print(f"{'='*78}")
    print(f"  {'US':>5s}  {'UCITS':>7s}  {'Weight':>7s}  {'Stop%':>6s}  {'Exchange':>15s}  {'ISIN':>14s}  Status")
    print(f"  {'─'*98}")

    dropped = []
    for us_ticker, w in held:
        info = to_ucits(us_ticker)
        if info is None:
            dropped.append(us_ticker)
            print(f"  {us_ticker:>5s}  {'—':>7s}  {w:7.2%}  {'—':>6s}  {'—':>15s}  {'—':>14s}  NO UCITS — drop")
            continue

        stop_pct_str = "—"
        if stop_df is not None and us_ticker in stop_df.index:
            stop_pct_str = f"{stop_df.loc[us_ticker, 'stop_pct']:.1%}"

        print(f"  {us_ticker:>5s}  {info['ticker']:>7s}  {w:7.2%}  "
              f"{stop_pct_str:>6s}  {info['exchange']:>15s}  {info['isin']:>14s}  {info['status']}")

    print(f"  {'─'*98}")
    print(f"  status: direct = UCITS tracks same index   proxy = similar but style drift")

    if dropped:
        print(f"\n  ⚠ {len(dropped)} position(s) have no UCITS equivalent — redistribute weight to others:")
        for t in dropped:
            print(f"    • {t}")

    print(f"\n  For Nordnet:")
    print(f"    1. Search by ISIN to find the exact listing")
    print(f"    2. Set trailing stop ('glidende stop loss') with the Stop% above")
    print(f"    3. Stop trails from price at order placement — let it run, don't reset weekly")


def cmd_optimize(strategy: str, n_trials: int = 300, stop_freq: str = "daily"):
    from optimize import run_optimization
    flags = {
        "v2":       dict(v2=True),
        "v3":       dict(v3=True),
        "sector":   dict(sector=True),
        "sector2":  dict(sector2=True),
        "sector2b": dict(sector2b=True),
        "sector2c": dict(sector2c=True),
        "sector2d": dict(sector2d=True),
        "sector2e": dict(sector2e=True),
        "sector2f": dict(sector2f=True),
    }
    run_optimization(n_trials=n_trials, stop_freq=stop_freq, **flags.get(strategy, {}))


# ---------------------------------------------------------------------------
# One-off comparison commands
# ---------------------------------------------------------------------------

def cmd_v2c_v2d_v2e():
    """V2c vs V2d vs V2e three-way comparison → v2c_v2d_v2e.png."""
    import configs.sector_v2c as cfg2c
    import configs.sector_v2d as cfg2d
    import configs.sector_v2e as cfg2e
    from data.pipelines.sector_v2c import load_all as load2c
    from data.pipelines.sector_v2d import load_all as load2d
    from data.pipelines.sector_v2e import load_all as load2e
    from strategies.sector_v2c.backtest import run as run2c
    from strategies.sector_v2d.backtest import run as run2d
    from strategies.sector_v2e.backtest import run as run2e

    _load_best(cfg2c, "sector2c")
    _load_best(cfg2d, "sector2d")
    _load_best(cfg2e, "sector2e")

    logger.info("Loading V2c data...")
    r2c = run2c(load2c())
    logger.info("Loading V2d data...")
    r2d = run2d(load2d())
    logger.info("Loading V2e data...")
    r2e = run2e(load2e())

    save_path = os.path.join(_BASE, "v2c_v2d_v2e.png")
    plot_v2c_v2d_v2e(r2c, r2d, r2e, save_path=save_path)
    print(f"Saved → {save_path}")


def cmd_v2d_v2e():
    """V2d vs V2e supercycle comparison → v2d_v2e.png."""
    import configs.sector_v2d as cfg2d
    import configs.sector_v2e as cfg2e
    from data.pipelines.sector_v2d import load_all as load2d
    from data.pipelines.sector_v2e import load_all as load2e
    from strategies.sector_v2d.backtest import run as run2d
    from strategies.sector_v2e.backtest import run as run2e

    _load_best(cfg2d, "sector2d")
    _load_best(cfg2e, "sector2e")

    logger.info("Loading V2d data...")
    r2d = run2d(load2d())
    logger.info("Loading V2e data...")
    r2e = run2e(load2e())

    save_path = os.path.join(_BASE, "v2d_v2e.png")
    plot_v2d_v2e(r2d, r2e, save_path=save_path)
    print(f"Saved → {save_path}")


def cmd_v2c_long():
    """V2c + V2d extended backtests (2002–present) → v2c_extended.png."""
    import configs.sector_v2c as cfg2c
    import configs.sector_v2d as cfg2d
    from data.pipelines.sector_v2c import load_all as load2c
    from data.pipelines.sector_v2d import load_all as load2d
    from strategies.sector_v2c.backtest import run as run2c
    from strategies.sector_v2d.backtest import run as run2d
    from analysis.performance import plot_v2c_extended

    _load_best(cfg2c, "sector2c")
    logger.info("Loading V2c data...")
    r2c = run2c(load2c())

    try:
        _load_best(cfg2d, "sector2d")
    except FileNotFoundError:
        logger.warning("best_params_sector2d.json not found — V2d uses default params")
    logger.info("Loading V2d data...")
    r2d = run2d(load2d())

    save_path = os.path.join(_BASE, "v2c_extended.png")
    plot_v2c_extended(r2c, results_v2d=r2d, save_path=save_path)
    print(f"Saved → {save_path}")


def cmd_compare_sector():
    """Sector V2 / V2b / V2c comparison → sector_comparison.png."""
    import configs.sector_v2  as cfg2
    import configs.sector_v2b as cfg2b
    import configs.sector_v2c as cfg2c
    from data.pipelines.sector_v2  import load_all as load2
    from data.pipelines.sector_v2b import load_all as load2b
    from data.pipelines.sector_v2c import load_all as load2c
    from strategies.sector_v2.backtest  import run as run2
    from strategies.sector_v2b.backtest import run as run2b
    from strategies.sector_v2c.backtest import run as run2c
    from analysis.performance import plot_sector_comparison

    _load_best(cfg2,  "sector2")
    _load_best(cfg2b, "sector2b")
    _load_best(cfg2c, "sector2c")

    logger.info("Running V2/V2b/V2c backtests...")
    r2  = run2(load2())
    r2b = run2b(load2b())
    r2c = run2c(load2c())

    save_path = os.path.join(_BASE, "sector_comparison.png")
    plot_sector_comparison(r2, r2b, r2c, save_path=save_path)
    print(f"Saved → {save_path}")


def cmd_compare():
    """Bond V1 / V2 / V3 comparison → backtest_comparison.png."""
    import configs.bond_v1 as cfg1
    import configs.bond_v2 as cfg2
    import configs.bond_v3 as cfg3
    from data.pipelines.bond_v1 import load_all as load1
    from data.pipelines.bond_v2 import load_all as load2
    from data.pipelines.bond_v3 import load_all as load3
    from strategies.bond_v1.backtest import run as run1
    from strategies.bond_v2.backtest import run as run2
    from strategies.bond_v3.backtest import run as run3

    _load_best(cfg1, "v1"); r1 = run1(*load1())
    _load_best(cfg2, "v2"); r2 = run2(*load2())
    _load_best(cfg3, "v3"); r3 = run3(*load3())

    save_path = os.path.join(_BASE, "backtest_comparison.png")
    plot_comparison(r1, r2, r3, save_path=save_path)
    print(f"Saved → {save_path}")


# ---------------------------------------------------------------------------
# CLI — each subcommand shares one STRATEGY positional; no flag duplication
# ---------------------------------------------------------------------------

_STRATEGY_HELP  = f"Strategy name. Choices: {', '.join(STRATEGY_CHOICES)}  (default: v1)"
_STOP_FREQ_HELP = "Stop cadence for sector2 only: daily|weekly|monthly (default: daily)"


def main():
    parser = argparse.ArgumentParser(
        prog="main.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("fetch", help="Refresh market data cache")
    p.add_argument("strategy", nargs="?", default="v1", choices=STRATEGY_CHOICES, help=_STRATEGY_HELP)

    p = sub.add_parser("backtest", help="Run backtest and save charts")
    p.add_argument("strategy",    nargs="?", default="v1", choices=STRATEGY_CHOICES, help=_STRATEGY_HELP)
    p.add_argument("--best",      action="store_true", help="Load optimised params")
    p.add_argument("--stop-freq", default="daily", choices=["daily", "weekly", "monthly"],
                   dest="stop_freq", help=_STOP_FREQ_HELP)

    p = sub.add_parser("weights", help="Show today's signal and effective weights")
    p.add_argument("strategy",    nargs="?", default="v1", choices=STRATEGY_CHOICES, help=_STRATEGY_HELP)
    p.add_argument("--stop-freq", default="daily", choices=["daily", "weekly", "monthly"],
                   dest="stop_freq", help=_STOP_FREQ_HELP)
    p.add_argument("--ucits",     action="store_true",
                   help="Also print UCITS-equivalent tickers and ISINs for Nordnet retail trading")

    p = sub.add_parser("optimize", help="Run Optuna hyperparameter search")
    p.add_argument("strategy",    nargs="?", default="v1", choices=STRATEGY_CHOICES, help=_STRATEGY_HELP)
    p.add_argument("--trials",    type=int, default=300, help="Number of Optuna trials")
    p.add_argument("--stop-freq", default="daily", choices=["daily", "weekly", "monthly"],
                   dest="stop_freq", help=_STOP_FREQ_HELP)

    sub.add_parser("compare",        help="Bond V1/V2/V3 comparison → backtest_comparison.png")
    sub.add_parser("compare-sector", help="Sector V2/V2b/V2c comparison → sector_comparison.png")
    sub.add_parser("v2c-long",       help="V2c+V2d extended history (2002–present) → v2c_extended.png")
    sub.add_parser("v2d-v2e",        help="V2d vs V2e supercycle comparison → v2d_v2e.png")
    sub.add_parser("v2c-v2d-v2e",   help="V2c vs V2d vs V2e three-way comparison → v2c_v2d_v2e.png")

    args = parser.parse_args()
    if args.cmd is None:
        parser.print_help()
        return

    strategy  = getattr(args, "strategy",  "v1")
    stop_freq = getattr(args, "stop_freq", "daily")

    _setup_logging(strategy)

    if   args.cmd == "fetch":          cmd_fetch(strategy)
    elif args.cmd == "backtest":       cmd_backtest(strategy, use_best=args.best, stop_freq=stop_freq)
    elif args.cmd == "weights":        cmd_weights(strategy, stop_freq=stop_freq, ucits=getattr(args, "ucits", False))
    elif args.cmd == "optimize":       cmd_optimize(strategy, n_trials=args.trials, stop_freq=stop_freq)
    elif args.cmd == "compare":        cmd_compare()
    elif args.cmd == "compare-sector": cmd_compare_sector()
    elif args.cmd == "v2c-long":       cmd_v2c_long()
    elif args.cmd == "v2d-v2e":        cmd_v2d_v2e()
    elif args.cmd == "v2c-v2d-v2e":   cmd_v2c_v2d_v2e()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
