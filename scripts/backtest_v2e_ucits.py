"""
Backtest sector_v2e assuming UCITS-tradeable universe.

Methodology:
- Drop XBI and IGV (no acceptable UCITS equivalent on Nordnet)
- Use US ETF price data as proxy for UCITS (most UCITS variants track the
  identical underlying index — see memory: sector_v2e_ucits_mapping)
- Apply additional annualized TER drag (~15bp) to capture UCITS expense
  difference vs US sector ETFs

This is NOT a true UCITS backtest (UCITS funds mostly post-date 2010, many
post-date 2020). It is a "what-if" estimate of how the strategy would have
behaved on the UCITS-tradeable subset of the universe, with realistic cost.

Run:
    python scripts/backtest_v2e_ucits.py
"""
import json
import warnings
import sys
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import configs.sector_v2e as config
from data.pipelines.sector_v2e import load_all
from strategies.sector_v2e.backtest import run
from analysis.performance import summary


EXCLUDED = {"XBI", "IGV"}             # no UCITS equivalent
EXTRA_TER_ANNUAL = 0.0015              # +15bp/yr UCITS TER drag estimate
PARAMS_FILE = ROOT / "best_params_sector2e.json"


def patch_universe(cfg, excluded: set[str]) -> None:
    """Remove excluded tickers from all universe-related config lists."""
    cfg.ETF_UNIVERSE        = [t for t in cfg.ETF_UNIVERSE        if t not in excluded]
    cfg.ALL_TICKERS         = [t for t in cfg.ALL_TICKERS         if t not in excluded]
    cfg.TRAILING_STOP_ETFS  = [t for t in cfg.TRAILING_STOP_ETFS  if t not in excluded]

    cfg.CLUSTERS = {
        k: [t for t in v if t not in excluded]
        for k, v in cfg.CLUSTERS.items()
    }
    cfg.CLUSTERS = {k: v for k, v in cfg.CLUSTERS.items() if v}  # drop empty
    cfg.ETF_TO_CLUSTER = {
        etf: cluster
        for cluster, etfs in cfg.CLUSTERS.items()
        for etf in etfs
    }


def apply_best_params(cfg, params_path: Path) -> dict:
    """Patch best params onto config the same way Optuna does."""
    params = json.loads(params_path.read_text())
    for k, v in params.items():
        setattr(cfg, k, v)
    return params


def apply_ter_drag(daily_returns: pd.Series, ter_annual: float) -> pd.Series:
    """Subtract daily-equivalent TER drag from daily returns."""
    daily_drag = ter_annual / 252.0
    return daily_returns - daily_drag


def rebuild_nav(daily_returns: pd.Series) -> pd.Series:
    return (1.0 + daily_returns).cumprod()


def slice_window(series: pd.Series, start: str, end: str | None = None) -> pd.Series:
    if end is None:
        return series.loc[start:]
    return series.loc[start:end]


def main():
    print("="*80)
    print("Sector V2e — UCITS-tradeable variant backtest")
    print("="*80)

    # 1) Baseline V2e (US ETFs, full universe) — run first BEFORE patching config
    print("\n[1/3] Running baseline V2e (full US universe)...")
    apply_best_params(config, PARAMS_FILE)
    full_prices = load_all()
    baseline = run(full_prices)
    base_ret = baseline["daily_returns"]
    base_nav = baseline["nav"]

    # 2) UCITS variant: drop XBI, IGV from universe
    print("[2/3] Running UCITS variant (drop XBI, IGV)...")
    patch_universe(config, EXCLUDED)

    ucits_prices = full_prices[[c for c in full_prices.columns if c not in EXCLUDED]]
    ucits = run(ucits_prices)
    ucits_ret = ucits["daily_returns"]

    # 3) UCITS variant + TER drag
    print("[3/3] Applying +15bp annual TER drag...")
    ucits_ret_after_ter = apply_ter_drag(ucits_ret, EXTRA_TER_ANNUAL)
    ucits_nav_after_ter = rebuild_nav(ucits_ret_after_ter)

    # ───── Summaries ─────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON (full available window)")
    print("="*80)

    print("\n--- Baseline V2e (US ETFs, full universe) ---")
    print(summary(base_ret, base_nav, "Baseline V2e"))

    print("\n--- UCITS variant (drop XBI, IGV, no TER drag) ---")
    print(summary(ucits_ret, ucits["nav"], "UCITS no-cost"))

    print("\n--- UCITS variant + 15bp/yr TER drag (realistic live) ---")
    print(summary(ucits_ret_after_ter, ucits_nav_after_ter, "UCITS live"))

    # ───── Sub-period analysis ───────────────────────────────────────────────
    print("\n" + "="*80)
    print("SUB-PERIOD ANALYSIS")
    print("="*80)

    windows = [
        ("2000-2010", "2000-01-01", "2009-12-31"),
        ("2010-2020", "2010-01-01", "2019-12-31"),
        ("2020-2026", "2020-01-01", None),
        ("Full",      "2000-01-01", None),
    ]
    for label, s, e in windows:
        b_ret  = slice_window(base_ret,  s, e)
        u_ret  = slice_window(ucits_ret_after_ter, s, e)
        if b_ret.empty or u_ret.empty:
            continue
        print(f"\n--- {label} ---")
        print(summary(b_ret, rebuild_nav(b_ret), "Baseline V2e"))
        print(summary(u_ret, rebuild_nav(u_ret), "UCITS + TER"))

    # ───── Year-by-year ───────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("YEAR-BY-YEAR RETURNS")
    print("="*80)
    yearly_base  = base_ret.resample("YE").apply(lambda r: (1+r).prod()-1)
    yearly_ucits = ucits_ret_after_ter.resample("YE").apply(lambda r: (1+r).prod()-1)
    yearly = pd.DataFrame({
        "Baseline":  yearly_base,
        "UCITS+TER": yearly_ucits,
        "Diff (pp)": (yearly_ucits - yearly_base) * 100,
    })
    yearly.index = yearly.index.year
    print(yearly.to_string(float_format=lambda x: f"{x:+.2%}"))


if __name__ == "__main__":
    main()
