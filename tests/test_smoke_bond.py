"""
Smoke tests for bond strategies V1 / V2 / V3.

Requires FRED_API_KEY env var and either cached CSV data or Supabase credentials.
Skipped automatically when neither is available.
"""
import warnings
import pytest

warnings.filterwarnings("ignore")


def _data_available() -> bool:
    """True if cached bond price/macro data can be loaded without a network call."""
    import os
    try:
        import configs.bond_v1 as cfg
        cache_dir = cfg.DATA_DIR
        return (
            os.path.exists(os.path.join(cache_dir, "etf_prices.csv")) and
            os.path.exists(os.path.join(cache_dir, "fred_data.csv"))
        )
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _data_available(),
    reason="Bond cached data not available — skipping smoke tests",
)


def _run_bond_version(version: str):
    """Load, run, and return (results, ret_2011, nav_2011) for a bond version."""
    import json, os, importlib
    cfg = importlib.import_module(f"configs.bond_{version}")
    pipeline = importlib.import_module(f"data.pipelines.bond_{version}")
    backtest = importlib.import_module(f"strategies.bond_{version}.backtest")
    from analysis.performance import sharpe, max_drawdown

    best_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), f"best_params{'_' + version if version != 'v1' else ''}.json")
    if os.path.exists(best_file):
        with open(best_file) as f:
            params = json.load(f)
        for k, v in params.items():
            setattr(cfg, k, v)

    macro, prices = pipeline.load_all()
    results = backtest.run(macro, prices)
    return results


def _check_results(results: dict, label: str, min_cagr: float, min_sharpe: float, max_dd_floor: float):
    """Assert weight sanity and performance floors on 2011+ slice."""
    from analysis.performance import sharpe, max_drawdown

    # Weight sums must be ~1.0
    sums = results["weights"].sum(axis=1)
    bad  = sums[abs(sums - 1.0) > 0.01]
    assert len(bad) == 0, f"{label}: weight sums broken on {len(bad)} dates"

    ret = results["daily_returns"].dropna()
    nav = results["nav"]

    # Slice to 2011+
    ret_2011 = ret.loc[ret.index >= "2011-01-01"]
    nav_2011 = (1 + ret_2011.fillna(0)).cumprod()
    n        = len(ret_2011)
    ann_ret  = float(nav_2011.iloc[-1] ** (252 / n) - 1)
    sr       = sharpe(ret_2011)
    mdd      = max_drawdown(nav_2011)

    assert ann_ret > min_cagr,  f"{label}: CAGR {ann_ret:.1%} < {min_cagr:.0%} floor"
    assert sr      > min_sharpe, f"{label}: Sharpe {sr:.2f} < {min_sharpe} floor"
    assert mdd     > max_dd_floor, f"{label}: Max DD {mdd:.1%} worse than {max_dd_floor:.0%} floor"


def test_bond_v1_smoke():
    results = _run_bond_version("v1")
    _check_results(results, "Bond V1", min_cagr=0.12, min_sharpe=1.8, max_dd_floor=-0.20)


def test_bond_v2_smoke():
    results = _run_bond_version("v2")
    _check_results(results, "Bond V2", min_cagr=0.10, min_sharpe=1.5, max_dd_floor=-0.25)


def test_bond_v3_smoke():
    results = _run_bond_version("v3")
    _check_results(results, "Bond V3", min_cagr=0.10, min_sharpe=1.5, max_dd_floor=-0.25)


def test_bond_v1_weight_sanity_standalone():
    """Verify weight sum constraint without loading best params."""
    import configs.bond_v1 as cfg
    from data.pipelines.bond_v1 import load_all
    from strategies.bond_v1.backtest import run

    macro, prices = load_all()
    results = run(macro, prices)

    sums = results["weights"].sum(axis=1)
    bad  = sums[abs(sums - 1.0) > 0.01]
    assert len(bad) == 0, f"Weight sums broken: {bad}"
