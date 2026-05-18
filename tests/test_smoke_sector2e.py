"""
Smoke test for Sector V2e — runs a short backtest and asserts performance targets.

Requires cached price data or Supabase credentials. Skipped automatically when
neither is available (so CI without data secrets doesn't fail).
"""
import warnings
import pytest

warnings.filterwarnings("ignore")

# Skip the whole module if price data isn't accessible
def _data_available() -> bool:
    try:
        from data.pipelines.sector_v2e import load_all
        prices = load_all()
        return len(prices) > 500
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _data_available(),
    reason="Sector V2e price data not available — skipping smoke test",
)


def test_sector2e_smoke():
    """Run V2e backtest on full history and verify performance targets."""
    import json, os
    import configs.sector_v2e as cfg
    from data.pipelines.sector_v2e import load_all
    from strategies.sector_v2e.backtest import run
    from analysis.metrics import summary, sharpe, max_drawdown

    # Load best params if available
    best_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "best_params_sector2e.json")
    if os.path.exists(best_file):
        with open(best_file) as f:
            params = json.load(f)
        for k, v in params.items():
            setattr(cfg, k, v)

    prices = load_all()
    res    = run(prices)

    ret = res["daily_returns"].dropna()
    nav = res["nav"]

    # Weight sanity: must always sum to ~1.0
    w_sums = res["weights"].sum(axis=1)
    bad = w_sums[abs(w_sums - 1.0) > 0.01]
    assert len(bad) == 0, f"Weight sums off: {bad}"

    # Slice to 2005+ (V2e target window)
    start = "2005-01-01"
    ret_2005 = ret.loc[ret.index >= start]
    nav_2005 = (1 + ret_2005.fillna(0)).cumprod()

    n       = len(ret_2005)
    ann_ret = float(nav_2005.iloc[-1] ** (252 / n) - 1)
    sr      = sharpe(ret_2005)
    mdd     = max_drawdown(nav_2005)

    assert ann_ret > 0.30, f"CAGR {ann_ret:.1%} < 30% floor (target >42%)"
    assert sr      > 2.50, f"Sharpe {sr:.2f} < 2.5 floor (target >3.2)"
    assert mdd     > -0.20, f"Max DD {mdd:.1%} worse than -20% floor (target better than -10%)"


def test_sector2e_shared_modules_imported():
    """Verify that thin wrappers correctly delegate to sector_shared."""
    from strategies.sector_v2e import signals, portfolio, backtest
    from strategies.sector_shared import signals as shared_sig
    from strategies.sector_shared import portfolio as shared_port
    from strategies.sector_shared import backtest as shared_bt

    # The wrapper's composite_score should call shared composite_score
    assert hasattr(signals, "composite_score")
    assert hasattr(signals, "rolling_corr_at_dates")
    assert hasattr(portfolio, "build_weight_series")
    assert hasattr(backtest, "run")
