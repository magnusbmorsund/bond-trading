"""Tests for analysis.metrics — pure math, no data fetching."""
import numpy as np
import pandas as pd
import pytest

from analysis.metrics import sharpe, max_drawdown, calmar, drawdown_series, summary


@pytest.fixture
def flat_returns():
    """1% daily return every day."""
    idx = pd.date_range("2020-01-02", periods=252, freq="B")
    return pd.Series(0.01, index=idx)


@pytest.fixture
def nav_monotone(flat_returns):
    return (1 + flat_returns).cumprod()


@pytest.fixture
def crash_nav():
    """NAV that rises 50%, crashes 40% from peak, then recovers."""
    idx = pd.date_range("2020-01-02", periods=300, freq="B")
    prices = np.ones(300)
    prices[:100]  = np.linspace(1.0, 1.5, 100)   # rise to 1.5
    prices[100:150] = np.linspace(1.5, 0.9, 50)  # crash to 0.9 (−40% from peak)
    prices[150:] = np.linspace(0.9, 1.8, 150)    # recovery
    return pd.Series(prices, index=idx)


# ── sharpe ──────────────────────────────────────────────────────────────────

def test_sharpe_positive_for_positive_returns(flat_returns):
    assert sharpe(flat_returns) > 0


def test_sharpe_zero_for_zero_returns():
    idx = pd.date_range("2020-01-02", periods=252, freq="B")
    returns = pd.Series(0.0, index=idx)
    # std = 0, result should be NaN or inf — not crash
    result = sharpe(returns)
    assert np.isnan(result) or not np.isfinite(result)


def test_sharpe_negative_for_negative_returns():
    idx = pd.date_range("2020-01-02", periods=252, freq="B")
    returns = pd.Series(-0.005, index=idx)
    assert sharpe(returns) < 0


# ── max_drawdown ─────────────────────────────────────────────────────────────

def test_max_drawdown_monotone_nav_is_zero(nav_monotone):
    assert max_drawdown(nav_monotone) == pytest.approx(0.0, abs=1e-10)


def test_max_drawdown_crash_nav(crash_nav):
    mdd = max_drawdown(crash_nav)
    assert mdd < 0
    assert mdd == pytest.approx(-0.40, abs=0.02)


def test_max_drawdown_returns_float(nav_monotone):
    assert isinstance(max_drawdown(nav_monotone), float)


# ── calmar ───────────────────────────────────────────────────────────────────

def test_calmar_positive_for_profitable_strategy(flat_returns, nav_monotone):
    c = calmar(nav_monotone, flat_returns)
    assert c > 0


def test_calmar_inf_for_zero_drawdown(flat_returns, nav_monotone):
    # A monotone-rising nav has exactly 0 drawdown → calmar is +infinity
    # (return / 0). The sign must stay positive for a profitable strategy.
    c = calmar(nav_monotone, flat_returns)
    assert np.isinf(c) and c > 0


# ── drawdown_series ───────────────────────────────────────────────────────────

def test_drawdown_series_starts_at_zero(crash_nav):
    dd = drawdown_series(crash_nav)
    assert dd.iloc[0] == pytest.approx(0.0)


def test_drawdown_series_never_positive(crash_nav):
    dd = drawdown_series(crash_nav)
    assert (dd <= 1e-10).all()


def test_drawdown_series_min_matches_max_drawdown(crash_nav):
    dd  = drawdown_series(crash_nav)
    mdd = max_drawdown(crash_nav)
    assert dd.min() == pytest.approx(mdd)


# ── summary ───────────────────────────────────────────────────────────────────

def test_summary_returns_series(flat_returns, nav_monotone):
    s = summary(flat_returns, nav_monotone, "Test")
    assert isinstance(s, pd.Series)
    assert s.name == "Test"


def test_summary_has_expected_keys(flat_returns, nav_monotone):
    s = summary(flat_returns, nav_monotone)
    for key in ("Ann. Return", "Sharpe Ratio", "Max Drawdown", "Calmar Ratio",
                "Best Month", "Worst Month", "Total Return", "Start", "End"):
        assert key in s.index


def test_summary_ann_return_positive_for_positive_rets(flat_returns, nav_monotone):
    s = summary(flat_returns, nav_monotone)
    ann_ret = float(s["Ann. Return"].strip("%")) / 100
    assert ann_ret > 0
