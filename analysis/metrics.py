"""Pure performance metric functions — no matplotlib dependency."""
import pandas as pd
import numpy as np


def sharpe(returns: pd.Series, rf: float = 0.0, periods: int = 252) -> float:
    excess = returns - rf / periods
    return float(excess.mean() / excess.std() * np.sqrt(periods))


def max_drawdown(nav: pd.Series) -> float:
    peak = nav.cummax()
    dd   = (nav - peak) / peak
    return float(dd.min())


def calmar(nav: pd.Series, returns: pd.Series, periods: int = 252) -> float:
    ann_ret = (nav.iloc[-1] ** (periods / len(returns))) - 1
    mdd = abs(max_drawdown(nav))
    if mdd > 0:
        return ann_ret / mdd
    # Zero drawdown: calmar = return / 0 is mathematically infinite. Preserve the
    # sign so a profitable, never-drawing strategy reads as +inf (not nan, which
    # would discard the "this is good" signal). Flat strategy → nan.
    if ann_ret > 0:
        return float("inf")
    if ann_ret < 0:
        return float("-inf")
    return np.nan


def drawdown_series(nav: pd.Series) -> pd.Series:
    peak = nav.cummax()
    return (nav - peak) / peak


def summary(returns: pd.Series, nav: pd.Series, label: str = "Strategy") -> pd.Series:
    n = len(returns)
    ann_ret  = (nav.iloc[-1] ** (252 / n)) - 1
    ann_vol  = returns.std() * np.sqrt(252)
    sr       = sharpe(returns)
    mdd      = max_drawdown(nav)
    cal      = calmar(nav, returns)
    win_rate = (returns > 0).mean()

    monthly_ret = (1 + returns).resample("ME").prod() - 1
    best_month  = monthly_ret.max()
    worst_month = monthly_ret.min()

    return pd.Series({
        "Ann. Return":      f"{ann_ret:.1%}",
        "Ann. Volatility":  f"{ann_vol:.1%}",
        "Sharpe Ratio":     f"{sr:.2f}",
        "Max Drawdown":     f"{mdd:.1%}",
        "Calmar Ratio":     f"{cal:.2f}",
        "Win Rate (daily)": f"{win_rate:.1%}",
        "Best Month":       f"{best_month:.1%}",
        "Worst Month":      f"{worst_month:.1%}",
        "Total Return":     f"{nav.iloc[-1] - 1:.1%}",
        "Start":            str(nav.index[0].date()),
        "End":              str(nav.index[-1].date()),
    }, name=label)
