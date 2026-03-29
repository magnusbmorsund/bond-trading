"""
Performance analytics and plotting.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

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
    return ann_ret / mdd if mdd > 0 else np.nan


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
        "Ann. Return":    f"{ann_ret:.1%}",
        "Ann. Volatility":f"{ann_vol:.1%}",
        "Sharpe Ratio":   f"{sr:.2f}",
        "Max Drawdown":   f"{mdd:.1%}",
        "Calmar Ratio":   f"{cal:.2f}",
        "Win Rate (daily)":f"{win_rate:.1%}",
        "Best Month":     f"{best_month:.1%}",
        "Worst Month":    f"{worst_month:.1%}",
        "Total Return":   f"{nav.iloc[-1] - 1:.1%}",
        "Start":          str(nav.index[0].date()),
        "End":            str(nav.index[-1].date()),
    }, name=label)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _annual_stats(returns: pd.Series, nav: pd.Series) -> pd.DataFrame:
    """Year-by-year Ann. Return, Max DD, Volatility."""
    rows = []
    for year, grp in returns.groupby(returns.index.year):
        nav_yr  = nav.loc[grp.index]
        ann_ret = (1 + grp).prod() - 1
        ann_vol = grp.std() * np.sqrt(252)
        mdd     = max_drawdown(nav_yr)
        rows.append({"Year": year, "Return": ann_ret, "Max DD": mdd, "Volatility": ann_vol})
    return pd.DataFrame(rows).set_index("Year")


def plot_results(results: dict, save_path: str = None):
    fig, axes = plt.subplots(4, 1, figsize=(14, 18), gridspec_kw={"height_ratios": [3, 2, 2, 2]})
    fig.suptitle("Bond Rotation Strategy — Backtest Results", fontsize=14, fontweight="bold")

    nav    = results["nav"] * 100_000
    nav_bm = results["nav_bm"] * 100_000
    ret    = results["daily_returns"]
    ret_bm = results["daily_returns_bm"]
    w      = results["weights"]

    # ── Panel 1: NAV ──────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(nav.index,    nav,    label="Strategy",       linewidth=1.5)
    ax.plot(nav_bm.index, nav_bm, label="Equal-Weight BM",linewidth=1.2, alpha=0.7, linestyle="--")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_ylabel("Portfolio Value (start = $100,000)")
    ax.set_title("Cumulative NAV")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Drawdown ─────────────────────────────────────────────────
    ax = axes[1]
    ax.fill_between(nav.index,    drawdown_series(nav),    0, alpha=0.5, label="Strategy",        color="steelblue")
    ax.fill_between(nav_bm.index, drawdown_series(nav_bm), 0, alpha=0.3, label="Equal-Weight BM", color="orange")
    ax.set_ylabel("Drawdown")
    ax.set_title("Drawdown")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Portfolio weights ─────────────────────────────────────────
    ax = axes[2]
    # Generate one color per ETF grouped by bucket: duration=blues, inflation=green,
    # credit=oranges/reds, commodities=golds. Falls back to a colormap for any size.
    import matplotlib.cm as cm
    n = len(w.columns)
    colors = [cm.tab20(i / max(n - 1, 1)) for i in range(n)]
    w.plot.area(ax=ax, stacked=True, color=colors, alpha=0.85, linewidth=0)
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1)
    ax.set_title("Monthly Portfolio Weights")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)

    # ── Panel 4: Rolling 12m Sharpe ────────────────────────────────────────
    ax = axes[3]
    roll_sr    = ret.rolling(252).apply(lambda x: sharpe(x))
    roll_sr_bm = ret_bm.rolling(252).apply(lambda x: sharpe(x))
    ax.plot(roll_sr.index,    roll_sr,    label="Strategy",       linewidth=1.2)
    ax.plot(roll_sr_bm.index, roll_sr_bm, label="Equal-Weight BM",linewidth=1.0, alpha=0.7, linestyle="--")
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_ylabel("Sharpe (12m rolling)")
    ax.set_title("Rolling 12-Month Sharpe Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved chart → {save_path}")
    else:
        plt.show()

    plt.close()


def plot_annual_stats(results: dict, save_path: str = None):
    """Standalone PNG: year-by-year Return / Max DD / Volatility table."""
    ret    = results["daily_returns"]
    ret_bm = results["daily_returns_bm"]

    strat = _annual_stats(ret, results["nav"])
    bench = _annual_stats(ret_bm, results["nav_bm"])
    years = sorted(set(strat.index) | set(bench.index))

    col_labels = ["Year", "Return", "Max DD", "Vol", "BM Return", "BM Max DD", "BM Vol"]
    table_data = []
    for yr in years:
        s = strat.loc[yr] if yr in strat.index else pd.Series({"Return": np.nan, "Max DD": np.nan, "Volatility": np.nan})
        b = bench.loc[yr] if yr in bench.index else pd.Series({"Return": np.nan, "Max DD": np.nan, "Volatility": np.nan})
        table_data.append([
            str(yr),
            f"{s['Return']:+.1%}", f"{s['Max DD']:.1%}", f"{s['Volatility']:.1%}",
            f"{b['Return']:+.1%}", f"{b['Max DD']:.1%}", f"{b['Volatility']:.1%}",
        ])

    n_rows = len(table_data)
    fig_h  = max(4, 0.35 * n_rows + 1.5)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    fig.suptitle("Annual Statistics — Strategy vs Equal-Weight Benchmark", fontsize=13, fontweight="bold")
    ax.axis("off")

    tbl = ax.table(cellText=table_data, colLabels=col_labels, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.5)

    for i, yr in enumerate(years):
        ret_val = strat.loc[yr, "Return"] if yr in strat.index else 0
        bm_val  = bench.loc[yr, "Return"] if yr in bench.index else 0
        row_idx = i + 1
        for col in range(4):
            tbl[row_idx, col].set_facecolor("#d4edda" if ret_val >= 0 else "#f8d7da")
        for col in range(4, 7):
            tbl[row_idx, col].set_facecolor("#d4edda" if bm_val >= 0 else "#f8d7da")

    for col in range(len(col_labels)):
        tbl[0, col].set_facecolor("#343a40")
        tbl[0, col].set_text_props(color="white", fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved chart → {save_path}")
    else:
        plt.show()
    plt.close()


def plot_annual_allocations(results: dict, save_path: str = None, top_n: int = 5):
    """Standalone PNG: top-N average allocations per ETF, year by year."""
    w = results["weights"].copy()
    w.index = pd.to_datetime(w.index)
    years = sorted(w.index.year.unique())

    # Build rows: each row is a year, columns are top_n ETFs + "Other"
    rows = {}
    for yr in years:
        avg = w[w.index.year == yr].mean().sort_values(ascending=False)
        top = avg.head(top_n)
        other = avg.iloc[top_n:].sum()
        rows[yr] = {**{etf: v for etf, v in top.items()}, "Other": other}

    df = pd.DataFrame(rows).T.fillna(0)

    # --- figure ---
    n_rows  = len(years)
    fig_h   = max(4, 0.38 * n_rows + 1.5)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    fig.suptitle(f"Annual Average Allocation — Top {top_n} Positions per Year", fontsize=13, fontweight="bold")
    ax.axis("off")

    col_labels = ["Year"] + [f"#{i+1}" for i in range(top_n)] + ["Other"]
    table_data = []
    for yr in years:
        row_avgs = w[w.index.year == yr].mean().sort_values(ascending=False)
        top      = row_avgs.head(top_n)
        other    = row_avgs.iloc[top_n:].sum()
        cells    = [str(yr)]
        for etf, val in top.items():
            cells.append(f"{etf}  {val:.1%}")
        # pad if fewer than top_n ETFs
        while len(cells) < top_n + 1:
            cells.append("—")
        cells.append(f"{other:.1%}")
        table_data.append(cells)

    tbl = ax.table(cellText=table_data, colLabels=col_labels, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)

    # Alternate row shading
    for i in range(len(table_data)):
        shade = "#f2f2f2" if i % 2 == 0 else "#ffffff"
        for col in range(len(col_labels)):
            tbl[i + 1, col].set_facecolor(shade)

    for col in range(len(col_labels)):
        tbl[0, col].set_facecolor("#343a40")
        tbl[0, col].set_text_props(color="white", fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved chart → {save_path}")
    else:
        plt.show()
    plt.close()


def plot_comparison(results_v1: dict, results_v2: dict, save_path: str = None):
    """
    4-panel comparison chart: V1 vs V2 vs equal-weight benchmark.
      Panel 1: Cumulative NAV
      Panel 2: Drawdown
      Panel 3: Rolling 12-month Sharpe
      Panel 4: Summary stats table
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 20),
                             gridspec_kw={"height_ratios": [3, 2, 2, 2]})
    fig.suptitle("Strategy Comparison — V1 vs V2 (2005–2026)", fontsize=14, fontweight="bold")

    nav1    = results_v1["nav"] * 100_000
    nav2    = results_v2["nav"] * 100_000
    nav_bm  = results_v1["nav_bm"] * 100_000   # same benchmark for both
    ret1    = results_v1["daily_returns"]
    ret2    = results_v2["daily_returns"]
    ret_bm  = results_v1["daily_returns_bm"]

    C1, C2, CBM = "#2196F3", "#FF5722", "#9E9E9E"

    # ── Panel 1: NAV ──────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(nav1.index,   nav1,   label="V1",                color=C1,  linewidth=1.8)
    ax.plot(nav2.index,   nav2,   label="V2",                color=C2,  linewidth=1.8)
    ax.plot(nav_bm.index, nav_bm, label="Equal-Weight BM",   color=CBM, linewidth=1.0, linestyle="--", alpha=0.7)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_ylabel("Portfolio Value (start = $100,000)")
    ax.set_title("Cumulative NAV")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Drawdown ─────────────────────────────────────────────────
    ax = axes[1]
    ax.fill_between(nav1.index,   drawdown_series(nav1),   0, alpha=0.4, label="V1",              color=C1)
    ax.fill_between(nav2.index,   drawdown_series(nav2),   0, alpha=0.4, label="V2",              color=C2)
    ax.fill_between(nav_bm.index, drawdown_series(nav_bm), 0, alpha=0.2, label="Equal-Weight BM", color=CBM)
    ax.set_ylabel("Drawdown")
    ax.set_title("Drawdown")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Rolling 12m Sharpe ────────────────────────────────────────
    ax = axes[2]
    roll1  = ret1.rolling(252).apply(lambda x: sharpe(x))
    roll2  = ret2.rolling(252).apply(lambda x: sharpe(x))
    rollbm = ret_bm.rolling(252).apply(lambda x: sharpe(x))
    ax.plot(roll1.index,  roll1,  label="V1",              color=C1,  linewidth=1.5)
    ax.plot(roll2.index,  roll2,  label="V2",              color=C2,  linewidth=1.5)
    ax.plot(rollbm.index, rollbm, label="Equal-Weight BM", color=CBM, linewidth=1.0, linestyle="--", alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_ylabel("Sharpe (12m rolling)")
    ax.set_title("Rolling 12-Month Sharpe Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Panel 4: Summary stats table ───────────────────────────────────────
    ax = axes[3]
    ax.axis("off")

    s1 = summary(ret1,   results_v1["nav"], "V1")
    s2 = summary(ret2,   results_v2["nav"], "V2")
    sbm = summary(ret_bm, results_v1["nav_bm"], "EW Benchmark")

    metrics = ["Ann. Return", "Ann. Volatility", "Sharpe Ratio",
               "Max Drawdown", "Calmar Ratio", "Worst Month", "Best Month"]
    col_labels = ["Metric", "V1", "V2", "EW Benchmark"]
    table_data = [[m, s1[m], s2[m], sbm[m]] for m in metrics]

    tbl = ax.table(cellText=table_data, colLabels=col_labels, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 2.2)

    # Header row
    for col in range(4):
        tbl[0, col].set_facecolor("#343a40")
        tbl[0, col].set_text_props(color="white", fontweight="bold")
    # V1 column blue tint, V2 column orange tint
    for row in range(1, len(metrics) + 1):
        tbl[row, 0].set_facecolor("#f5f5f5")
        tbl[row, 1].set_facecolor("#E3F2FD")
        tbl[row, 2].set_facecolor("#FBE9E7")
        tbl[row, 3].set_facecolor("#F5F5F5")

    ax.set_title("Summary Statistics", fontsize=11, fontweight="bold", pad=12)

    for ax in axes[:3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved chart → {save_path}")
    else:
        plt.show()
    plt.close()


def print_summary_table(results: dict):
    ret    = results["daily_returns"]
    nav    = results["nav"]
    ret_bm = results["daily_returns_bm"]
    nav_bm = results["nav_bm"]

    s1 = summary(ret,    nav,    "Strategy")
    s2 = summary(ret_bm, nav_bm, "EW Benchmark")

    tbl = pd.concat([s1, s2], axis=1)
    print("\n" + "=" * 52)
    print("BACKTEST SUMMARY")
    print("=" * 52)
    print(tbl.to_string())
    print("=" * 52)
