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
    def _roll_sr(r, w=252):
        return r.rolling(w).mean() / r.rolling(w).std() * np.sqrt(w)
    roll_sr    = _roll_sr(ret)
    roll_sr_bm = _roll_sr(ret_bm)
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


def plot_comparison(results_v1: dict, results_v2: dict, results_v3: dict = None, save_path: str = None):
    """
    4-panel comparison chart: V1 vs V2 vs V3 (optional) vs equal-weight benchmark.
      Panel 1: Cumulative NAV
      Panel 2: Drawdown
      Panel 3: Rolling 12-month Sharpe
      Panel 4: Summary stats table
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 20),
                             gridspec_kw={"height_ratios": [3, 2, 2, 2]})
    title = "Strategy Comparison — V1 vs V2 vs V3 (2005–2026)" if results_v3 is not None \
            else "Strategy Comparison — V1 vs V2 (2005–2026)"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    nav1    = results_v1["nav"] * 100_000
    nav2    = results_v2["nav"] * 100_000
    nav_bm  = results_v1["nav_bm"] * 100_000   # same benchmark for both
    ret1    = results_v1["daily_returns"]
    ret2    = results_v2["daily_returns"]
    ret_bm  = results_v1["daily_returns_bm"]

    C1, C2, C3, CBM = "#2196F3", "#FF5722", "#4CAF50", "#9E9E9E"

    nav3  = results_v3["nav"] * 100_000 if results_v3 is not None else None
    ret3  = results_v3["daily_returns"]  if results_v3 is not None else None

    # ── Panel 1: NAV ──────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(nav1.index,   nav1,   label="V1",                color=C1,  linewidth=1.8)
    ax.plot(nav2.index,   nav2,   label="V2",                color=C2,  linewidth=1.8)
    if nav3 is not None:
        ax.plot(nav3.index, nav3, label="V3",                color=C3,  linewidth=1.8)
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
    if nav3 is not None:
        ax.fill_between(nav3.index, drawdown_series(nav3), 0, alpha=0.4, label="V3",              color=C3)
    ax.fill_between(nav_bm.index, drawdown_series(nav_bm), 0, alpha=0.2, label="Equal-Weight BM", color=CBM)
    ax.set_ylabel("Drawdown")
    ax.set_title("Drawdown")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Rolling 12m Sharpe ────────────────────────────────────────
    ax = axes[2]
    def _roll_sr(r, w=252):
        return r.rolling(w).mean() / r.rolling(w).std() * np.sqrt(w)
    roll1  = _roll_sr(ret1)
    roll2  = _roll_sr(ret2)
    rollbm = _roll_sr(ret_bm)
    ax.plot(roll1.index,  roll1,  label="V1",              color=C1,  linewidth=1.5)
    ax.plot(roll2.index,  roll2,  label="V2",              color=C2,  linewidth=1.5)
    if ret3 is not None:
        roll3 = _roll_sr(ret3)
        ax.plot(roll3.index, roll3, label="V3",            color=C3,  linewidth=1.5)
    ax.plot(rollbm.index, rollbm, label="Equal-Weight BM", color=CBM, linewidth=1.0, linestyle="--", alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_ylabel("Sharpe (12m rolling)")
    ax.set_title("Rolling 12-Month Sharpe Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Panel 4: Summary stats table ───────────────────────────────────────
    ax = axes[3]
    ax.axis("off")

    s1  = summary(ret1,   results_v1["nav"],    "V1")
    s2  = summary(ret2,   results_v2["nav"],    "V2")
    sbm = summary(ret_bm, results_v1["nav_bm"], "EW Benchmark")

    metrics = ["Ann. Return", "Ann. Volatility", "Sharpe Ratio",
               "Max Drawdown", "Calmar Ratio", "Worst Month", "Best Month"]

    if results_v3 is not None:
        s3 = summary(ret3, results_v3["nav"], "V3")
        col_labels = ["Metric", "V1", "V2", "V3", "EW Benchmark"]
        table_data = [[m, s1[m], s2[m], s3[m], sbm[m]] for m in metrics]
        n_cols = 5
    else:
        col_labels = ["Metric", "V1", "V2", "EW Benchmark"]
        table_data = [[m, s1[m], s2[m], sbm[m]] for m in metrics]
        n_cols = 4

    tbl = ax.table(cellText=table_data, colLabels=col_labels, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 2.2)

    # Header row
    for col in range(n_cols):
        tbl[0, col].set_facecolor("#343a40")
        tbl[0, col].set_text_props(color="white", fontweight="bold")
    # Column tints
    col_colors = ["#f5f5f5", "#E3F2FD", "#FBE9E7", "#E8F5E9", "#F5F5F5"]
    for row in range(1, len(metrics) + 1):
        for col in range(n_cols):
            tbl[row, col].set_facecolor(col_colors[col])

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


def plot_sector_comparison(results_v2: dict, results_v2b: dict, results_v2c: dict = None, save_path: str = None):
    """
    4-panel comparison: Sector V2 (monthly) vs V2b (weekly) vs V2c (cross-asset + corr filter).
    Style mirrors strategy_comparison_2011_2026.png: NAV, Drawdown, Rolling Sharpe, Table.
    """
    import matplotlib.ticker as mticker

    fig, axes = plt.subplots(4, 1, figsize=(14, 22),
                             gridspec_kw={"height_ratios": [3, 2, 2, 3]})
    fig.suptitle(
        "Sector Rotation: V2 (Monthly) vs V2b (Weekly) vs V2c (Cross-Asset + Corr Filter)\n"
        "2011–2026  |  0% cash on trailing stops",
        fontsize=14, fontweight="bold"
    )

    START = "2011-01-01"
    nav2  = (results_v2["nav"]    * 100_000).loc[START:]
    nav2b = (results_v2b["nav"]   * 100_000).loc[START:]
    navbm = (results_v2["nav_bm"] * 100_000).loc[START:]

    series_list = [nav2, nav2b, navbm]
    if results_v2c is not None:
        nav2c = (results_v2c["nav"] * 100_000).loc[START:]
        series_list.append(nav2c)

    t0 = max(s.index[0] for s in series_list)
    nav2  = nav2.loc[t0:]  / nav2.loc[t0]  * 100_000
    nav2b = nav2b.loc[t0:] / nav2b.loc[t0] * 100_000
    navbm = navbm.loc[t0:] / navbm.loc[t0] * 100_000
    if results_v2c is not None:
        nav2c = nav2c.loc[t0:] / nav2c.loc[t0] * 100_000

    ret2  = results_v2["daily_returns"].loc[t0:]
    ret2b = results_v2b["daily_returns"].loc[t0:]
    retbm = results_v2["daily_returns_bm"].loc[t0:]
    if results_v2c is not None:
        ret2c = results_v2c["daily_returns"].loc[t0:]

    C2, C2B, C2C, CBM = "#2196F3", "#FF5722", "#4CAF50", "#9E9E9E"

    # ── Panel 1: Log-scale NAV ────────────────────────────────────────────
    ax = axes[0]
    ax.semilogy(nav2.index,  nav2,  label="V2  (monthly)",         color=C2,  linewidth=1.8)
    ax.semilogy(nav2b.index, nav2b, label="V2b (weekly)",          color=C2B, linewidth=1.8)
    if results_v2c is not None:
        ax.semilogy(nav2c.index, nav2c, label="V2c (cross-asset)", color=C2C, linewidth=1.8)
    ax.semilogy(navbm.index, navbm, label="Equal-weight BM",       color=CBM, linewidth=1.0, linestyle="--", alpha=0.7)

    # End labels — sort by final value, assign vertical offsets to avoid overlap
    end_labels = [(nav2, C2, "V2"), (nav2b, C2B, "V2b")]
    if results_v2c is not None:
        end_labels.append((nav2c, C2C, "V2c"))
    end_labels.sort(key=lambda x: x[0].iloc[-1])
    offsets = [0.55, 1.0, 1.6] if len(end_labels) == 3 else [0.75, 1.35]
    for (series, color, text), mult in zip(end_labels, offsets):
        y = series.iloc[-1]
        ax.annotate(f"  {text}\n  ${y:,.0f}",
                    xy=(series.index[-1], y * mult), color=color,
                    fontsize=9, fontweight="bold", va="center")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_ylabel("Portfolio Value  (log scale, start = $100,000)")
    ax.set_title("Cumulative NAV")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, which="both")

    # ── Panel 2: Drawdown ────────────────────────────────────────────────
    ax = axes[1]
    dd2  = drawdown_series(nav2)
    dd2b = drawdown_series(nav2b)
    ddbm = drawdown_series(navbm)
    ax.fill_between(nav2.index,  dd2,  0, alpha=0.45, label=f"V2   MaxDD {dd2.min():.1%}",  color=C2)
    ax.fill_between(nav2b.index, dd2b, 0, alpha=0.45, label=f"V2b  MaxDD {dd2b.min():.1%}", color=C2B)
    if results_v2c is not None:
        dd2c = drawdown_series(nav2c)
        ax.fill_between(nav2c.index, dd2c, 0, alpha=0.45, label=f"V2c  MaxDD {dd2c.min():.1%}", color=C2C)
    ax.fill_between(navbm.index, ddbm, 0, alpha=0.20, label="Benchmark",                    color=CBM)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel("Drawdown")
    ax.set_title("Drawdown from Peak")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # ── Panel 3: Rolling 12-month Sharpe ─────────────────────────────────
    ax = axes[2]
    def _rolling_sharpe(r, w=252):
        mu  = r.rolling(w).mean()
        sig = r.rolling(w).std()
        return (mu / sig * np.sqrt(w)).rename(r.name)

    roll2  = _rolling_sharpe(ret2)
    roll2b = _rolling_sharpe(ret2b)
    rollbm = _rolling_sharpe(retbm)
    ax.plot(roll2.index,  roll2,  label="V2  monthly", color=C2,  linewidth=1.5)
    ax.plot(roll2b.index, roll2b, label="V2b weekly",  color=C2B, linewidth=1.5)
    if results_v2c is not None:
        roll2c = _rolling_sharpe(ret2c)
        ax.plot(roll2c.index, roll2c, label="V2c cross-asset", color=C2C, linewidth=1.5)
    ax.plot(rollbm.index, rollbm, label="Benchmark",   color=CBM, linewidth=0.9, linestyle="--", alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.7, linestyle=":")
    ax.set_ylabel("Sharpe (12m rolling)")
    ax.set_title("Rolling 12-Month Sharpe Ratio")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # ── Panel 4: Summary stats table ─────────────────────────────────────
    ax = axes[3]
    ax.axis("off")

    s2  = summary(ret2,  nav2  / 100_000, "V2 (monthly)")
    s2b = summary(ret2b, nav2b / 100_000, "V2b (weekly)")
    sbm = summary(retbm, navbm / 100_000, "EW Benchmark")

    metrics = ["Ann. Return", "Sharpe Ratio", "Max Drawdown",
               "Calmar Ratio", "Worst Month", "Total Return"]

    col_labels = ["Strategy", "Ann. Return", "Sharpe Ratio",
                  "Max Drawdown", "Calmar Ratio", "Worst Month", "Total Return"]

    table_rows = [("V2 (monthly)", s2), ("V2b (weekly)", s2b)]
    if results_v2c is not None:
        s2c = summary(ret2c, nav2c / 100_000, "V2c (cross-asset)")
        table_rows.append(("V2c (cross-asset)", s2c))
    table_rows.append(("EW Benchmark", sbm))

    rows = []
    for lbl, s in table_rows:
        rows.append([lbl] + [s[m] for m in metrics])

    tbl = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 2.2)

    all_row_colors = ["#E3F2FD", "#FBE9E7", "#E8F5E9", "#F5F5F5"]
    for row_i in range(len(rows)):
        for col in range(len(col_labels)):
            tbl[row_i + 1, col].set_facecolor(all_row_colors[row_i])
    for col in range(len(col_labels)):
        tbl[0, col].set_facecolor("#343a40")
        tbl[0, col].set_text_props(color="white", fontweight="bold")

    ax.set_title("Summary Statistics  (2011–2026, 0% cash on stops)",
                 fontsize=11, fontweight="bold", pad=12)

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


def plot_v2c_extended(results_v2c: dict, results_v2d: dict = None, save_path: str = None):
    """
    V2c (+ optional V2d overlay) extended history chart: 2002–present.
    4 panels: log NAV, drawdown, rolling Sharpe, stats table.
    Benchmark: equal-weight SECTOR_CORE.
    """
    import matplotlib.ticker as mticker

    # SHY/TLT/IEF launched July 2002; start chart from Aug 2002 so cash allocation works cleanly.
    START = "2002-08-01"

    nav_v2c = (results_v2c["nav"]    * 100_000)
    nav_bm  = (results_v2c["nav_bm"] * 100_000)

    # Align to first available date >= START
    candidates = [
        nav_v2c.loc[nav_v2c.index >= START].index[0],
        nav_bm.loc[nav_bm.index   >= START].index[0],
    ]
    if results_v2d is not None:
        nav_v2d_raw = results_v2d["nav"] * 100_000
        candidates.append(nav_v2d_raw.loc[nav_v2d_raw.index >= START].index[0])
    t0 = max(candidates)

    nav_v2c = nav_v2c.loc[t0:] / nav_v2c.loc[t0] * 100_000
    nav_bm  = nav_bm.loc[t0:]  / nav_bm.loc[t0]  * 100_000

    ret_v2c = results_v2c["daily_returns"].loc[t0:]
    ret_bm  = results_v2c["daily_returns_bm"].loc[t0:]

    nav_v2d = ret_v2d = None
    if results_v2d is not None:
        nav_v2d = nav_v2d_raw.loc[t0:] / nav_v2d_raw.loc[t0] * 100_000
        ret_v2d = results_v2d["daily_returns"].loc[t0:]

    end_yr  = nav_v2c.index[-1].year
    C2C, C2D, CBM = "#4CAF50", "#2196F3", "#9E9E9E"

    title_suffix = " vs V2d (liquid universe)" if results_v2d is not None else ""
    fig, axes = plt.subplots(4, 1, figsize=(14, 22),
                             gridspec_kw={"height_ratios": [3, 2, 2, 3]})
    fig.suptitle(
        f"Sector V2c{title_suffix} — Extended History  {t0.year}–{end_yr}\n"
        "Cross-Asset Universe · Cluster Caps · Adaptive Trailing Stops  |  Best Params",
        fontsize=14, fontweight="bold",
    )

    # ── Panel 1: Log-scale NAV ────────────────────────────────────────────────
    ax = axes[0]
    ax.semilogy(nav_v2c.index, nav_v2c, label="V2c (46 ETFs, incl illiquid)", color=C2C, linewidth=1.8)
    if nav_v2d is not None:
        ax.semilogy(nav_v2d.index, nav_v2d, label="V2d (38 ETFs, ≥$100M/day)", color=C2D, linewidth=1.8, linestyle="-.")
    ax.semilogy(nav_bm.index,  nav_bm,  label="EW Sector Core BM", color=CBM,
                linewidth=1.0, linestyle="--", alpha=0.7)

    # Universe expansion milestones
    milestones = [
        ("2004-11-18", "GLD\n+VNQ"),
        ("2007-04-04", "HYG\n+Miners"),
        ("2010-04-19", "~30\nETFs"),
        ("2019-12-03", "Full\nuniverse"),
    ]
    for dt_str, lbl in milestones:
        dt = pd.Timestamp(dt_str)
        if dt > nav_v2c.index[0]:
            ax.axvline(dt, color="#BDBDBD", linewidth=0.8, linestyle=":", alpha=0.6)
            y_pos = nav_v2c.loc[nav_v2c.index >= dt].iloc[0]
            ax.text(dt, y_pos * 1.6, lbl, fontsize=7, color="#757575",
                    ha="center", va="bottom", rotation=0)

    end_v2c = nav_v2c.iloc[-1]
    end_bm_ = nav_bm.iloc[-1]
    ax.annotate(f"  V2c\n  ${end_v2c:,.0f}",
                xy=(nav_v2c.index[-1], end_v2c * 1.5), color=C2C,
                fontsize=9, fontweight="bold", va="center")
    if nav_v2d is not None:
        end_v2d = nav_v2d.iloc[-1]
        ax.annotate(f"  V2d\n  ${end_v2d:,.0f}",
                    xy=(nav_v2d.index[-1], end_v2d * 0.85), color=C2D,
                    fontsize=9, fontweight="bold", va="center")
    ax.annotate(f"  BM\n  ${end_bm_:,.0f}",
                xy=(nav_bm.index[-1], end_bm_ * 0.6), color=CBM,
                fontsize=9, fontweight="bold", va="center")

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_ylabel("Portfolio Value  (log scale, start = $100,000)")
    ax.set_title("Cumulative NAV")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, which="both")

    # ── Panel 2: Drawdown ─────────────────────────────────────────────────────
    ax = axes[1]
    dd_v2c = drawdown_series(nav_v2c)
    dd_bm  = drawdown_series(nav_bm)
    ax.fill_between(nav_v2c.index, dd_v2c, 0, alpha=0.50,
                    label=f"V2c  MaxDD {dd_v2c.min():.1%}", color=C2C)
    if nav_v2d is not None:
        dd_v2d = drawdown_series(nav_v2d)
        ax.fill_between(nav_v2d.index, dd_v2d, 0, alpha=0.35,
                        label=f"V2d  MaxDD {dd_v2d.min():.1%}", color=C2D)
    ax.fill_between(nav_bm.index,  dd_bm,  0, alpha=0.20,
                    label=f"BM   MaxDD {dd_bm.min():.1%}",  color=CBM)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel("Drawdown")
    ax.set_title("Drawdown from Peak")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # ── Panel 3: Rolling 12-month Sharpe ─────────────────────────────────────
    ax = axes[2]

    def _rolling_sharpe(r, w=252):
        mu  = r.rolling(w).mean()
        sig = r.rolling(w).std()
        return (mu / sig * np.sqrt(w)).rename(r.name)

    roll_v2c = _rolling_sharpe(ret_v2c)
    roll_bm  = _rolling_sharpe(ret_bm)
    ax.plot(roll_v2c.index, roll_v2c, label="V2c", color=C2C, linewidth=1.5)
    if ret_v2d is not None:
        roll_v2d = _rolling_sharpe(ret_v2d)
        ax.plot(roll_v2d.index, roll_v2d, label="V2d", color=C2D, linewidth=1.5, linestyle="-.")
    ax.plot(roll_bm.index,  roll_bm,  label="BM",  color=CBM, linewidth=0.9,
            linestyle="--", alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.7, linestyle=":")
    ax.set_ylabel("Sharpe (12m rolling)")
    ax.set_title("Rolling 12-Month Sharpe Ratio")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # ── Panel 4: Summary stats table ─────────────────────────────────────────
    ax = axes[3]
    ax.axis("off")

    s_v2c = summary(ret_v2c, nav_v2c / 100_000, "V2c (cross-asset)")
    s_bm  = summary(ret_bm,  nav_bm  / 100_000, "EW Sector BM")

    metrics    = ["Ann. Return", "Sharpe Ratio", "Max Drawdown",
                  "Calmar Ratio", "Worst Month", "Total Return"]
    col_labels = ["Strategy", "Ann. Return", "Sharpe Ratio",
                  "Max Drawdown", "Calmar Ratio", "Worst Month", "Total Return"]

    rows = [["V2c (46 ETFs)"] + [s_v2c[m] for m in metrics]]
    row_colors = ["#E8F5E9"]
    if nav_v2d is not None:
        s_v2d = summary(ret_v2d, nav_v2d / 100_000, "V2d (liquid)")
        rows.append(["V2d (38 ETFs ≥$100M/day)"] + [s_v2d[m] for m in metrics])
        row_colors.append("#E3F2FD")
    rows.append(["EW Sector BM"] + [s_bm[m] for m in metrics])
    row_colors.append("#F5F5F5")

    tbl = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 2.2)

    for row_i, color in enumerate(row_colors):
        for col in range(len(col_labels)):
            tbl[row_i + 1, col].set_facecolor(color)
    for col in range(len(col_labels)):
        tbl[0, col].set_facecolor("#343a40")
        tbl[0, col].set_text_props(color="white", fontweight="bold")

    n_etfs_start = 13  # XL9 + EWJ + EWZ + EFA + IBB available in mid-2002
    ax.set_title(
        f"Summary Statistics  ({t0.year}–{end_yr})  ·  "
        f"V2c: 13 ETFs in 2002 → 46 by 2019  |  V2d: liquid-only subset (≥$100M/day)",
        fontsize=10, fontweight="bold", pad=12,
    )

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


def plot_v2d_v2e(results_v2d: dict, results_v2e: dict, save_path: str = None):
    """
    V2d vs V2e side-by-side comparison, 2002–present.
    4 panels: log NAV, drawdown, rolling Sharpe, stats table.
    """
    import matplotlib.ticker as mticker

    START = "2002-08-01"

    nav_v2d_raw = results_v2d["nav"] * 100_000
    nav_v2e_raw = results_v2e["nav"] * 100_000
    nav_bm_raw  = results_v2d["nav_bm"] * 100_000

    t0 = max(
        nav_v2d_raw.loc[nav_v2d_raw.index >= START].index[0],
        nav_v2e_raw.loc[nav_v2e_raw.index >= START].index[0],
        nav_bm_raw.loc[nav_bm_raw.index   >= START].index[0],
    )

    nav_v2d = nav_v2d_raw.loc[t0:] / nav_v2d_raw.loc[t0] * 100_000
    nav_v2e = nav_v2e_raw.loc[t0:] / nav_v2e_raw.loc[t0] * 100_000
    nav_bm  = nav_bm_raw.loc[t0:]  / nav_bm_raw.loc[t0]  * 100_000

    ret_v2d = results_v2d["daily_returns"].loc[t0:]
    ret_v2e = results_v2e["daily_returns"].loc[t0:]
    ret_bm  = results_v2d["daily_returns_bm"].loc[t0:]

    end_yr  = nav_v2d.index[-1].year
    C2D, C2E, CBM = "#2196F3", "#FF9800", "#9E9E9E"

    fig, axes = plt.subplots(4, 1, figsize=(14, 22),
                             gridspec_kw={"height_ratios": [3, 2, 2, 3]})
    fig.suptitle(
        f"Sector V2d vs V2e — Supercycle Momentum Comparison  {t0.year}–{end_yr}\n"
        "Liquid Universe (38 ETFs ≥$100M/day)  |  V2e adds 24m/36m lookbacks  |  Best Params",
        fontsize=14, fontweight="bold",
    )

    # ── Panel 1: Log NAV ──────────────────────────────────────────────────────
    ax = axes[0]
    ax.semilogy(nav_v2d.index, nav_v2d, label="V2d (12m/18m momentum)", color=C2D, linewidth=1.8)
    ax.semilogy(nav_v2e.index, nav_v2e, label="V2e (+ 24m/36m supercycle)", color=C2E, linewidth=1.8, linestyle="-.")
    ax.semilogy(nav_bm.index,  nav_bm,  label="EW Sector Core BM", color=CBM,
                linewidth=1.0, linestyle="--", alpha=0.7)

    for dt_str, lbl in [("2004-11-18", "GLD\n+VNQ"), ("2007-04-04", "HYG\n+Miners"),
                         ("2010-04-19", "~30\nETFs"), ("2019-12-03", "Full\nuniverse")]:
        dt = pd.Timestamp(dt_str)
        if dt > nav_v2d.index[0]:
            ax.axvline(dt, color="#BDBDBD", linewidth=0.8, linestyle=":", alpha=0.6)
            y_pos = nav_v2d.loc[nav_v2d.index >= dt].iloc[0]
            ax.text(dt, y_pos * 1.6, lbl, fontsize=7, color="#757575",
                    ha="center", va="bottom")

    end_2d = nav_v2d.iloc[-1]
    end_2e = nav_v2e.iloc[-1]
    end_bm = nav_bm.iloc[-1]
    ax.annotate(f"  V2d  ${end_2d:,.0f}", xy=(nav_v2d.index[-1], end_2d * 0.70),
                color=C2D, fontsize=9, fontweight="bold", va="center")
    ax.annotate(f"  V2e  ${end_2e:,.0f}", xy=(nav_v2e.index[-1], end_2e * 1.40),
                color=C2E, fontsize=9, fontweight="bold", va="center")
    ax.annotate(f"  BM   ${end_bm:,.0f}", xy=(nav_bm.index[-1], end_bm * 0.55),
                color=CBM, fontsize=9, fontweight="bold", va="center")

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_ylabel("Portfolio Value  (log scale, start = $100,000)")
    ax.set_title("Cumulative NAV")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, which="both")

    # ── Panel 2: Drawdown ──────────────────────────────────────────────────────
    ax = axes[1]
    dd_v2d = drawdown_series(nav_v2d)
    dd_v2e = drawdown_series(nav_v2e)
    dd_bm  = drawdown_series(nav_bm)
    ax.fill_between(nav_v2d.index, dd_v2d, 0, alpha=0.55,
                    label=f"V2d  MaxDD {dd_v2d.min():.1%}", color=C2D)
    ax.fill_between(nav_v2e.index, dd_v2e, 0, alpha=0.40,
                    label=f"V2e  MaxDD {dd_v2e.min():.1%}", color=C2E)
    ax.fill_between(nav_bm.index,  dd_bm,  0, alpha=0.20,
                    label=f"BM   MaxDD {dd_bm.min():.1%}",  color=CBM)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel("Drawdown")
    ax.set_title("Drawdown from Peak")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # ── Panel 3: Rolling 12m Sharpe ────────────────────────────────────────────
    ax = axes[2]

    def _rs(r, w=252):
        return (r.rolling(w).mean() / r.rolling(w).std() * np.sqrt(w))

    ax.plot(nav_v2d.index, _rs(ret_v2d), label="V2d", color=C2D, linewidth=1.5)
    ax.plot(nav_v2e.index, _rs(ret_v2e), label="V2e", color=C2E, linewidth=1.5, linestyle="-.")
    ax.plot(nav_bm.index,  _rs(ret_bm),  label="BM",  color=CBM, linewidth=0.9,
            linestyle="--", alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.7, linestyle=":")
    ax.set_ylabel("Sharpe (12m rolling)")
    ax.set_title("Rolling 12-Month Sharpe Ratio")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # ── Panel 4: Stats table ───────────────────────────────────────────────────
    ax = axes[3]
    ax.axis("off")

    s2d = summary(ret_v2d, nav_v2d / 100_000, "V2d")
    s2e = summary(ret_v2e, nav_v2e / 100_000, "V2e")
    s_bm = summary(ret_bm,  nav_bm  / 100_000, "BM")

    metrics    = ["Ann. Return", "Sharpe Ratio", "Max Drawdown",
                  "Calmar Ratio", "Worst Month", "Total Return"]
    col_labels = ["Strategy", "Ann. Return", "Sharpe Ratio",
                  "Max Drawdown", "Calmar Ratio", "Worst Month", "Total Return"]

    rows = [
        ["V2d (12m/18m momentum)"]      + [s2d[m] for m in metrics],
        ["V2e (+ 24m/36m supercycle)"]  + [s2e[m] for m in metrics],
        ["EW Sector Core BM"]           + [s_bm[m] for m in metrics],
    ]
    row_colors = ["#E3F2FD", "#FFF3E0", "#F5F5F5"]

    tbl = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 2.2)

    for row_i, color in enumerate(row_colors):
        for col in range(len(col_labels)):
            tbl[row_i + 1, col].set_facecolor(color)
    for col in range(len(col_labels)):
        tbl[0, col].set_facecolor("#343a40")
        tbl[0, col].set_text_props(color="white", fontweight="bold")

    ax.set_title(
        f"Summary Statistics  ({t0.year}–{end_yr})  ·  "
        "V2e adds 24m/36m supercycle lookbacks to the V2d liquid universe",
        fontsize=10, fontweight="bold", pad=12,
    )

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
