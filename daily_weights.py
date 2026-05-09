"""
Daily weights report — run by GitHub Actions.
Computes effective weights for V1/V2/V3, compares with previous day,
and writes positions + buy/sell orders to positions/YYYY-MM-DD.csv.
"""
import os
import sys
import warnings
import logging
from datetime import date

import pandas as pd
from strategies.backtest_core import DEFAULT_COST_MODEL

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

# Minimum absolute delta (%) to trigger a trade.
# Below this the round-trip cost exceeds the expected rebalance benefit.
# At Saxo rates: ~14 bps round-trip at low turnover → 0.5% threshold keeps
# cost drag well below the noise of daily returns.
_MIN_TRADE_PCT = 0.5


def _compute_weights(version: str):
    """Return (effective_weights Series, as_of date) for a strategy version."""
    if version == "v3":
        import configs.bond_v3 as cfg
        from data.pipelines.bond_v3 import load_all
        from strategies.bond_v3.backtest import run, effective_weights
    elif version == "v2":
        import configs.bond_v2 as cfg
        from data.pipelines.bond_v2 import load_all
        from strategies.bond_v2.backtest import run, effective_weights
    else:
        import configs.bond_v1 as cfg
        from data.pipelines.bond_v1 import load_all
        from strategies.bond_v1.backtest import run, effective_weights

    # Load best params if available
    suffix = f"_{version}" if version != "v1" else ""
    best_file = os.path.join(os.path.dirname(__file__), f"best_params{suffix}.json")
    if os.path.exists(best_file):
        import json
        with open(best_file) as f:
            params = json.load(f)
        for k, v in params.items():
            setattr(cfg, k, v)
        logger.info("Loaded best params from %s", best_file)

    macro, prices = load_all()
    results = run(macro, prices)

    signal_w = results["weights"].iloc[-1]
    as_of = results["weights"].index[-1].date()
    eff_w = effective_weights(signal_w, prices[cfg.ETF_UNIVERSE])

    return eff_w, as_of


def _load_previous(out_dir: str, today: str) -> pd.DataFrame | None:
    """Load the most recent positions file before today."""
    history_path = os.path.join(out_dir, "history.csv")
    if not os.path.exists(history_path):
        return None
    hist = pd.read_csv(history_path)
    prev_dates = hist[hist["date"] < today]["date"].unique()
    if len(prev_dates) == 0:
        return None
    last_date = sorted(prev_dates)[-1]
    logger.info("Previous positions from %s", last_date)
    return hist[hist["date"] == last_date]


def main():
    today = date.today().isoformat()
    out_dir = os.path.join(os.path.dirname(__file__), "positions")
    os.makedirs(out_dir, exist_ok=True)

    prev_df = _load_previous(out_dir, today)

    rows = []
    for version in ["v1", "v2", "v3"]:
        try:
            eff_w, as_of = _compute_weights(version)

            # Get previous weights for this strategy
            prev_weights = {}
            if prev_df is not None:
                prev_v = prev_df[prev_df["strategy"] == version]
                prev_weights = dict(zip(prev_v["etf"], prev_v["weight_pct"]))

            # All ETFs: union of current and previous
            all_etfs = set(e for e, w in eff_w.items() if w > 0.001) | set(prev_weights.keys())

            for etf in sorted(all_etfs):
                current = round(eff_w.get(etf, 0) * 100, 2)
                previous = round(prev_weights.get(etf, 0), 2)
                delta = round(current - previous, 2)

                if delta > _MIN_TRADE_PCT:
                    action = "BUY"
                elif delta < -_MIN_TRADE_PCT:
                    action = "SELL"
                else:
                    action = "HOLD"

                # Estimated round-trip cost in bps for this position change.
                # turnover_fraction = |delta| / 100 (weight as fraction of portfolio)
                turnover_frac = abs(delta) / 100.0
                est_cost_bps = round(
                    turnover_frac * DEFAULT_COST_MODEL.round_trip_bps(turnover_frac), 2
                ) if action != "HOLD" else 0.0

                rows.append({
                    "date": today,
                    "strategy": version,
                    "etf": etf,
                    "prev_weight_pct": previous,
                    "target_weight_pct": current,
                    "delta_pct": delta,
                    "action": action,
                    "est_cost_bps": est_cost_bps,
                })

            n_pos = sum(1 for r in rows if r["strategy"] == version and r["target_weight_pct"] > 0)
            n_trades = sum(1 for r in rows if r["strategy"] == version and r["action"] != "HOLD")
            logger.info("%s: %d positions, %d trades needed (as_of=%s)", version, n_pos, n_trades, as_of)
        except Exception as e:
            logger.error("%s failed: %s", version, e)

    if not rows:
        logger.error("No positions computed — aborting")
        sys.exit(1)

    df = pd.DataFrame(rows)

    # --- Excel workbook with Orders + Portfolio sheets ---
    orders = df[df["action"] != "HOLD"].copy()
    portfolio = df[df["target_weight_pct"] > 0][
        ["date", "strategy", "etf", "target_weight_pct"]
    ].copy()

    xlsx_path = os.path.join(out_dir, f"{today}.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        portfolio.to_excel(writer, sheet_name="Portfolio", index=False)
        if len(orders) > 0:
            orders.to_excel(writer, sheet_name="Orders", index=False)
        else:
            pd.DataFrame([{"note": "No trades needed today"}]).to_excel(
                writer, sheet_name="Orders", index=False
            )
    logger.info("Wrote %s (%d positions, %d trades)", xlsx_path, len(portfolio), len(orders))

    # --- Update history (for next day's comparison) ---
    history_path = os.path.join(out_dir, "history.csv")
    hist_rows = portfolio.rename(columns={"target_weight_pct": "weight_pct"})
    if os.path.exists(history_path):
        existing = pd.read_csv(history_path)
        existing = existing[existing["date"] != today]
        hist_rows = pd.concat([existing, hist_rows], ignore_index=True)
    hist_rows.to_csv(history_path, index=False)
    logger.info("Updated %s", history_path)

    _write_email_files(df, today)


def _write_email_files(df: pd.DataFrame, today: str) -> None:
    """Write /tmp/email_subject.txt and /tmp/email_body.txt for the workflow email step."""
    run_id  = os.environ.get("GITHUB_RUN_ID", "")
    repo    = os.environ.get("GITHUB_REPOSITORY", "magnusbmorsund/bond-trading")
    pos_url = f"https://github.com/{repo}/tree/main/positions"
    run_url = f"https://github.com/{repo}/actions/runs/{run_id}" if run_id else ""

    trades_by_version: dict[str, pd.DataFrame] = {}
    any_trades = False
    for version in ["v1", "v2", "v3"]:
        vdf    = df[df["strategy"] == version]
        trades = vdf[vdf["action"] != "HOLD"].sort_values("delta_pct", key=abs, ascending=False)
        trades_by_version[version] = trades
        if len(trades) > 0:
            any_trades = True

    subject = (
        f"ACTION NEEDED — Bond Strategy — {today}"
        if any_trades else
        f"No action — Bond Strategy — {today}"
    )

    lines = [subject, ""]

    for version in ["v1", "v2", "v3"]:
        trades = trades_by_version[version]
        n = len(trades)
        if n == 0:
            lines.append(f"{version.upper()} — No trades needed")
        else:
            lines.append(f"{version.upper()} — {n} trade{'s' if n > 1 else ''} needed")
            for _, r in trades.iterrows():
                sign = "+" if r["delta_pct"] > 0 else ""
                cost = f"   ~{r['est_cost_bps']:.1f} bps" if r.get("est_cost_bps", 0) > 0 else ""
                lines.append(
                    f"  {r['action']:>4s}  {r['etf']:<5s}"
                    f"  {r['prev_weight_pct']:6.2f}% -> {r['target_weight_pct']:6.2f}%"
                    f"  ({sign}{r['delta_pct']:.2f}%){cost}"
                )
            total_cost = trades["est_cost_bps"].sum()
            if total_cost > 0:
                lines.append(f"  Est. cost: {total_cost:.1f} bps")
        lines.append("")

    lines += ["---", f"Positions: {pos_url}"]
    if run_url:
        lines.append(f"Run log:   {run_url}")

    body = "\n".join(lines)

    with open("/tmp/email_subject.txt", "w") as f:
        f.write(subject)
    with open("/tmp/email_body.txt", "w") as f:
        f.write(body)
    logger.info("Email content written — %s", subject)


if __name__ == "__main__":
    main()
