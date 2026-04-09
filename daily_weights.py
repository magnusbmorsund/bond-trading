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

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


def _compute_weights(version: str):
    """Return (effective_weights Series, as_of date) for a strategy version."""
    if version == "v3":
        import config_v3 as cfg
        from data.pipeline_v3 import load_all
        from strategy_v3.backtest import run, effective_weights
    elif version == "v2":
        import config_v2 as cfg
        from data.pipeline_v2 import load_all
        from strategy_v2.backtest import run, effective_weights
    else:
        import config as cfg
        from data.pipeline import load_all
        from strategy.backtest import run, effective_weights

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

                if delta > 0.05:
                    action = "BUY"
                elif delta < -0.05:
                    action = "SELL"
                else:
                    action = "HOLD"

                rows.append({
                    "date": today,
                    "strategy": version,
                    "etf": etf,
                    "prev_weight_pct": previous,
                    "target_weight_pct": current,
                    "delta_pct": delta,
                    "action": action,
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

    # Write daily file
    daily_path = os.path.join(out_dir, f"{today}.csv")
    df.to_csv(daily_path, index=False)
    logger.info("Wrote %s", daily_path)

    # Update history (only target weights, for next day's comparison)
    history_path = os.path.join(out_dir, "history.csv")
    hist_rows = df[df["target_weight_pct"] > 0][["date", "strategy", "etf", "target_weight_pct"]].copy()
    hist_rows = hist_rows.rename(columns={"target_weight_pct": "weight_pct"})
    if os.path.exists(history_path):
        existing = pd.read_csv(history_path)
        existing = existing[existing["date"] != today]
        hist_rows = pd.concat([existing, hist_rows], ignore_index=True)
    hist_rows.to_csv(history_path, index=False)
    logger.info("Updated %s", history_path)

    # Print summary for GitHub Actions log / email body
    print(f"\n{'='*60}")
    print(f"DAILY ORDERS — {today}")
    print(f"{'='*60}")
    for version in ["v1", "v2", "v3"]:
        vdf = df[df["strategy"] == version].copy()
        if vdf.empty:
            continue
        trades = vdf[vdf["action"] != "HOLD"].sort_values("delta_pct", key=abs, ascending=False)
        holds = vdf[vdf["action"] == "HOLD"]

        print(f"\n  {version.upper()}:")
        if len(trades) > 0:
            for _, r in trades.iterrows():
                sign = "+" if r["delta_pct"] > 0 else ""
                print(f"    {r['action']:>4s}  {r['etf']:>5s}  {r['prev_weight_pct']:6.2f}% → {r['target_weight_pct']:6.2f}%  ({sign}{r['delta_pct']:.2f}%)")
        else:
            print(f"    No trades needed")
        if len(holds) > 0:
            hold_etfs = ", ".join(holds["etf"])
            print(f"    HOLD  {hold_etfs}")
        print(f"    {'─'*50}")
        total = vdf["target_weight_pct"].sum()
        print(f"    Total weight: {total:.2f}%")


if __name__ == "__main__":
    main()
