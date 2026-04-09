"""
Daily weights report — run by GitHub Actions.
Computes effective weights for V1/V2/V3 and writes to positions/YYYY-MM-DD.csv.
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


def main():
    today = date.today().isoformat()
    out_dir = os.path.join(os.path.dirname(__file__), "positions")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for version in ["v1", "v2", "v3"]:
        try:
            eff_w, as_of = _compute_weights(version)
            for etf, weight in eff_w.items():
                if weight > 0.001:
                    rows.append({
                        "date": today,
                        "strategy": version,
                        "etf": etf,
                        "weight_pct": round(weight * 100, 2),
                    })
            logger.info("%s: %d positions (as_of=%s)", version, sum(1 for r in rows if r["strategy"] == version), as_of)
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

    # Append to history
    history_path = os.path.join(out_dir, "history.csv")
    if os.path.exists(history_path):
        existing = pd.read_csv(history_path)
        # Remove any existing rows for today (idempotent re-runs)
        existing = existing[existing["date"] != today]
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(history_path, index=False)
    logger.info("Updated %s", history_path)

    # Print summary for GitHub Actions log / email body
    print(f"\n{'='*50}")
    print(f"DAILY POSITIONS — {today}")
    print(f"{'='*50}")
    for version in ["v1", "v2", "v3"]:
        vdf = pd.DataFrame(rows)
        vdf = vdf[vdf["strategy"] == version].sort_values("weight_pct", ascending=False)
        if vdf.empty:
            continue
        print(f"\n  {version.upper()}:")
        for _, r in vdf.iterrows():
            bar = "#" * int(r["weight_pct"] / 2.5)
            print(f"    {r['etf']:>5s}  {r['weight_pct']:6.2f}%  {bar}")
        print(f"    {'─'*35}")
        print(f"    {'Sum':>5s}  {vdf['weight_pct'].sum():6.2f}%")


if __name__ == "__main__":
    main()
