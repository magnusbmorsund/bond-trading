"""Live stop-fill tracker — measure intended-stop vs actual-fill slippage.

Purpose: validate the backtest's "fill ≈ stop price" assumption against reality.
The fill-at-stop research (see memory: stop-fill-model) showed the whole strategy
hinges on how close real fills land to the stop. This module logs every stop
trigger so we can measure the real slippage and feed it back into the model.

Mechanics — a persistent **stop book** that mirrors a Nordnet *glidende* stop:
  • Each held position trails a stop at `peak_since_placement * (1 - stop_pct)`.
  • `peak` RATCHETS UP on the daily High and never falls (unlike the backtest's
    rolling-window peak) — this is what a live trailing order actually does.
  • A breach is detected when the day's Low pierces the stop level. The fill is
    estimated from the OHLC: a gap (Open < stop) fills at the Open (worse); an
    intraday cross fills ~at the stop level.
  • `actual_fill` is left blank for the user to enter from Nordnet trade
    confirmations; `reconcile`/`report` then compute realised slippage in bps.

Pure functions + thin JSON/CSV persistence so the logic is unit-testable.
"""
from __future__ import annotations
import csv
import json
import os
from dataclasses import dataclass

EVENT_FIELDS = [
    "date", "strategy", "etf", "placed_date", "peak_price", "stop_pct",
    "stop_price", "open", "high", "low", "close",
    "breach_type", "est_fill", "est_slip_bps",
    "actual_fill", "actual_slip_bps", "note",
]


def estimate_fill(stop_price: float, open_: float | None) -> tuple[float, str]:
    """Estimate the fill price of a triggered stop from the day's open.

    A stop-sell rests at `stop_price`. If the session opens BELOW it (a gap), the
    order fills at the open — worse than the stop. Otherwise the price crossed the
    stop intraday and fills ≈ at the stop level.
    Returns (fill_price, breach_type).
    """
    if open_ is not None and open_ < stop_price:
        return float(open_), "gap_open"
    return float(stop_price), "intraday"


def slip_bps(stop_price: float, fill: float | None) -> float | None:
    """Slippage in bps, POSITIVE = filled worse (below) the stop. None if no fill."""
    if fill is None or not stop_price:
        return None
    return round((stop_price - fill) / stop_price * 1e4, 1)


def _to_float(x):
    try:
        f = float(x)
        return f if f == f else None  # drop NaN
    except (TypeError, ValueError):
        return None


def update_book(
    book: dict,
    strategy: str,
    today: str,
    held: dict,        # {etf: stop_pct}  — positions held today with their stop %
    ohlc: dict,        # {etf: {"Open","High","Low","Close"}}
) -> tuple[dict, list[dict]]:
    """Advance the per-strategy stop book by one day and return (book, events).

    `book[strategy]` maps etf -> {placed_date, peak, stop_pct}. For each held etf
    the peak ratchets on today's High and the stop trails below it; a triggered
    stop emits an event and is removed from the book. Positions no longer held
    (exited at rebalance, not stopped) are dropped silently.
    """
    sbook = book.setdefault(strategy, {})
    events: list[dict] = []

    # Drop positions that are no longer held and weren't stopped (sold at rebalance)
    for etf in [e for e in sbook if e not in held]:
        del sbook[etf]

    for etf, stop_pct in held.items():
        bars = ohlc.get(etf) or {}
        o = _to_float(bars.get("Open")); h = _to_float(bars.get("High"))
        lo = _to_float(bars.get("Low")); c = _to_float(bars.get("Close"))

        rec = sbook.get(etf)
        if rec is None:
            # New position — seed the peak with the best price we can see today.
            seed = max([v for v in (h, c, o) if v is not None], default=None)
            rec = {"placed_date": today, "peak": seed, "stop_pct": float(stop_pct)}
            sbook[etf] = rec

        rec["stop_pct"] = float(stop_pct)  # strategy may re-rate the stop %
        # Ratchet the peak up on today's high (never down).
        if h is not None:
            rec["peak"] = max(rec["peak"] or h, h)
        peak = rec["peak"]
        if peak is None:
            continue
        stop_price = peak * (1.0 - rec["stop_pct"])

        # Breach if the day's low pierced the stop.
        if lo is not None and lo <= stop_price:
            fill, btype = estimate_fill(stop_price, o)
            events.append({
                "date": today, "strategy": strategy, "etf": etf,
                "placed_date": rec["placed_date"], "peak_price": round(peak, 4),
                "stop_pct": round(rec["stop_pct"], 4), "stop_price": round(stop_price, 4),
                "open": o, "high": h, "low": lo, "close": c,
                "breach_type": btype, "est_fill": round(fill, 4),
                "est_slip_bps": slip_bps(stop_price, fill),
                "actual_fill": "", "actual_slip_bps": "", "note": "",
            })
            del sbook[etf]  # position is out; a new placement starts if re-bought

    return book, events


# ── persistence ──────────────────────────────────────────────────────────────

def load_book(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_book(book: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(book, f, indent=2, sort_keys=True)


def append_events(events: list[dict], path: str) -> None:
    if not events:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EVENT_FIELDS)
        if new:
            w.writeheader()
        for e in events:
            w.writerow({k: e.get(k, "") for k in EVENT_FIELDS})


def report(path: str) -> dict:
    """Aggregate the event log into slippage stats (estimated and actual)."""
    if not os.path.exists(path):
        return {"events": 0}
    rows = list(csv.DictReader(open(path)))
    est  = [_to_float(r["est_slip_bps"]) for r in rows]
    est  = [x for x in est if x is not None]
    act  = [_to_float(r["actual_slip_bps"]) for r in rows]
    act  = [x for x in act if x is not None]
    gaps = sum(1 for r in rows if r.get("breach_type") == "gap_open")

    def _mean(xs): return round(sum(xs) / len(xs), 1) if xs else None
    def _med(xs):
        if not xs: return None
        s = sorted(xs); n = len(s)
        return round((s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2), 1)

    return {
        "events": len(rows),
        "gap_opens": gaps,
        "gap_rate_pct": round(100 * gaps / len(rows), 1) if rows else None,
        "est_slip_bps_mean": _mean(est), "est_slip_bps_median": _med(est),
        "actual_logged": len(act),
        "actual_slip_bps_mean": _mean(act), "actual_slip_bps_median": _med(act),
    }
