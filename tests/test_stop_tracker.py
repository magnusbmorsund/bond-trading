"""Tests for analysis.stop_tracker — pure stop-book / fill logic, no data fetch."""
from analysis.stop_tracker import estimate_fill, slip_bps, update_book


def test_estimate_fill_intraday_cross():
    # open above stop → crossed intraday → fills ~at the stop
    fill, btype = estimate_fill(stop_price=90.0, open_=100.0)
    assert fill == 90.0 and btype == "intraday"


def test_estimate_fill_gap_open():
    # open below stop → gapped through → fills at the (worse) open
    fill, btype = estimate_fill(stop_price=90.0, open_=85.0)
    assert fill == 85.0 and btype == "gap_open"


def test_slip_bps_sign():
    # filled below the stop = positive (worse) slippage
    assert slip_bps(100.0, 99.0) == 100.0      # 1% = 100 bps
    assert slip_bps(100.0, 100.0) == 0.0
    assert slip_bps(100.0, None) is None


def test_peak_ratchets_and_no_breach():
    # Day 1: peak=110, stop=99; low=100 stays above → no breach
    book, ev = update_book({}, "s", "2026-01-01", {"AAA": 0.10},
                           {"AAA": {"Open": 100, "High": 110, "Low": 100, "Close": 108}})
    assert ev == []
    assert book["s"]["AAA"]["peak"] == 110
    # Day 2: higher high ratchets peak to 120, stop=108; low=112 stays above → no breach
    book, ev = update_book(book, "s", "2026-01-02", {"AAA": 0.10},
                           {"AAA": {"Open": 109, "High": 120, "Low": 112, "Close": 119}})
    assert book["s"]["AAA"]["peak"] == 120 and ev == []


def test_breach_emits_event_and_drops_position():
    book = {"s": {"AAA": {"placed_date": "2026-01-01", "peak": 120.0, "stop_pct": 0.10}}}
    # stop = 120*0.9 = 108; today gaps to open 105 (<108) and low 104 → gap fill at 105
    book, ev = update_book(book, "s", "2026-01-05", {"AAA": 0.10},
                           {"AAA": {"Open": 105, "High": 106, "Low": 104, "Close": 104}})
    assert len(ev) == 1
    e = ev[0]
    assert e["stop_price"] == 108.0
    assert e["breach_type"] == "gap_open" and e["est_fill"] == 105.0
    assert e["est_slip_bps"] == slip_bps(108.0, 105.0)
    assert "AAA" not in book["s"]  # position removed after trigger


def test_unheld_position_dropped_silently():
    book = {"s": {"AAA": {"placed_date": "2026-01-01", "peak": 120.0, "stop_pct": 0.10}}}
    book, ev = update_book(book, "s", "2026-01-06", {},  # AAA no longer held
                           {})
    assert ev == [] and "AAA" not in book["s"]
