import pytest

from autopilot.broker.base import Order
from autopilot.broker.paper import PaperBroker
from autopilot.config import Universe, load_settings

CFG = load_settings()
U = Universe.load()


def bar(o, h, l, c):
    return {"open": o, "high": h, "low": l, "close": c}


def bought(price=100.0, atr=2.0, stop_atr=2.0):
    b = PaperBroker(CFG, U, 1_000_000)
    b.start_day("2026-01-05")
    b.submit(Order(id="O1", symbol="INFY", side="buy", qty=100, reason="entry", created="2026-01-02",
                   limit_price=price * 1.03, stop_atr=stop_atr, atr=atr, target_atr=4.0, prob=0.6))
    b.process_open({"INFY": bar(price, price, price, price)})
    return b


def test_entry_creates_gtt_oco():
    b = bought()
    pos = b.positions["INFY"]
    g = b.gtts[pos.gtt_id]
    fill_px = 100 * (1 + CFG["execution"]["slippage_pct"])
    assert g.stop_trigger == pytest.approx(fill_px - 4.0, abs=0.01)
    assert g.target_trigger == pytest.approx(fill_px + 8.0, abs=0.01)
    assert b.cash < 1_000_000 - 100 * 100   # paid price + charges


def test_entry_skipped_when_gapping_above_limit():
    b = PaperBroker(CFG, U, 1_000_000)
    b.start_day("2026-01-05")
    b.submit(Order(id="O1", symbol="INFY", side="buy", qty=10, reason="entry", created="x",
                   limit_price=103, stop_atr=2, atr=2, target_atr=4))
    b.process_open({"INFY": bar(110, 111, 109, 110)})
    assert "INFY" not in b.positions
    assert b.processed_orders[0].status == "cancelled"


def test_intraday_stop_fills_near_trigger():
    b = bought()
    g = next(iter(b.gtts.values()))
    b.start_day("2026-01-06")
    b.process_open({"INFY": bar(99, 99.5, 99, 99)})
    b.process_intraday({"INFY": bar(99, 99.5, 90, 92)})
    assert "INFY" not in b.positions
    assert b.fills_today[-1].reason == "stop_loss"
    assert b.fills_today[-1].price >= g.stop_limit


def test_gap_down_through_limit_leaves_position_unprotected():
    b = bought()
    b.start_day("2026-01-06")
    # opens far below the stop limit and never trades back up to it
    b.process_open({"INFY": bar(80, 82, 78, 81)})
    b.process_intraday({"INFY": bar(80, 82, 78, 81)})
    unprotected = b.end_day({"INFY": 81})
    assert unprotected == ["INFY"]
    assert b.positions["INFY"].unprotected and b.positions["INFY"].gtt_id is None


def test_target_hit():
    b = bought()
    b.start_day("2026-01-06")
    b.process_open({"INFY": bar(101, 101, 101, 101)})
    b.process_intraday({"INFY": bar(101, 120, 100, 115)})
    assert b.fills_today[-1].reason == "target"
    assert b.fills_today[-1].pnl > 0


def test_trailing_modify_only_raises():
    b = bought()
    pos = b.positions["INFY"]
    old = b.gtts[pos.gtt_id].stop_trigger
    assert not b.modify_stop("INFY", old - 1)
    assert b.modify_stop("INFY", old + 1)


def test_state_roundtrip():
    b = bought()
    b2 = PaperBroker(CFG, U, 0)
    b2.load_state(b.to_state())
    assert b2.cash == b.cash and set(b2.positions) == {"INFY"} and b2.gtts.keys() == b.gtts.keys()


def test_tick_mode_stop_fills_between_trigger_and_limit():
    b = bought()
    b.process_tick({"INFY": 96.0})          # trigger 96.05, limit ~95.57
    assert "INFY" not in b.positions


def test_tick_mode_gap_below_limit_waits_then_fills_on_recovery():
    b = bought()
    b.process_tick({"INFY": 95.0})          # below the stop limit: order rests
    assert "INFY" in b.positions
    b.process_tick({"INFY": 95.8})          # trades back through the limit
    assert "INFY" not in b.positions
