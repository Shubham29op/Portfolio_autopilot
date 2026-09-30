import pandas as pd

from autopilot.broker.base import Order
from autopilot.broker.paper import PaperBroker
from autopilot.config import Universe, load_settings
from autopilot.regime import regime_today
from autopilot.sizing import risk_fraction

CFG = load_settings()
U = Universe.load()


def bar(o, h, l, c):
    return {"open": o, "high": h, "low": l, "close": c}


def test_scale_out_keeps_half_with_breakeven_stop():
    b = PaperBroker(CFG, U, 1_000_000)
    b.start_day("2026-01-05")
    b.submit(Order(id="O1", symbol="INFY", side="buy", qty=100, reason="entry", created="x",
                   stop_atr=2, atr=2, target_atr=3, prob=0.6))
    b.process_open({"INFY": bar(100, 100, 100, 100)})
    b.start_day("2026-01-06")
    b.process_open({"INFY": bar(101, 101, 101, 101)})
    b.process_intraday({"INFY": bar(101, 110, 100, 108)})       # target 106.05 hit
    pos = b.positions["INFY"]
    assert pos.qty == 50 and b.fills_today[-1].reason == "target"
    g = b.gtts[pos.gtt_id]
    assert g.stop_trigger >= pos.avg_price and g.target_trigger > 1e9   # breakeven+, trailing only


def test_risk_fraction_scales_and_caps():
    base = CFG["risk"]["risk_per_trade"]
    low = risk_fraction(prob=0.55, p_min=0.55, regime={"risk_mult": 1.0}, mkt_vol=0.16, cfg=CFG)
    high = risk_fraction(prob=0.85, p_min=0.55, regime={"risk_mult": 1.25}, mkt_vol=0.12, cfg=CFG)
    off = risk_fraction(prob=0.85, p_min=0.55, regime={"risk_mult": 0.0}, mkt_vol=0.12, cfg=CFG)
    assert low < base and high == CFG["risk"]["max_risk_per_trade"] and off == 0
    calm = risk_fraction(prob=0.6, p_min=0.55, regime=None, mkt_vol=0.10, cfg=CFG)
    wild = risk_fraction(prob=0.6, p_min=0.55, regime=None, mkt_vol=0.40, cfg=CFG)
    assert calm > wild


def test_regime_tiers():
    strong = regime_today(pd.Series({"dist_sma200": 0.05, "vol_21": 0.14, "breadth_sma200": 0.7, "mkt_ret_63": 0.04}), CFG)
    caut = regime_today(pd.Series({"dist_sma200": 0.05, "vol_21": 0.35, "breadth_sma200": 0.7, "mkt_ret_63": 0.04}), CFG)
    off = regime_today(pd.Series({"dist_sma200": -0.05, "vol_21": 0.14, "breadth_sma200": 0.7, "mkt_ret_63": 0.04}), CFG)
    assert strong["level"] == "strong" and strong["extra_positions"] == 2
    assert caut["level"] == "cautious" and caut["risk_mult"] == 0.5
    assert off["level"] == "off" and not off["risk_on"]
