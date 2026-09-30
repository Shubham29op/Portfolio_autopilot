import pytest

from autopilot.config import load_settings
from autopilot.costs import delivery_charges

CFG = load_settings()["charges"]


def test_stock_buy_1_lakh():
    c = delivery_charges("buy", 100_000, "stock", CFG)
    assert c.brokerage == 0
    assert c.stt == pytest.approx(100)
    assert c.stamp == pytest.approx(15)
    assert c.dp == 0
    assert c.total == pytest.approx(100 + 2.97 + 0.1 + 15 + 0.18 * (2.97 + 0.1), abs=0.02)


def test_stock_sell_has_dp_and_no_stamp():
    c = delivery_charges("sell", 100_000, "stock", CFG)
    assert c.stamp == 0
    assert c.dp == pytest.approx(15.34)
    assert c.stt == pytest.approx(100)


def test_etf_stt_is_tiny_and_gold_has_none():
    assert delivery_charges("buy", 100_000, "equity_etf", CFG).stt == 0
    assert delivery_charges("sell", 100_000, "equity_etf", CFG).stt == pytest.approx(1)
    assert delivery_charges("sell", 100_000, "gold_etf", CFG).stt == 0


def test_dp_can_be_skipped_for_second_sell_same_day():
    assert delivery_charges("sell", 50_000, "stock", CFG, charge_dp=False).dp == 0


def test_inr_format():
    from autopilot.fmt import inr
    assert inr(160020.5) == "₹1,60,020.50"
    assert inr(-8256.94) == "−₹8,256.94"
    assert inr(999) == "₹999.00"
