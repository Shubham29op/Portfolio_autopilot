"""Zerodha NSE equity-delivery charges. Tax is intentionally excluded in v1."""
from __future__ import annotations

from dataclasses import asdict, dataclass

CRORE = 1e7


@dataclass
class Charges:
    brokerage: float = 0.0
    stt: float = 0.0
    exchange: float = 0.0
    sebi: float = 0.0
    ipft: float = 0.0
    stamp: float = 0.0
    gst: float = 0.0
    dp: float = 0.0

    @property
    def total(self) -> float:
        return round(sum(asdict(self).values()), 2)

    def to_dict(self) -> dict:
        d = {k: round(v, 2) for k, v in asdict(self).items()}
        d["total"] = self.total
        return d


def delivery_charges(side: str, value: float, instrument_type: str, cfg: dict,
                     charge_dp: bool = True) -> Charges:
    """Charges for one executed delivery order.

    side: "buy" | "sell"; value: price * qty in INR.
    charge_dp: DP is per scrip per day on sells; the broker passes False for
    the 2nd+ sell of the same scrip on the same day.
    """
    if side not in ("buy", "sell"):
        raise ValueError(f"bad side {side}")
    c = Charges()
    c.brokerage = value * cfg["brokerage_pct"]
    c.stt = value * cfg["stt"][instrument_type][side]
    c.exchange = value * cfg["exchange_txn_pct"]
    c.sebi = value * cfg["sebi_per_crore"] / CRORE
    c.ipft = value * cfg["ipft_per_crore"] / CRORE
    if side == "buy":
        c.stamp = value * cfg["stamp_buy_pct"]
    c.gst = (c.brokerage + c.exchange + c.sebi + c.ipft) * cfg["gst_pct"]
    if side == "sell" and charge_dp:
        c.dp = cfg["dp_per_scrip_sell"]  # already GST-inclusive
    return c


def round_trip_cost_pct(value: float, instrument_type: str, cfg: dict) -> float:
    """Approximate buy+sell cost as a fraction of value (used for sizing/filters)."""
    b = delivery_charges("buy", value, instrument_type, cfg).total
    s = delivery_charges("sell", value, instrument_type, cfg).total
    return (b + s) / value if value else 0.0
