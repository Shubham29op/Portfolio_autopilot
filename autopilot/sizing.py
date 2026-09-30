"""Position sizing: risk a fixed fraction of equity per trade, then apply caps."""
from __future__ import annotations

import math

from autopilot.costs import delivery_charges


def risk_fraction(*, prob: float, p_min: float, regime: dict | None, mkt_vol: float | None, cfg: dict) -> float:
    """Risk per trade = base x conviction x regime x market-vol scaling, capped."""
    rk = cfg["risk"]
    c = rk["conviction_mult"]
    conv = c["min"] + (c["max"] - c["min"]) * min(1.0, max(0.0, (prob - p_min) / max(1e-9, 0.80 - p_min)))
    reg = (regime or {}).get("risk_mult", 1.0)
    vol_mult = 1.0
    if mkt_vol and mkt_vol == mkt_vol and mkt_vol > 0:
        vol_mult = min(1.25, max(0.5, rk["market_vol_ref"] / mkt_vol))
    return min(rk["max_risk_per_trade"], rk["risk_per_trade"] * conv * reg * vol_mult)


def size_entry(*, price: float, atr: float, stop_atr: float, equity: float, cash_available: float,
               sector_exposure: float, instrument_type: str, cfg: dict,
               risk_frac: float | None = None) -> tuple[int, str]:
    rk = cfg["risk"]
    per_share_risk = stop_atr * atr
    if per_share_risk <= 0 or price <= 0:
        return 0, "invalid price/atr"
    qty = math.floor(equity * (risk_frac or rk["risk_per_trade"]) / per_share_risk)
    limits = {"risk budget": qty,
              "position cap": math.floor(equity * rk["max_position_pct"] / price),
              "sector cap": math.floor(max(0.0, equity * rk["max_sector_pct"] - sector_exposure) / price)}
    spend = max(0.0, cash_available - equity * rk["cash_buffer_pct"])
    ch_pct = delivery_charges("buy", 1e6, instrument_type, cfg["charges"]).total / 1e6
    limits["cash"] = math.floor(spend / (price * (1 + ch_pct)))
    binding = min(limits, key=limits.get)
    qty = max(0, limits[binding])
    if qty * price < rk["min_position_value"]:
        return 0, f"below minimum position value (limited by {binding})"
    return qty, binding
