"""Market regime tiers: how much equity risk to run today."""
from __future__ import annotations

import pandas as pd

TIERS = ("off", "cautious", "normal", "strong")


def regime_today(bench_feats: pd.Series, cfg: dict) -> dict:
    r = cfg["regime"]
    above = bool(bench_feats.get("dist_sma200", float("nan")) > 0)
    vol = float(bench_feats.get("vol_21", float("nan")))
    calm = vol < r["vol_high"]
    breadth = float(bench_feats.get("breadth_sma200", 0.5) or 0.5)
    mkt_ret_63 = float(bench_feats.get("mkt_ret_63", 0.0) or 0.0)
    if not above:
        level = "off"
    elif not calm or breadth < r.get("breadth_weak", 0.4):
        level = "cautious"
    elif breadth > r.get("breadth_strong", 0.6) and mkt_ret_63 > 0:
        level = "strong"
    else:
        level = "normal"
    tier = r["tiers"][level]
    why = [("Nifty above" if above else "Nifty below") + " its 200-day average",
           f"volatility {vol:.0%} {'normal' if calm else 'elevated'}",
           f"{breadth:.0%} of the universe in uptrends"]
    return {"risk_on": level != "off", "level": level, "risk_mult": tier["risk_mult"],
            "extra_positions": tier["extra_positions"], "why": ", ".join(why), "bench_vol": vol}
