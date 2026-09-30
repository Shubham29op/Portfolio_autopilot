"""Turns model probabilities + trend into entry and exit decisions.

The model ranks; volatility maths sets prices. Stops/targets are ATR multiples,
with the stop distance widened for higher-conviction entries (more room).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def stop_atr_for(p: float, cfg: dict) -> float:
    s = cfg["strategy"]
    conf = np.clip((p - s["p_enter"]) / max(1e-9, 1 - s["p_enter"]) * 2, 0, 1)
    return round(s["stop_atr_low_conf"] + conf * (s["stop_atr_high_conf"] - s["stop_atr_low_conf"]), 2)


def trading_days_between(a: str, b: str) -> int:
    return int(np.busday_count(pd.Timestamp(a).date(), pd.Timestamp(b).date()))


def exit_decisions(positions: dict, feats: pd.DataFrame, probs: pd.Series, date: str,
                   cfg: dict) -> list[dict]:
    """Discretionary exits (time stop, max hold, model exit). Stops live in GTTs."""
    s = cfg["strategy"]
    out = []
    for sym, pos in positions.items():
        held = trading_days_between(pos.entry_date, date)
        close = feats.at[sym, "close"] if sym in feats.index else pos.last_price
        gain_atr = (close - pos.avg_price) / pos.atr_at_entry if pos.atr_at_entry else 0
        p = float(probs.get(sym, np.nan))
        if held >= s["max_hold_days"]:
            out.append(dict(symbol=sym, reason="max_hold",
                            why=f"Held {held} trading days, the 1-year maximum."))
        elif held < s["min_hold_days"]:
            continue
        elif held >= s["time_stop_days"] and gain_atr < s["time_stop_min_gain_atr"]:
            out.append(dict(symbol=sym, reason="time_stop",
                            why=f"No progress after {held} days (+{gain_atr:.1f} ATR); capital is better used elsewhere."))
        elif not np.isnan(p) and p < s["p_exit"]:
            out.append(dict(symbol=sym, reason="model_exit",
                            why=f"Model confidence fell to {p:.0%} (exit below {s['p_exit']:.0%})."))
    return out


def technical_score(f: pd.Series) -> float:
    """0-100: risk-adjusted momentum rank, trimmed when overbought."""
    base = 100 * float(f.get("rank_mom_risk_adj", 0.5) if pd.notna(f.get("rank_mom_risk_adj")) else 0.5)
    return base * (0.8 if f.get("rsi_14", 50) > 75 else 1.0)


def ml_score(p: float, rank_exp_ret: float | None, cfg: dict) -> float:
    w = cfg["strategy"].get("rank_exp_ret_weight", 0.0)
    if rank_exp_ret is None or rank_exp_ret != rank_exp_ret:
        return 100 * p
    return 100 * ((1 - w) * p + w * rank_exp_ret)


def combined_score(fund: float | None, p: float, tech: float, weights: dict, ml: float | None = None) -> float:
    parts = {"ml": (ml if ml is not None else p * 100, weights["ml"]), "technical": (tech, weights["technical"])}
    if fund is not None:
        parts["fundamental"] = (fund, weights["fundamental"])
    tot = sum(w for _, w in parts.values())
    return round(sum(v * w for v, w in parts.values()) / tot, 1)


def entry_candidates(feats: pd.DataFrame, probs: pd.Series, regime: dict, universe,
                     held: set, cooldown: dict, date: str, cfg: dict,
                     research: dict | None = None, news_blocks: dict | None = None,
                     news_boosts: dict | None = None, exp_ret: pd.Series | None = None) -> tuple[list[dict], list[dict]]:
    """Returns (ranked candidates, per-symbol signal rows for the dashboard).

    research=None  -> technical/ML only (backtests).
    research=dict  -> stage-3 decision: stocks must be on this month's approved list.
    """
    s, m = cfg["strategy"], cfg["model"]
    rc = cfg.get("research", {})
    p_min = rc.get("p_enter_approved", s["p_enter"]) if research is not None else s["p_enter"]
    rows, cands = [], []
    er_rank = exp_ret.rank(pct=True) if exp_ret is not None and len(exp_ret) else None
    for sym in universe.tradable:
        if sym not in feats.index or sym not in probs.index:
            continue
        f, p = feats.loc[sym], float(probs[sym])
        er = float(exp_ret[sym]) if exp_ret is not None and sym in exp_ret.index else None
        er_r = float(er_rank[sym]) if er_rank is not None and sym in er_rank.index else None
        if er is not None and er < 0:
            blockers_er = True
        else:
            blockers_er = False
        inst = universe[sym]
        blockers = []
        fund, fund_note = None, ""
        if research is not None:
            if inst.needs_research:
                a = research.get("approved", {}).get(sym)
                if research.get("missing"):
                    blockers.append("monthly research not run yet")
                elif a is None:
                    blockers.append("not on this month's approved list")
                else:
                    fund = a["fundamental_score"]
                    fund_note = f"fundamentals {fund:.0f}/100"
                    nrd = a.get("next_results_date")
                    if nrd and 0 <= trading_days_between(date, nrd) <= rc["earnings_blackout_days"]:
                        blockers.append(f"results due {nrd}")
            elif inst.type == "equity_etf":
                fund = research.get("sector_scores", {}).get(inst.sector)
                if fund is not None:
                    fund_note = f"sector score {fund:.0f}/100"
        if sym in held:
            blockers.append("already held")
        if news_blocks and sym in news_blocks:
            blockers.append(f"negative news ({news_blocks[sym]['why']}) until {news_blocks[sym]['until']}")
        if sym in cooldown and trading_days_between(cooldown[sym], date) < s["cooldown_days"]:
            blockers.append("cooling down after exit")
        if p < p_min:
            blockers.append(f"probability {p:.0%} below {p_min:.0%}")
        if not (f["dist_sma200"] > 0 and f["sma50_above_200"] > 0):
            blockers.append("trend not confirmed")
        if inst.is_equity_risk and not regime["risk_on"]:
            blockers.append("market regime is risk-off")
        if pd.isna(f["atr"]) or f["atr"] <= 0:
            blockers.append("no ATR")
        if blockers_er:
            blockers.append(f"model expects a negative return ({er:+.1%})")
        tech = technical_score(f)
        mls = ml_score(p, er_r, cfg)
        combined = combined_score(fund, p, tech, rc["weights"], mls) if research is not None else None
        boost = (news_boosts or {}).get(sym)
        if combined is not None and boost:
            combined = min(100.0, combined + boost["points"])
        if combined is not None and combined < rc["min_combined_score"]:
            blockers.append(f"combined score {combined:.0f} below {rc['min_combined_score']}")
        eligible = not blockers
        stop_atr = stop_atr_for(p, cfg)
        ev = p * m["label_target_atr"] - (1 - p) * stop_atr
        why = (f"{p:.0%} chance of +{m['label_target_atr']:.0f} ATR before -{m['label_stop_atr']:.0f} ATR "
               f"within ~{m['horizon_days']} days"
               + (f"; expected {er:+.1%} in ~{m['horizon_days']} days" if er is not None else "")
               + f"; {f['dist_sma200']:+.1%} vs 200-day average; "
               f"6-month momentum in top {1 - f['rank_ret_126']:.0%} of universe")
        if combined is not None:
            why = f"Combined {combined:.0f}/100" + (f" ({fund_note}, technical {tech:.0f})" if fund_note
                                                     else f" (technical {tech:.0f})") + f". {why}"
            if boost:
                why += f"; good news +{boost['points']} ({boost['why']})"
        rows.append(dict(symbol=sym, prob=round(p, 4), eligible=eligible,
                         action="candidate" if eligible else "skip",
                         why=why if eligible else "; ".join(blockers)))
        if eligible:
            cands.append(dict(symbol=sym, prob=p, ev=ev, stop_atr=stop_atr, atr=float(f["atr"]),
                              close=float(f["close"]), why=why, sector=inst.sector, type=inst.type,
                              combined=combined, ml=mls))
    key = (lambda c: c["combined"]) if research is not None else (lambda c: c["ml"])
    cands.sort(key=key, reverse=True)
    return cands, rows
