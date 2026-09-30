"""The daily brain: shared by backtest and live paper so results are comparable."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from autopilot.broker.base import Order
from autopilot.fmt import inr
from autopilot.regime import regime_today
from autopilot.sizing import size_entry
from autopilot.strategy import entry_candidates, exit_decisions

log = logging.getLogger(__name__)

STOPPED_MODES = ("stopped", "halted")


class Engine:
    def __init__(self, cfg: dict, universe, broker, ledger, use_research: bool = False):
        self.cfg, self.universe, self.broker, self.ledger = cfg, universe, broker, ledger
        self.benchmark = cfg["benchmark"]
        self.use_research = use_research and cfg.get("research", {}).get("enabled", False)

    # --------------------------------------------------------- backtest day
    def run_bar_day(self, date: str, bars: dict[str, dict], feats: pd.DataFrame,
                    preds) -> None:
        self.broker.start_day(date)
        self.broker.process_open(bars)
        self.broker.process_intraday(bars)
        closes = {s: b["close"] for s, b in bars.items()}
        self.close_day(date, closes, feats, preds)

    # ----------------------------------------------------------- end of day
    def close_day(self, date: str, closes: dict[str, float], feats: pd.DataFrame,
                  preds) -> None:
        """preds: DataFrame[prob, exp_ret] indexed by symbol (or a bare prob Series)."""
        L, B = self.ledger, self.broker
        xs_score = None
        if isinstance(preds, pd.DataFrame):
            probs = preds["prob"] if "prob" in preds else pd.Series(dtype=float)
            exp_ret = preds["exp_ret"] if "exp_ret" in preds else None
            tradable = [s for s in preds.index if s in self.universe.by_symbol and self.universe[s].tradable]
            if exp_ret is not None and "prob_xs" in preds and tradable:
                p = preds.loc[tradable]
                xs_score = (p["exp_ret"].rank(pct=True) + p["prob_xs"].rank(pct=True)) / 2
                xs_score = xs_score.rank(pct=True)
        else:
            probs, exp_ret = preds, None
        unprotected = B.end_day(closes)
        last = L.get("last_date")
        if last:
            B.accrue_idle_cash(max(1, (pd.Timestamp(date) - pd.Timestamp(last)).days))
        L.set("last_date", date)

        self.record_fills(date)
        cooldown = L.get("cooldown", {})

        # ------- equity, drawdown, circuit breaker
        equity = B.equity(closes)
        mode = self._maybe_auto_resume(date, L.control(), equity, feats)
        peak = max(L.get("peak_equity", equity), equity)
        L.set("peak_equity", peak)
        dd = 1 - equity / peak if peak else 0.0
        bench_px = closes.get(self.benchmark)
        if bench_px and L.get("bench_start") is None:
            L.set("bench_start", [bench_px, equity])
        bs = L.get("bench_start")
        bench_val = bench_px / bs[0] * bs[1] if bench_px and bs else None
        L.equity_point(date, round(equity, 2), round(B.cash, 2), round(B.invested(), 2),
                       round(bench_val, 2) if bench_val else None, round(dd, 4),
                       round(L.get("charges_cum", 0.0), 2))

        orders_today = 0
        protective_syms = set()

        def protect_exit(sym, reason, why):
            if sym in protective_syms or sym not in B.positions:
                return
            protective_syms.add(sym)
            o = Order(id=B.new_id("O"), symbol=sym, side="sell", qty=B.positions[sym].qty,
                      reason=reason, created=date, protective=True)
            B.submit(o)
            L.order(o)
            L.event(date, reason, why, sym)

        for sym in unprotected:
            protect_exit(sym, "stop_unfilled",
                         "Stop triggered on a gap but the limit order did not fill; exiting at next open.")

        if mode not in STOPPED_MODES and dd >= self.cfg["risk"]["circuit_breaker_dd"]:
            B.cancel_queued(lambda o: o.side == "buy")
            for sym in list(B.positions):
                protect_exit(sym, "circuit_breaker",
                             f"Portfolio is {dd:.1%} below its peak; circuit breaker exits everything.")
            L.set_control("halted")
            L.set("halted_on", date)
            L.event(date, "halted", f"Circuit breaker tripped at {dd:.1%} drawdown. "
                                    "Trading halted until you resume it.")
            mode = "halted"

        if mode == "exit_all":
            B.cancel_queued(lambda o: o.side == "buy")
            for sym in list(B.positions):
                protect_exit(sym, "exit_all", "You asked to exit everything.")
            L.set_control("stopped")
            mode = "stopped"

        # ------- trailing stops (protective: run in every mode except stopped/halted)
        if mode not in STOPPED_MODES:
            self._trail_stops(date, closes, feats)

        # ------- monthly research review of holdings (once per new research list)
        research = None
        if self.use_research:
            research = L.get("research_current") or {"approved": {}, "sector_scores": {}, "missing": True}
            if (not research.get("missing") and research["version"] != L.get("research_applied")
                    and mode not in STOPPED_MODES):
                self._apply_research_review(date, research, closes, feats, protect_exit)
                L.set("research_applied", research["version"])

        # ------- discretionary decisions
        regime = {"risk_on": False, "why": "no benchmark data"}
        if self.benchmark in feats.index:
            regime = regime_today(feats.loc[self.benchmark], self.cfg)
        L.set("regime", regime)

        cap = self.cfg["risk"]["max_new_trades_per_day"]
        if mode == "running":
            for d in exit_decisions(B.positions, feats, probs, date, self.cfg, xs_score):
                if orders_today >= cap:
                    break
                if d["symbol"] in protective_syms:
                    continue
                o = Order(id=B.new_id("O"), symbol=d["symbol"], side="sell",
                          qty=B.positions[d["symbol"]].qty, reason=d["reason"], created=date)
                B.submit(o)
                L.order(o)
                L.event(date, d["reason"], d["why"], d["symbol"])
                orders_today += 1

        from autopilot.news.guard import active
        blocks, boosts = active(L.get("news_blocks"), date), active(L.get("news_boosts"), date)
        L.set("news_blocks", blocks)
        L.set("news_boosts", boosts)
        cands, rows = entry_candidates(feats, probs, regime, self.universe, set(B.positions),
                                       cooldown, date, self.cfg, research, blocks, boosts, exp_ret, xs_score)
        if mode == "running" and orders_today < cap:
            p_min = (self.cfg["research"]["p_enter_approved"] if self.use_research
                     else self.cfg["strategy"]["p_enter"])
            orders_today += self._queue_entries(date, cands, closes, cap - orders_today, rows, regime, p_min)
        elif mode != "running":
            for r in rows:
                if r["eligible"]:
                    r["action"], r["why"] = "skip", f"autopilot is {mode}"
        L.signals(date, rows)
        L.set("cooldown", cooldown)
        L.set("broker_state", B.to_state())
        L.set("last_equity", equity)
        L.commit()

    # -------------------------------------------------------------- helpers
    def _apply_research_review(self, date, research, closes, feats, protect_exit) -> None:
        rc = self.cfg["research"]
        for sym, pos in list(self.broker.positions.items()):
            if not self.universe[sym].needs_research:
                continue
            if sym in research.get("red_flags", []):
                protect_exit(sym, "red_flag", "Monthly research found red flags; exiting next session.")
            elif sym not in research["approved"]:
                if rc["dropped_from_list"] == "exit":
                    protect_exit(sym, "dropped_from_list", "No longer on the approved list; exiting.")
                else:
                    c = closes.get(sym, pos.last_price)
                    atr_now = feats.at[sym, "atr"] if sym in feats.index else pos.atr_at_entry
                    new = c - rc["tighten_stop_atr"] * atr_now
                    if self.broker.modify_stop(sym, new):
                        self.ledger.event(date, "tightened", f"Dropped from this month's approved list; "
                                                             f"stop tightened to {inr(new)}.", sym)

    def _maybe_auto_resume(self, date, mode, equity, feats) -> str:
        """Backtests simulate you pressing Resume after a cool-off; live stays manual."""
        n = self.cfg["risk"].get("auto_resume_days")
        if mode != "halted" or not n:
            return mode
        since = self.ledger.get("halted_on", date)
        if (pd.Timestamp(date) - pd.Timestamp(since)).days < n or self.benchmark not in feats.index:
            return mode
        if not regime_today(feats.loc[self.benchmark], self.cfg)["risk_on"]:
            return mode
        self.ledger.set_control("running")
        self.ledger.set("peak_equity", equity)
        self.ledger.event(date, "resumed", f"Auto-resumed after {n}+ days and a risk-on regime "
                                           "(backtest only; live needs your Resume).")
        return "running"

    def _queue_entries(self, date, cands, closes, budget, rows, regime=None, p_min=0.55) -> int:
        from autopilot.sizing import risk_fraction
        B, L, rk = self.broker, self.ledger, self.cfg["risk"]
        regime = regime or {}
        open_slots = (rk["max_positions"] + regime.get("extra_positions", 0) - len(B.positions)
                      - sum(1 for o in B.queue if o.side == "buy"))
        equity = B.equity(closes)
        pending_spend = sum((o.limit_price or 0) * o.qty for o in B.queue if o.side == "buy")
        sell_proceeds = sum(B.positions[o.symbol].qty * closes.get(o.symbol, 0)
                            for o in B.queue if o.side == "sell" and o.symbol in B.positions)
        cash = B.cash + sell_proceeds - pending_spend
        sector_exp = {}
        for s, p in B.positions.items():
            sec = self.universe[s].sector
            sector_exp[sec] = sector_exp.get(sec, 0) + p.qty * closes.get(s, p.avg_price)
        row_by_sym = {r["symbol"]: r for r in rows}
        n = 0
        for c in cands:
            if n >= budget or open_slots <= 0:
                break
            rf = risk_fraction(prob=c["prob"], p_min=p_min, regime=regime,
                               mkt_vol=regime.get("bench_vol"), cfg=self.cfg)
            qty, binding = size_entry(price=c["close"], atr=c["atr"], stop_atr=c["stop_atr"],
                                      equity=equity, cash_available=cash,
                                      sector_exposure=sector_exp.get(c["sector"], 0.0),
                                      instrument_type=c["type"], cfg=self.cfg, risk_frac=rf)
            if qty <= 0:
                row_by_sym[c["symbol"]].update(action="skip", why=f"sizing: {binding}")
                continue
            limit = round(c["close"] * (1 + self.cfg["strategy"]["max_gap_pct"]), 2)
            o = Order(id=B.new_id("O"), symbol=c["symbol"], side="buy", qty=qty, reason="entry",
                      created=date, limit_price=limit, stop_atr=c["stop_atr"], atr=c["atr"],
                      target_atr=self.cfg["strategy"]["target_atr"], prob=c["prob"])
            B.submit(o)
            L.order(o)
            L.event(date, "entry", f"Buy {qty} @ up to {inr(limit)}. {c['why']}. "
                                   f"Stop {c['stop_atr']} ATR ({inr(c['stop_atr'] * c['atr'])}/share), "
                                   f"risking {rf:.2%} of equity, size limited by {binding}.", c["symbol"],
                    {"prob": c["prob"], "ev_r": c["ev"], "stop_atr": c["stop_atr"], "atr": c["atr"],
                     "risk_frac": rf, "regime": regime.get("level")})
            row_by_sym[c["symbol"]].update(action="buy queued")
            cash -= qty * c["close"]
            sector_exp[c["sector"]] = sector_exp.get(c["sector"], 0) + qty * c["close"]
            open_slots -= 1
            n += 1
        return n

    def _trail_stops(self, date, closes, feats) -> None:
        s = self.cfg["strategy"]
        for sym, pos in self.broker.positions.items():
            c = closes.get(sym)
            if c is None:
                continue
            pos.highest_close = max(pos.highest_close, c)
            if pos.highest_close - pos.avg_price < s["trail_activate_atr"] * pos.atr_at_entry:
                continue
            atr_now = feats.at[sym, "atr"] if sym in feats.index else pos.atr_at_entry
            new_stop = pos.highest_close - s["trail_atr"] * atr_now
            gain_atr = (pos.highest_close - pos.avg_price) / pos.atr_at_entry
            if gain_atr >= s.get("lock_profit_trigger_atr", 99):     # never give a winner back
                new_stop = max(new_stop, pos.avg_price + s["lock_profit_atr"] * pos.atr_at_entry)
            if self.broker.modify_stop(sym, new_stop):
                self.ledger.event(date, "trail", f"Raised stop to {inr(new_stop)} "
                                                 f"(highest close {inr(pos.highest_close)}).", sym)

    def record_fills(self, date) -> None:
        """Write new fills/orders to the ledger and clear them (safe to call intraday)."""
        L, B = self.ledger, self.broker
        charges = L.get("charges_cum", 0.0)
        cooldown = L.get("cooldown", {})
        for f in B.fills_today:
            if f.side == "sell" and f.symbol not in B.positions:
                cooldown[f.symbol] = date
            L.fill(f)
            charges += f.charges.get("total", 0)
            if f.side == "sell":
                L.event(date, f.reason, f"Sold {f.qty} @ {inr(f.price)}; net P&L {inr(f.pnl)} "
                                        f"after {inr(f.charges['total'])} charges.", f.symbol,
                        {"pnl": f.pnl})
            else:
                pos = B.positions.get(f.symbol)
                g = B.gtts.get(pos.gtt_id) if pos and pos.gtt_id else None
                msg = f"Bought {f.qty} @ {inr(f.price)} (charges {inr(f.charges['total'])})."
                if g:
                    msg += f" Protected by stop {inr(g.stop_trigger)}, target {inr(g.target_trigger)}."
                L.event(date, "filled", msg, f.symbol)
        for o in B.processed_orders:
            L.order(o)
            if o.status in ("cancelled", "rejected") and o.side == "buy":
                L.event(date, "entry_skipped", f"Entry not filled: {o.note}.", o.symbol)
        L.set("charges_cum", charges)
        L.set("cooldown", cooldown)
        B.fills_today, B.processed_orders = [], []
