"""Paper broker: simulates Zerodha delivery fills, charges and GTT OCO behaviour.

Two modes share the same state:
  * bar mode (backtest): process_open -> process_intraday -> end_day per daily bar
  * tick mode (live paper): process_tick with last traded prices

GTT realism (matches Kite terms):
  * a trigger places a LIMIT order; stop leg limit = trigger * (1 - buffer)
  * gap-down through the limit -> order may not fill that day -> cancelled at EOD
  * triggers are one-shot; an unfilled stop leaves the position unprotected and
    the engine exits it at the next open
"""
from __future__ import annotations

import itertools
import math
from dataclasses import asdict

from autopilot.broker.base import GTT, Fill, Order, Position
from autopilot.costs import delivery_charges


class PaperBroker:
    def __init__(self, cfg: dict, universe, cash: float):
        self.cfg = cfg
        self.universe = universe
        self.cash = float(cash)
        self.positions: dict[str, Position] = {}
        self.gtts: dict[str, GTT] = {}
        self.queue: list[Order] = []
        self.fills_today: list[Fill] = []
        self.processed_orders: list[Order] = []
        self.dp_charged_today: set[str] = set()
        self.today: str | None = None
        self._ids = itertools.count(1)
        self.slip = cfg["execution"]["slippage_pct"]
        self.stop_buffer = cfg["execution"]["stop_limit_buffer_pct"]

    # ----------------------------------------------------------------- state
    def new_id(self, prefix: str) -> str:
        return f"{prefix}{next(self._ids)}"

    def to_state(self) -> dict:
        return {
            "cash": self.cash,
            "positions": {s: asdict(p) for s, p in self.positions.items()},
            "gtts": {g: asdict(x) for g, x in self.gtts.items()},
            "queue": [asdict(o) for o in self.queue],
            "next_id": next(self._ids),
        }

    def load_state(self, st: dict) -> None:
        self.cash = st["cash"]
        self.positions = {s: Position(**p) for s, p in st["positions"].items()}
        self.gtts = {g: GTT(**x) for g, x in st["gtts"].items()}
        self.queue = [Order(**o) for o in st["queue"]]
        self._ids = itertools.count(st["next_id"])

    def equity(self, prices: dict[str, float] | None = None) -> float:
        val = 0.0
        for s, p in self.positions.items():
            px = (prices or {}).get(s) or p.last_price or p.avg_price
            val += p.qty * px
        return self.cash + val

    def invested(self) -> float:
        return sum(p.qty * (p.last_price or p.avg_price) for p in self.positions.values())

    # ------------------------------------------------------------ day cycle
    def start_day(self, date: str) -> None:
        self.today = date
        self.fills_today = []
        self.processed_orders = []
        self.dp_charged_today = set()

    def accrue_idle_cash(self, days: int = 1) -> float:
        rate = self.cfg["cash"]["idle_yield_annual"]
        interest = max(self.cash, 0.0) * rate * days / 365
        self.cash += interest
        return interest

    # ---------------------------------------------------------------- orders
    def submit(self, order: Order) -> None:
        self.queue.append(order)

    def cancel_queued(self, predicate=lambda o: True) -> list[Order]:
        cancelled = [o for o in self.queue if predicate(o)]
        for o in cancelled:
            o.status = "cancelled"
        self.queue = [o for o in self.queue if not predicate(o)]
        return cancelled

    def _charges(self, side: str, symbol: str, value: float) -> dict:
        charge_dp = side == "sell" and symbol not in self.dp_charged_today
        c = delivery_charges(side, value, self.universe[symbol].type, self.cfg["charges"], charge_dp)
        if side == "sell":
            self.dp_charged_today.add(symbol)
        return c.to_dict()

    def _fill_buy(self, order: Order, price: float) -> Fill | None:
        qty = order.qty
        # shrink to available cash (price * qty + charges)
        while qty > 0:
            ch = self._charges("buy", order.symbol, price * qty)
            if price * qty + ch["total"] <= self.cash:
                break
            qty -= max(1, qty // 20)
        if qty <= 0:
            order.status, order.note = "rejected", "insufficient cash"
            return None
        ch = self._charges("buy", order.symbol, price * qty)
        self.cash -= price * qty + ch["total"]
        stop_dist = order.stop_atr * order.atr
        pos = Position(symbol=order.symbol, qty=qty, avg_price=price, entry_date=self.today,
                       atr_at_entry=order.atr, stop_atr=order.stop_atr, highest_close=price,
                       prob_at_entry=order.prob, last_price=price, entry_charges=ch["total"])
        stop_trig = round(price - stop_dist, 2)
        tgt = round(price + order.target_atr * order.atr, 2)
        frac = self.cfg["strategy"].get("scale_out_fraction", 1.0)
        tgt_qty = max(1, int(qty * frac)) if qty > 1 else qty
        g = GTT(id=self.new_id("G"), symbol=order.symbol, qty=qty, stop_trigger=stop_trig,
                stop_limit=round(stop_trig * (1 - self.stop_buffer), 2),
                target_trigger=tgt, target_limit=tgt)
        self.gtts[g.id] = g
        pos.gtt_id = g.id
        self.positions[order.symbol] = pos
        order.status, order.qty = "filled", qty
        f = Fill(order.id, order.symbol, "buy", qty, round(price, 2), self.today, order.reason, ch)
        self.fills_today.append(f)
        return f

    def _fill_sell(self, symbol: str, qty: int, price: float, order_id: str, reason: str) -> Fill:
        pos = self.positions[symbol]
        ch = self._charges("sell", symbol, price * qty)
        self.cash += price * qty - ch["total"]
        entry_share = pos.entry_charges * qty / pos.qty
        pnl = (price - pos.avg_price) * qty - ch["total"] - entry_share
        pos.qty -= qty
        pos.entry_charges -= entry_share
        if pos.qty <= 0:
            if pos.gtt_id and pos.gtt_id in self.gtts:
                self.gtts[pos.gtt_id].status = "cancelled"
                del self.gtts[pos.gtt_id]
            del self.positions[symbol]
        f = Fill(order_id, symbol, "sell", qty, round(price, 2), self.today, reason, ch, round(pnl, 2))
        self.fills_today.append(f)
        return f

    def modify_stop(self, symbol: str, new_trigger: float) -> bool:
        """Kite equivalent: modify the GTT's stop leg (only ever raised)."""
        pos = self.positions.get(symbol)
        if not pos or not pos.gtt_id or pos.gtt_id not in self.gtts:
            return False
        g = self.gtts[pos.gtt_id]
        if new_trigger <= g.stop_trigger or g.status != "active":
            return False
        g.stop_trigger = round(new_trigger, 2)
        g.stop_limit = round(new_trigger * (1 - self.stop_buffer), 2)
        return True

    # ------------------------------------------------------------ bar mode
    def process_open(self, bars: dict[str, dict], check_gtts: bool = True) -> list[Fill]:
        """At the open: GTT gap triggers, then queued sells, then queued buys."""
        for g in (list(self.gtts.values()) if check_gtts else []):
            b = bars.get(g.symbol)
            if not b or g.status != "active":
                continue
            o = b["open"]
            if o <= g.stop_trigger:
                g.status, g.pending_leg = "triggered", "stop"
                if o >= g.stop_limit:     # limit sell marketable at the open
                    self._fill_gtt(g, o * (1 - self.slip), "stop")
            elif o >= g.target_trigger:
                g.status = "triggered"
                self._fill_gtt(g, o, "target")

        sells = [o for o in self.queue if o.side == "sell"]
        buys = [o for o in self.queue if o.side == "buy"]
        for o in sells:
            b = bars.get(o.symbol)
            if not b or o.symbol not in self.positions:
                o.status, o.note = "cancelled", "no bar or no position"
                continue
            qty = min(o.qty, self.positions[o.symbol].qty)
            self._cancel_gtt(o.symbol)
            self._fill_sell(o.symbol, qty, b["open"] * (1 - self.slip), o.id, o.reason)
            o.status = "filled"
        for o in buys:
            b = bars.get(o.symbol)
            if not b:
                o.status, o.note = "cancelled", "no bar"
                continue
            px = b["open"] * (1 + self.slip)
            if o.limit_price and px > o.limit_price:
                o.status, o.note = "cancelled", f"gapped above limit {o.limit_price:.2f}"
                continue
            if o.symbol in self.positions:
                o.status, o.note = "cancelled", "already held"
                continue
            self._fill_buy(o, px)
        self.processed_orders.extend(self.queue)
        self.queue = []
        return self.fills_today

    def process_intraday(self, bars: dict[str, dict]) -> None:
        for g in list(self.gtts.values()):
            b = bars.get(g.symbol)
            if not b:
                continue
            if g.status == "triggered" and g.pending_leg == "stop":
                if b["high"] >= g.stop_limit:           # limit sell eventually fills
                    self._fill_gtt(g, g.stop_limit, "stop")
                continue
            if g.status != "active":
                continue
            if b["low"] <= g.stop_trigger:              # stop assumed first if both hit
                g.status = "triggered"
                self._fill_gtt(g, max(g.stop_limit, g.stop_trigger * (1 - self.slip)), "stop")
            elif b["high"] >= g.target_trigger:
                g.status = "triggered"
                self._fill_gtt(g, g.target_limit, "target")

    def end_day(self, closes: dict[str, float]) -> list[str]:
        """Mark to market; unfilled triggered stops expire -> positions unprotected."""
        unprotected = []
        for g in list(self.gtts.values()):
            if g.status == "triggered" and g.pending_leg == "stop":
                del self.gtts[g.id]
                pos = self.positions.get(g.symbol)
                if pos:
                    pos.unprotected, pos.gtt_id = True, None
                    unprotected.append(g.symbol)
        for s, p in self.positions.items():
            if s in closes:
                p.last_price = closes[s]
        return unprotected

    # ----------------------------------------------------------- tick mode
    def process_tick(self, prices: dict[str, float]) -> None:
        """Live paper: evaluate GTT legs against last traded prices."""
        for g in list(self.gtts.values()):
            px = prices.get(g.symbol)
            if px is None:
                continue
            pos = self.positions.get(g.symbol)
            if pos:
                pos.last_price = px
            if g.status == "triggered" and g.pending_leg == "stop":
                if px >= g.stop_limit:
                    self._fill_gtt(g, g.stop_limit, "stop")
                continue
            if g.status != "active":
                continue
            if px <= g.stop_trigger:
                g.status, g.pending_leg = "triggered", "stop"
                if px >= g.stop_limit:
                    self._fill_gtt(g, px * (1 - self.slip), "stop")
            elif px >= g.target_trigger:
                g.status = "triggered"
                self._fill_gtt(g, g.target_limit, "target")

    def execute_queue_at(self, prices: dict[str, float]) -> None:
        """Live paper: fill queued orders at current prices (inside the execution window)."""
        bars = {s: {"open": p, "high": p, "low": p, "close": p} for s, p in prices.items()}
        waiting = [o for o in self.queue if o.symbol not in bars]
        self.queue = [o for o in self.queue if o.symbol in bars]
        self.process_open(bars, check_gtts=False)   # GTTs are handled by process_tick
        self.queue = waiting

    # -------------------------------------------------------------- helpers
    def _fill_gtt(self, g: GTT, price: float, leg: str) -> None:
        g.status, g.pending_leg = "triggered", None
        self.gtts.pop(g.id, None)
        if g.symbol not in self.positions:
            return
        pos = self.positions[g.symbol]
        if leg == "stop":
            reason = "trailing_stop" if (pos.highest_close > pos.avg_price and g.stop_trigger > pos.avg_price) else "stop_loss"
            self._fill_sell(g.symbol, min(g.qty, pos.qty), price, g.id, reason)
            return
        # target leg: scale out part of the position, keep the rest with a breakeven-or-better stop
        frac = self.cfg["strategy"].get("scale_out_fraction", 1.0)
        sell_qty = pos.qty if frac >= 1.0 or pos.qty <= 1 else max(1, int(pos.qty * frac))
        self._fill_sell(g.symbol, sell_qty, price, g.id, "target")
        if g.symbol in self.positions:                        # remainder rides the trend
            pos = self.positions[g.symbol]
            new_stop = round(max(g.stop_trigger, pos.avg_price + 0.5 * pos.atr_at_entry), 2)
            g2 = GTT(id=self.new_id("G"), symbol=g.symbol, qty=pos.qty, stop_trigger=new_stop,
                     stop_limit=round(new_stop * (1 - self.stop_buffer), 2),
                     target_trigger=1e12, target_limit=1e12)      # no second target: trailing only
            self.gtts[g2.id] = g2
            pos.gtt_id = g2.id

    def _cancel_gtt(self, symbol: str) -> None:
        pos = self.positions.get(symbol)
        if pos and pos.gtt_id:
            self.gtts.pop(pos.gtt_id, None)
            pos.gtt_id = None
