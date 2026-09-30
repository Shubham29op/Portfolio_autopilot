"""NewsGuard: turns classified headlines into risk actions. News can only REDUCE risk
(exit, tighten, cancel, block) plus a small, decaying score boost for good news that
never creates a buy on its own."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

from autopilot.broker.base import Order
from autopilot.fmt import inr
from autopilot.news.classify import classify_llm, classify_rules
from autopilot.news.sources import GoogleNews, NSEAnnouncements

log = logging.getLogger(__name__)


class NewsGuard:
    def __init__(self, cfg: dict, universe, ledger, broker, provider=None, sources=None):
        self.cfg, self.n = cfg, cfg["news"]
        self.universe, self.ledger, self.broker, self.provider = universe, ledger, broker, provider
        if sources is None:
            sources = []
            if self.n["sources"].get("nse_announcements"):
                sources.append(NSEAnnouncements())
            if self.n["sources"].get("google_news"):
                sources.append(GoogleNews())
        self.sources = sources

    # ------------------------------------------------------------ watchlist
    def watchlist(self) -> tuple[list[str], set[str], set[str]]:
        held = set(self.broker.positions)
        queued = {o.symbol for o in self.broker.queue if o.side == "buy"}
        research = self.ledger.get("research_current") or {}
        approved = set(research.get("approved", {}))
        syms = [s for s in sorted(held | queued | approved)
                if s in self.universe.by_symbol and self.universe[s].type == "stock"]
        return syms, held, queued

    def _llm_budget_left(self, today: str) -> int:
        used = self.ledger.get("news_llm_used", {})
        return self.n["llm_calls_per_day"] - used.get(today, 0)

    def _spend_llm(self, today: str) -> None:
        used = self.ledger.get("news_llm_used", {})
        self.ledger.set("news_llm_used", {today: used.get(today, 0) + 1})

    # ---------------------------------------------------------------- check
    def check(self, now: datetime) -> list[dict]:
        today = now.date().isoformat()
        now_utc = now.astimezone(timezone.utc)
        syms, held, queued = self.watchlist()
        seen = {r["id"] for r in self.ledger.rows("SELECT id FROM news WHERE seen_at >= ?",
                                                  ((now_utc - timedelta(days=4)).isoformat(),))}
        fresh = []
        for sym in syms:
            name = self.universe[sym].name
            for src in self.sources:
                for it in src.fetch(sym, name):
                    if it["id"] not in seen:
                        seen.add(it["id"])
                        fresh.append(it)
                time.sleep(self.n.get("request_pause_s", 0.3))
        actions = []
        # official filings first so they can confirm Google headlines in the same batch
        fresh.sort(key=lambda i: not i["official"])
        for it in fresh:
            actions.append(self._handle(it, now, today, held, queued))
        self.ledger.commit()
        return [a for a in actions if a["action"] != "none"]

    def _handle(self, it: dict, now, today, held, queued) -> dict:
        sym = it["symbol"]
        age_h = (now.astimezone(timezone.utc) - it["published"]).total_seconds() / 3600 \
            if it["published"] else None
        cls = classify_rules(it["title"])
        if cls is None and (sym in held or sym in queued) and self.provider is not None \
                and self._llm_budget_left(today) > 0 and (age_h is None or age_h <= self.n["max_age_hours"]):
            self._spend_llm(today)
            cls = classify_llm(self.provider, sym, it["title"])
        action = "none"
        if cls is None:
            action = "none"
        elif age_h is not None and age_h > self.n["max_age_hours"]:
            action = "none (too old)"
        else:
            action = self._act(sym, cls, it, today, held, queued)
        self.ledger.db.execute(
            "INSERT OR IGNORE INTO news VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (it["id"], now.astimezone(timezone.utc).isoformat(timespec="seconds"),
             it["published"].isoformat() if it["published"] else None, sym, it["source"], it["publisher"],
             it["title"], it["url"], int(it["official"]), (cls or {}).get("event"),
             (cls or {}).get("direction"), (cls or {}).get("severity"), (cls or {}).get("classifier"), action))
        return {"symbol": sym, "title": it["title"], "action": action}

    def _confirmed(self, sym: str, it: dict) -> bool:
        if it["official"]:
            return True
        rows = self.ledger.rows("SELECT 1 FROM news WHERE symbol=? AND official=1 AND severity>=5 "
                                "AND direction='neg' AND seen_at >= ?",
                                (sym, (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()))
        return bool(rows)

    def _act(self, sym, cls, it, today, held, queued) -> str:
        a, L = self.n["actions"], self.ledger
        until = (datetime.fromisoformat(today) + timedelta(days=a["block_entry_days"] * 7 // 5)).date().isoformat()
        label = f"{cls['event']} ({'filing' if it['official'] else it['publisher'] or it['source']})"
        if cls["direction"] == "pos":
            boosts = L.get("news_boosts", {})
            boosts[sym] = {"points": a["positive_boost_points"], "until": until, "why": label}
            L.set("news_boosts", boosts)
            L.event(today, "news_positive", f"Good news: {it['title']}. Score +{a['positive_boost_points']} "
                                            f"until {until} (never a buy on its own).", sym)
            return f"boost +{a['positive_boost_points']}"

        # negative
        done = []
        if cls["severity"] >= 3:
            blocks = L.get("news_blocks", {})
            blocks[sym] = {"until": until, "why": label}
            L.set("news_blocks", blocks)
            done.append(f"entries blocked until {until}")
        if sym in queued and cls["severity"] >= a["cancel_queued_buy_min_severity"]:
            self.broker.cancel_queued(lambda o: o.symbol == sym and o.side == "buy")
            done.append("queued buy cancelled")
        if sym in held:
            pos = self.broker.positions[sym]
            px = pos.last_price or pos.avg_price
            if cls["severity"] >= 5 and self._confirmed(sym, it) and a["severe_confirmed"] == "exit":
                if not any(o.symbol == sym and o.side == "sell" for o in self.broker.queue):
                    o = Order(id=self.broker.new_id("O"), symbol=sym, side="sell", qty=pos.qty,
                              reason="news_exit", created=today, protective=True)
                    self.broker.submit(o)
                    L.order(o)
                done.append("exit at next price check")
            else:
                k = a["severe_unconfirmed_stop_atr"] if cls["severity"] >= 5 else a["moderate_stop_atr"]
                new = px - k * pos.atr_at_entry
                if self.broker.modify_stop(sym, new):
                    done.append(f"stop tightened to {inr(new)}")
        msg = f"{'Severe' if cls['severity'] >= 5 else 'Negative'} news: {it['title']}. " + "; ".join(done) + "."
        L.event(today, "news_negative", msg, sym, {"url": it["url"], "severity": cls["severity"]})
        return "; ".join(done) or "logged"


def active(entries: dict, today: str) -> dict:
    """Drop expired blocks/boosts."""
    return {s: v for s, v in (entries or {}).items() if v["until"] >= today}
