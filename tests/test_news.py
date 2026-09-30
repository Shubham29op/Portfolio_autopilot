from datetime import datetime, timedelta, timezone

import pandas as pd

from autopilot.broker.base import Order
from autopilot.broker.paper import PaperBroker
from autopilot.config import Universe, load_settings
from autopilot.ledger import Ledger
from autopilot.news.classify import classify_rules
from autopilot.news.guard import NewsGuard, active
from autopilot.news.sources import item_id, parse_rss

CFG = load_settings()
U = Universe.load()
NOW = datetime(2026, 10, 1, 11, 0, tzinfo=timezone.utc)

RSS = """<rss><channel>
<item><title>Infosys bags $500 million order from European bank - Moneycontrol</title>
<link>https://x/1</link><pubDate>Thu, 01 Oct 2026 09:00:00 GMT</pubDate><source url="m">Moneycontrol</source></item>
</channel></rss>"""


def test_parse_rss_strips_publisher():
    items = parse_rss(RSS, "INFY")
    assert items[0]["title"] == "Infosys bags $500 million order from European bank"
    assert items[0]["publisher"] == "Moneycontrol" and items[0]["published"].tzinfo


def test_rules():
    assert classify_rules("Statutory auditor resigns citing concerns")["severity"] == 5
    assert classify_rules("Company cuts FY27 revenue guidance")["event"] == "guidance cut"
    assert classify_rules("Q2 net profit jumps 30%")["direction"] == "pos"
    assert classify_rules("Board meeting on Oct 20") is None


class FakeSource:
    def __init__(self, items):
        self.items = items

    def fetch(self, symbol, name=""):
        return [i for i in self.items if i["symbol"] == symbol]


def item(sym, title, official=False, hours_ago=1):
    return {"id": item_id("t", sym, title, "u"), "symbol": sym, "source": "nse" if official else "google_news",
            "publisher": "NSE filing" if official else "ET", "title": title, "url": "u",
            "published": NOW - timedelta(hours=hours_ago), "official": official}


def setup(items):
    cfg = load_settings(overrides={"news": {"request_pause_s": 0}})
    L = Ledger(":memory:")
    B = PaperBroker(cfg, U, 1_000_000)
    B.start_day("2026-10-01")
    for s in ("INFY", "TCS", "WIPRO"):
        B.submit(Order(id=B.new_id("O"), symbol=s, side="buy", qty=50, reason="entry", created="x",
                       stop_atr=2, atr=2, target_atr=4))
    B.process_open({s: {"open": 100, "high": 100, "low": 100, "close": 100} for s in ("INFY", "TCS", "WIPRO")})
    B.submit(Order(id=B.new_id("O"), symbol="HCLTECH", side="buy", qty=10, reason="entry", created="x",
                   limit_price=103, stop_atr=2, atr=2, target_atr=4))
    L.set("research_current", {"approved": {"ITC": {}}, "version": "v"})
    g = NewsGuard(cfg, U, L, B, provider=None, sources=[FakeSource(items)])
    return g, L, B


def test_guard_actions():
    g, L, B = setup([
        item("INFY", "Auditor resigns, cites accounting irregularities", official=True),
        item("TCS", "Auditor resigns at TCS, report says"),                 # severe, unconfirmed
        item("WIPRO", "Wipro cuts revenue guidance for FY27"),              # moderate
        item("HCLTECH", "HCLTech net profit falls 12%"),                    # queued buy -> cancel
        item("ITC", "ITC bags large export order"),                         # positive
        item("ITC", "Promoter selling at ITC", hours_ago=72),               # too old
    ])
    stops = {s: B.gtts[B.positions[s].gtt_id].stop_trigger for s in ("TCS", "WIPRO")}
    acts = {a["symbol"] + ":" + a["title"][:10]: a["action"] for a in g.check(NOW)}
    assert any(o.symbol == "INFY" and o.reason == "news_exit" and o.protective for o in B.queue)
    assert B.gtts[B.positions["TCS"].gtt_id].stop_trigger > stops["TCS"]
    assert B.gtts[B.positions["WIPRO"].gtt_id].stop_trigger > stops["WIPRO"]
    assert not any(o.symbol == "HCLTECH" and o.side == "buy" for o in B.queue)
    assert "ITC" in L.get("news_boosts") and "WIPRO" in L.get("news_blocks")
    old = L.rows("SELECT action FROM news WHERE title LIKE 'Promoter selling%'")[0]["action"]
    assert old.startswith("none")
    assert g.check(NOW) == []                                               # deduplicated


def test_blocks_and_boosts_reach_entry_decision():
    from autopilot.strategy import entry_candidates
    feats = pd.DataFrame({"close": 100.0, "atr": 2.0, "dist_sma200": 0.1, "sma50_above_200": 1.0,
                          "rank_ret_126": 0.9, "rank_mom_risk_adj": 0.5, "rsi_14": 60.0}, index=["INFY", "ITC"])
    probs = pd.Series(0.55, index=["INFY", "ITC"])
    research = {"approved": {"INFY": {"fundamental_score": 60}, "ITC": {"fundamental_score": 60}},
                "sector_scores": {}}
    blocks = {"INFY": {"until": "2026-10-08", "why": "guidance cut"}}
    boosts = {"ITC": {"points": 5, "until": "2026-10-08", "why": "order win"}}
    c0, _ = entry_candidates(feats, probs, {"risk_on": True}, U, set(), {}, "2026-10-01", CFG, research)
    c1, rows = entry_candidates(feats, probs, {"risk_on": True}, U, set(), {}, "2026-10-01", CFG, research,
                                blocks, boosts)
    assert "INFY" not in [c["symbol"] for c in c1]
    # 56/100 without news is below the 60 bar; +5 good-news boost tips a qualified stock over,
    # but only because fundamentals, ML and trend already agreed
    assert "ITC" not in [c["symbol"] for c in c0]
    assert next(c for c in c1 if c["symbol"] == "ITC")["combined"] == 61.0
    assert active(blocks, "2026-10-09") == {}


def test_live_tick_runs_protective_orders_outside_window(tmp_path, monkeypatch):
    from autopilot.live import LivePaper
    cfg = load_settings(overrides={"data": {"provider": "synthetic"}, "news": {"enabled": False},
                                   "paths": {"db": str(tmp_path / "p.db"), "models": str(tmp_path / "m")}})
    lp = LivePaper(cfg, U)
    B = lp.broker
    B.start_day("2026-10-01")
    B.submit(Order(id="O1", symbol="INFY", side="buy", qty=10, reason="entry", created="x",
                   stop_atr=2, atr=2, target_atr=4))
    B.process_open({"INFY": {"open": 100, "high": 100, "low": 100, "close": 100}})
    B.submit(Order(id="O2", symbol="INFY", side="sell", qty=10, reason="news_exit", created="x", protective=True))
    B.submit(Order(id="O3", symbol="TCS", side="buy", qty=10, reason="entry", created="x",
                   limit_price=200, stop_atr=2, atr=2, target_atr=4))
    monkeypatch.setattr(lp, "in_exec_window", lambda now: False)
    monkeypatch.setattr(lp.provider, "last_prices", lambda syms: {"INFY": 101.0, "TCS": 150.0}, raising=False)
    lp.tick()
    assert "INFY" not in B.positions                       # protective exit filled at 09:xx
    assert any(o.symbol == "TCS" for o in B.queue)         # discretionary buy waits for 10:30
