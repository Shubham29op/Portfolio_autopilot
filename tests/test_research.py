import json
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

from autopilot.config import Universe, load_settings
from autopilot.ledger import Ledger
from autopilot.research.deep_dive import DeepResearcher, parse_verdict
from autopilot.research.importer import import_snapshot
from autopilot.research.monthly import run_monthly_research
from autopilot.research.scoring import score_snapshot
from autopilot.strategy import entry_candidates

U = Universe.load()
STOCKS = [s for s, i in U.by_symbol.items() if i.type == "stock"]


def screener_csv(tmp_path, overrides=None):
    rng = np.random.default_rng(1)
    rows = []
    for s in STOCKS:
        pat = rng.uniform(500, 4000)
        rows.append({
            "Name": s.title(), "NSE Code": s, "CMP Rs.": rng.uniform(100, 3000),
            "Sales Var 3Yrs %": rng.uniform(-5, 30), "Profit Var 3Yrs %": rng.uniform(0, 35),
            "Qtr Sales Var %": rng.uniform(-10, 30), "Qtr Profit Var %": rng.uniform(-20, 40),
            "ROCE %": rng.uniform(8, 35), "ROE 3Yr %": rng.uniform(8, 30), "OPM %": rng.uniform(8, 30),
            "OPM Last Year %": rng.uniform(8, 30), "Debt / Eq": rng.uniform(0, 1.2),
            "Int Coverage": rng.uniform(4, 40), "CF Opr Last Yr Rs.Cr.": pat * rng.uniform(0.8, 1.4),
            "PAT 12M Rs.Cr.": pat, "P/E": rng.uniform(12, 60),
            "Hist PE 5Yrs": rng.uniform(15, 50), "Ind PE": rng.uniform(15, 45),
            "Pledged %": 0.0, "Chg in Prom Hold %": rng.uniform(-1, 1),
            "Chg in FII Hold %": rng.uniform(-2, 2), "Chg in DII Hold %": rng.uniform(-2, 2),
        })
    df = pd.DataFrame(rows).set_index("NSE Code", drop=False)
    for sym, col, val in (overrides or []):
        df.loc[sym, col] = val
    path = tmp_path / "screener.csv"
    df.to_csv(path, index=False)
    return path


def test_import_matches_symbols_and_derives_metrics(tmp_path):
    df, summary = import_snapshot([screener_csv(tmp_path)], U, "2026-10", tmp_path / "snap")
    assert len(df) == len(STOCKS) and not summary["missing_from_export"]
    assert df["cfo_to_pat"].notna().all()
    assert (tmp_path / "snap" / "2026-10.csv").exists()


def test_filters_and_financial_exemption(tmp_path):
    cfg = load_settings()
    path = screener_csv(tmp_path, [("TATASTEEL", "Debt / Eq", 3.0), ("ITC", "Pledged %", 25.0),
                                   ("HDFCBANK", "Debt / Eq", 8.0)])
    df, _ = import_snapshot([path], U, "2026-10", tmp_path / "snap")
    sc = score_snapshot(df, cfg)
    assert not sc.at["TATASTEEL", "passes_filters"]
    assert not sc.at["ITC", "passes_filters"]
    assert sc.at["HDFCBANK", "passes_filters"]      # bank leverage is not a red flag
    assert sc["quant_score"].between(0, 100).all()


def test_parse_verdict_handles_fences_and_clamps():
    v = parse_verdict('```json\n{"verdict":"approve","conviction":140,"sector_view":"x",'
                      '"next_results_date":"2026-10-20","red_flags":[]}\n```')
    assert v["conviction"] == 100 and v["sector_view"] == "neutral"
    with pytest.raises(ValueError):
        parse_verdict('{"verdict":"maybe"}')


class FakeClient:
    """Mimics anthropic client.messages.create, including one pause_turn."""

    def __init__(self, verdicts, fail=()):
        self.verdicts, self.fail, self.calls = verdicts, set(fail), 0
        self.messages = self

    def create(self, model, max_tokens, system, messages, tools=None):
        self.calls += 1
        sym = messages[0]["content"].split("NSE:")[1].split(" ")[0]
        if sym in self.fail:
            raise RuntimeError("rate limited")
        if len(messages) == 1 and sym == "INFY":     # first call pauses
            return NS(stop_reason="pause_turn", content=[NS(type="server_tool_use")])
        text = json.dumps(self.verdicts.get(sym, {"verdict": "approve", "conviction": 70,
                                                  "sector_view": "tailwind", "red_flags": [],
                                                  "next_results_date": "2026-10-20",
                                                  "sources": [{"title": "t", "url": "https://x"}]}))
        search = NS(type="web_search_tool_result", content=[NS(url=f"https://news/{sym}", title="t")])
        return NS(stop_reason="end_turn", content=[search, NS(type="text", text=text)])


def test_monthly_research_end_to_end(tmp_path):
    cfg = load_settings(overrides={"research": {"min_quant_score": 0, "deep_dive_top_n": 100,
                                                "snapshots": str(tmp_path / "snap")}})
    L = Ledger(tmp_path / "p.db")
    client = FakeClient({"TCS": {"verdict": "approve", "conviction": 80, "red_flags": ["auditor resigned 2026-09-01"]},
                         "WIPRO": {"verdict": "reject", "conviction": 20, "red_flags": []}},
                        fail={"LT"})
    from autopilot.llm import AnthropicProvider
    researcher = DeepResearcher(cfg, provider=AnthropicProvider(cfg["llm"]["anthropic"], client=client))
    cur = run_monthly_research(cfg, U, L, month="2026-10", files=[screener_csv(tmp_path)],
                               use_llm=True, researcher=researcher, held={"SBIN"})
    rows = {r["symbol"]: r for r in L.rows("SELECT * FROM research WHERE month='2026-10'")}
    assert "TCS" not in cur["approved"] and "TCS" in cur["red_flags"]
    assert rows["WIPRO"]["verdict"] == "reject" and not rows["WIPRO"]["approved"]
    # LLM failure -> numbers-only fallback (config research.on_llm_failure)
    assert rows["LT"]["verdict"] == "error" and rows["LT"]["approved"]
    assert "approved on numbers" in rows["LT"]["reason"]
    assert cur["mode"] == "mixed"
    assert rows["INFY"]["verdict"] == "approve"          # survived a pause_turn continuation
    assert rows["SBIN"]["verdict"] is not None          # held stocks always reviewed
    assert cur["approved"] and all(0 <= a["fundamental_score"] <= 100 for a in cur["approved"].values())


def _feats(symbols):
    return pd.DataFrame({"close": 100.0, "atr": 2.0, "dist_sma200": 0.1, "sma50_above_200": 1.0,
                         "rank_ret_126": 0.9, "rank_mom_risk_adj": 0.9, "rsi_14": 60.0}, index=symbols)


def test_entry_gate_uses_research():
    cfg = load_settings()
    syms = ["INFY", "TCS", "ITBEES", "NIFTYBEES"]
    probs = pd.Series(0.7, index=syms)
    research = {"approved": {"INFY": {"fundamental_score": 80, "next_results_date": None},
                             "TCS": {"fundamental_score": 85, "next_results_date": "2026-10-05"}},
                "sector_scores": {"Information Technology": 70}}
    cands, rows = entry_candidates(_feats(syms), probs, {"risk_on": True}, U, set(), {}, "2026-10-01",
                                   cfg, research)
    why = {r["symbol"]: r["why"] for r in rows}
    picked = [c["symbol"] for c in cands]
    assert "INFY" in picked and "ITBEES" in picked and "NIFTYBEES" in picked
    assert "TCS" not in picked and "results due" in why["TCS"]
    missing = {"approved": {}, "sector_scores": {}, "missing": True}
    cands2, rows2 = entry_candidates(_feats(syms), probs, {"risk_on": True}, U, set(), {}, "2026-10-01",
                                     cfg, missing)
    assert {c["symbol"] for c in cands2} == {"ITBEES", "NIFTYBEES"}


def test_engine_review_tightens_dropped_and_exits_red_flags():
    from autopilot.broker.base import Order
    from autopilot.broker.paper import PaperBroker
    from autopilot.engine import Engine
    cfg = load_settings()
    L = Ledger(":memory:")
    B = PaperBroker(cfg, U, 1_000_000)
    B.start_day("2026-09-01")
    for sym in ("INFY", "TCS"):
        B.submit(Order(id=B.new_id("O"), symbol=sym, side="buy", qty=50, reason="entry", created="x",
                       stop_atr=2, atr=2, target_atr=4))
    B.process_open({s: {"open": 100, "high": 100, "low": 100, "close": 100} for s in ("INFY", "TCS")})
    L.set("research_current", {"version": "v1", "approved": {}, "red_flags": ["TCS"], "sector_scores": {}})
    E = Engine(cfg, U, B, L, use_research=True)
    B.start_day("2026-10-01")
    old_stop = B.gtts[B.positions["INFY"].gtt_id].stop_trigger
    E.close_day("2026-10-01", {"INFY": 110, "TCS": 110}, _feats(["INFY", "TCS"]), pd.Series(0.3, index=["INFY", "TCS"]))
    assert B.gtts[B.positions["INFY"].gtt_id].stop_trigger > old_stop     # tightened
    assert any(o.symbol == "TCS" and o.reason == "red_flag" for o in B.queue)


class FakeHTTP:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), 0

    def post(self, url, json, headers):
        self.calls += 1
        code, body = self.responses.pop(0)
        req = __import__("httpx").Request("POST", url)
        return __import__("httpx").Response(code, json=body, request=req)


def _gemini_body(text, grounded=True):
    cand = {"content": {"parts": [{"text": text}]}}
    if grounded:
        cand["groundingMetadata"] = {"webSearchQueries": ["q"],
                                     "groundingChunks": [{"web": {"uri": "https://src", "title": "moneycontrol.com"}}]}
    return {"candidates": [cand]}


def test_gemini_grounded_verdict_and_fallbacks():
    from autopilot.llm import GeminiProvider
    cfg = load_settings()
    gcfg = {**cfg["llm"]["gemini"], "min_seconds_between_calls": 0}
    ok = json.dumps({"verdict": "approve", "conviction": 66, "red_flags": []})
    http = FakeHTTP([(200, _gemini_body(ok)), (200, _gemini_body(ok, grounded=False)),
                     (429, {"error": {"message": "Quota exceeded for requests per day"}})])
    r = DeepResearcher(cfg, provider=GeminiProvider(gcfg, api_key="k", http=http))
    v1 = r.research("INFY", "IT", {}, "2026-10")
    assert v1["verdict"] == "approve" and v1["sources"][0]["url"] == "https://src"
    v2 = r.research("TCS", "IT", {}, "2026-10")
    assert v2["unavailable"] and "without web search" in v2["error"]
    v3 = r.research("WIPRO", "IT", {}, "2026-10")
    assert v3["unavailable"] and r.exhausted
    v4 = r.research("HCLTECH", "IT", {}, "2026-10")     # no more calls once exhausted
    assert v4["unavailable"] and http.calls == 3
