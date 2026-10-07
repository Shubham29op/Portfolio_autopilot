"""Claude-powered deep dive: reads latest results, concall commentary and news via web search,
returns a structured verdict. The LLM judges; numbers come from the Screener snapshot.

Fails closed: any error -> verdict "error" -> the stock is NOT approved this month.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import date

log = logging.getLogger(__name__)

SYSTEM = """You are a sell-side equity research analyst covering Indian listed companies (NSE).
You are reviewing ONE company for a systematic fund that holds positions for 1-12 months.

Use web search to find, in this order:
1. The latest quarterly results (press release or exchange filing) and what changed vs last year.
2. The latest earnings call / concall transcript or investor presentation: guidance, demand commentary, margins, capex.
3. News from the last 90 days: regulatory actions, management or auditor changes, pledges, block deals, litigation, rating actions.
4. Sector conditions: demand cycle, policy (e.g. PLI, duties), input costs, rate sensitivity.
5. The next results / board meeting date if announced.

Rules:
- Base judgements on sources you actually found. Never invent numbers; use null when unknown.
- The quantitative snapshot you are given is from Screener.in; use it, don't recompute it.
- red_flags must be concrete and sourced (e.g. "auditor resigned 2026-08-12"), not generic risks.
- verdict: "approve" = fundamentals support owning it for 1-12 months; "watch" = mixed; "reject" = avoid.
- Reply with ONLY a JSON object, no prose, matching this schema:
{"verdict": "approve|watch|reject", "conviction": 0-100, "sector_view": "tailwind|neutral|headwind",
 "thesis": "2-3 sentences", "key_risks": ["..."], "red_flags": ["..."],
 "latest_quarter": {"period": "e.g. Q1 FY27", "revenue_yoy_pct": number|null, "profit_yoy_pct": number|null,
                    "summary": "1 sentence"},
 "guidance": "1 sentence or null", "next_results_date": "YYYY-MM-DD or null",
 "sources": [{"title": "...", "url": "..."}]}"""


def _metric_lines(metrics: dict) -> str:
    keep = ["price", "market_cap", "sales_growth_3y", "profit_growth_3y", "qtr_sales_yoy", "qtr_profit_yoy",
            "roce", "roe_3y_avg", "opm", "opm_last_year", "debt_to_equity", "interest_coverage",
            "cfo_to_pat", "pe", "pe_5y_median", "industry_pe", "promoter_holding", "promoter_change",
            "pledged_pct", "fii_change", "dii_change",
            "fc_rev_growth", "fc_eps_growth_1y", "fc_eps_growth_3y", "fc_fcf_yield", "fc_upside"]
    lines = []
    for k in keep:
        v = metrics.get(k)
        if v is not None and v == v:  # not NaN
            lines.append(f"- {k}: {round(v, 2) if isinstance(v, float) else v}")
    return "\n".join(lines)


def parse_verdict(text: str) -> dict:
    """Extract and validate the JSON verdict from the model's final text."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("no JSON object in response")
    data = json.loads(cleaned[start:end + 1])
    if data.get("verdict") not in ("approve", "watch", "reject"):
        raise ValueError(f"bad verdict {data.get('verdict')!r}")
    data["conviction"] = float(max(0, min(100, float(data.get("conviction") or 0))))
    if data.get("sector_view") not in ("tailwind", "neutral", "headwind"):
        data["sector_view"] = "neutral"
    data["red_flags"] = [str(x) for x in (data.get("red_flags") or []) if str(x).strip()]
    data["key_risks"] = [str(x) for x in (data.get("key_risks") or [])]
    nrd = data.get("next_results_date")
    try:
        data["next_results_date"] = date.fromisoformat(nrd).isoformat() if nrd else None
    except ValueError:
        data["next_results_date"] = None
    data["sources"] = [s for s in (data.get("sources") or []) if isinstance(s, dict) and s.get("url")]
    return data


class DeepResearcher:
    """Runs one grounded deep dive per company through a pluggable LLM provider."""

    def __init__(self, cfg: dict, provider=None):
        from autopilot.llm import make_provider
        self.provider = provider or make_provider(cfg["llm"])
        self.exhausted = False

    @staticmethod
    def available(cfg: dict) -> bool:
        from autopilot.llm import make_provider
        return make_provider(cfg["llm"]) is not None

    def research(self, symbol: str, sector: str, metrics: dict, month: str) -> dict:
        from autopilot.llm import QuotaExhausted

        if self.provider is None or self.exhausted:
            return _unavailable("LLM quota exhausted earlier in this run" if self.exhausted else "no LLM")
        user = (f"Company: NSE:{symbol} ({metrics.get('name') or symbol}), sector: {sector}.\n"
                f"Review month: {month}. Today is {date.today().isoformat()}.\n\n"
                f"Screener.in snapshot:\n{_metric_lines(metrics)}\n\nReturn the JSON verdict.")
        try:
            res = self.provider.search(SYSTEM, user)
            if not res.searched:
                return _unavailable("model answered without web search (no grounding on this key/model)",
                                    res.sources)
            verdict = parse_verdict(res.text)
            # grounded sources from the provider are the trustworthy list
            verdict["sources"] = (res.sources or verdict["sources"])[:10]
            verdict["error"], verdict["unavailable"] = None, False
            return verdict
        except QuotaExhausted as exc:
            self.exhausted = True
            log.warning("LLM quota exhausted at %s: %s", symbol, exc)
            return _unavailable(f"quota exhausted: {str(exc)[:120]}")
        except Exception as exc:
            log.warning("deep dive failed for %s: %s", symbol, exc)
            return _unavailable(str(exc)[:300])


def _unavailable(reason: str, sources=None) -> dict:
    return {"verdict": "error", "conviction": 0.0, "sector_view": "neutral", "thesis": "",
            "key_risks": [], "red_flags": [], "next_results_date": None,
            "sources": (sources or [])[:8], "error": reason, "unavailable": True}
