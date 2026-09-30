"""Headline classification: keyword rules first; LLM only for ambiguous headlines on stocks
we hold or are about to buy. The LLM alone can never reach severity 5 (exit) - only rules
on an official filing can."""
from __future__ import annotations

import json
import re

# (regex, event, direction, severity)
RULES = [
    (r"\bfraud|forensic audit|accounting irregularit|whistle ?blower|siphon", "fraud / governance", "neg", 5),
    (r"auditors? (has |have )?resign|resignation of (the )?(statutory )?auditor", "auditor resignation", "neg", 5),
    (r"\b(ed|cbi|income tax) (raid|search|searches)|search operation|\barrest", "enforcement action", "neg", 5),
    (r"insolvency|nclt admits|defaults? on|missed (interest|debt|bond) payment", "default / insolvency", "neg", 5),
    (r"sebi (bans|bars|restrains|impounds)|trading suspended|licen[cs]e (cancelled|revoked)", "regulatory ban", "neg", 5),
    (r"pledged? shares? invoked|invocation of pledge", "pledge invoked", "neg", 5),
    (r"(rating|credit rating).{0,20}downgrad|downgrad.{0,20}(rating|to sell|to underperform)", "downgrade", "neg", 4),
    (r"(cuts?|lowers?|slashes?|trims?) (its |fy\d* )?(revenue |margin |earnings )?guidance|profit warning", "guidance cut", "neg", 4),
    (r"usfda.{0,30}(warning letter|import alert|observations?|form 483)", "regulatory observation", "neg", 4),
    (r"\b(ceo|cfo|md|managing director|chairman)\b.{0,20}(resigns|quits|steps down)", "management exit", "neg", 4),
    (r"promoters?.{0,25}(sell|offload|dilute)|stake sale by promoter", "promoter selling", "neg", 4),
    (r"fire (at|breaks out)|plant shut|production halt|strike at", "operational disruption", "neg", 4),
    (r"(tax|gst) demand|show[- ]cause notice|penalty", "tax / penalty", "neg", 3),
    (r"(net )?profit (falls|drops|declines|slumps|plunges|tanks)|revenue (falls|drops|declines)|misses (estimates|street)|weak (q\d|quarter|results)", "weak results", "neg", 3),
    (r"target (price )?(cut|lowered)|cut to (sell|reduce)", "target cut", "neg", 3),
    (r"(bags?|wins?|secures?|receives?) .{0,30}order|order (win|inflow)|contract worth", "order win", "pos", 3),
    (r"(net )?profit (jumps|rises|surges|soars|climbs)|record (profit|revenue)|beats (estimates|street)", "strong results", "pos", 3),
    (r"upgrad.{0,20}(to buy|rating)|target (price )?(raised|hiked)|initiates? .{0,15}buy", "upgrade", "pos", 3),
    (r"buyback|bonus issue", "buyback / bonus", "pos", 3),
    (r"usfda (approval|nod)|approval from usfda", "regulatory approval", "pos", 3),
]
COMPILED = [(re.compile(p, re.I), e, d, s) for p, e, d, s in RULES]

LLM_SYSTEM = """You classify one Indian stock-market headline for a risk system.
Return ONLY JSON: {"relevant": true|false, "direction": "neg"|"pos"|"neutral", "severity": 0-4,
"event": "short label", "confidence": 0-1}.
severity: 0 noise, 1-2 minor, 3 material, 4 serious (guidance cut, downgrade, regulator action, key exec exit).
relevant=false if the headline is not specifically about this company."""


def classify_rules(title: str) -> dict | None:
    for rx, event, direction, sev in COMPILED:
        if rx.search(title):
            return {"event": event, "direction": direction, "severity": sev, "classifier": "rules"}
    return None


def classify_llm(provider, symbol: str, title: str) -> dict | None:
    try:
        res = provider.complete(LLM_SYSTEM, f"Company: NSE:{symbol}\nHeadline: {title}")
        t = res.text
        data = json.loads(t[t.find("{"): t.rfind("}") + 1])
    except Exception:
        return None
    if not data.get("relevant") or float(data.get("confidence", 0)) < 0.7:
        return None
    sev = int(min(4, max(0, int(data.get("severity", 0)))))    # LLM capped at 4
    if data.get("direction") not in ("neg", "pos") or sev < 3:
        return None
    return {"event": str(data.get("event", "llm"))[:60], "direction": data["direction"],
            "severity": sev, "classifier": "llm"}
