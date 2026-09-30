"""Stage 1 orchestrator: fundamentals snapshot (Screener file or yfinance) -> quant score ->
LLM deep dive -> approved list."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from autopilot.config import resolve
from autopilot.research.deep_dive import DeepResearcher
from autopilot.research.importer import import_snapshot
from autopilot.research.scoring import score_snapshot, sector_momentum_from_features

log = logging.getLogger(__name__)
VIEW_ADJ = {"tailwind": 5.0, "neutral": 0.0, "headwind": -5.0}


def inbox_files(cfg: dict) -> list[Path]:
    inbox = resolve(cfg["research"]["inbox"])
    inbox.mkdir(parents=True, exist_ok=True)
    return sorted([p for p in inbox.iterdir() if p.suffix.lower() in (".csv", ".xlsx", ".xls")])


def load_snapshot(cfg: dict, universe, month: str, files: list[Path] | None = None,
                  source: str | None = None, bars=None, fetcher=None):
    """Screener file(s) if given/in the inbox (source auto|screener), else yfinance."""
    r = cfg["research"]
    source = source or r.get("source", "auto")
    files = files or (inbox_files(cfg) if source in ("auto", "screener") else [])
    if files:
        snap, summary = import_snapshot(files, universe, month, resolve(r["snapshots"]))
        summary["source"] = "screener"
        return snap, summary
    if source == "screener":
        raise FileNotFoundError(f"No Screener export found in {resolve(r['inbox'])}")
    from autopilot.research.yf_fundamentals import YFinanceFundamentals
    fetcher = fetcher or YFinanceFundamentals(**r.get("yfinance", {}))
    snap, summary = fetcher.snapshot(universe, month, resolve(r["snapshots"]), bars)
    if snap.empty:
        raise RuntimeError("yfinance returned no fundamentals (network or Yahoo issue)")
    return snap, summary


def run_monthly_research(cfg: dict, universe, ledger, month: str | None = None,
                         files: list[Path] | None = None, use_llm: bool | None = None,
                         feats_today: pd.DataFrame | None = None, held: set | None = None,
                         researcher: DeepResearcher | None = None, source: str | None = None,
                         bars=None, fetcher=None) -> dict:
    r = cfg["research"]
    month = month or datetime.now().strftime("%Y-%m")
    held = held or set()

    snap, import_summary = load_snapshot(cfg, universe, month, files, source, bars, fetcher)
    scores = score_snapshot(snap, cfg, sector_momentum_from_features(feats_today, universe))

    if use_llm is None:
        use_llm = researcher is not None or DeepResearcher.available(cfg)
    if use_llm and researcher is None:
        researcher = DeepResearcher(cfg)
    fallback = r.get("on_llm_failure", "numbers_only")

    eligible = scores[scores["passes_filters"] & (scores["quant_score"] >= r["min_quant_score"])]
    shortlist = list(eligible.head(r["deep_dive_top_n"]).index)
    review = shortlist + [s for s in held if s in scores.index and s not in shortlist]

    verdicts: dict[str, dict] = {}
    for i, sym in enumerate(review, 1):
        if use_llm:
            log.info("deep dive %d/%d: %s", i, len(review), sym)
            verdicts[sym] = researcher.research(sym, snap.at[sym, "sector"], snap.loc[sym].to_dict(), month)

    # sector view: majority LLM view per sector nudges every stock in that sector
    sector_adj = {}
    if verdicts:
        views = pd.DataFrame([(snap.at[s, "sector"], VIEW_ADJ[v.get("sector_view", "neutral")])
                              for s, v in verdicts.items() if v["verdict"] != "error"],
                             columns=["sector", "adj"])
        if not views.empty:
            sector_adj = views.groupby("sector")["adj"].mean().to_dict()

    blend = r["fundamental_blend"]
    approved, red_flag_syms, rows = {}, [], []
    for sym, row in scores.iterrows():
        v = verdicts.get(sym)
        adj = sector_adj.get(row["sector"], 0.0)
        if v and v["verdict"] != "error":
            fscore = blend["quant"] * row["quant_score"] + blend["llm_conviction"] * v["conviction"] + adj
        else:
            fscore = row["quant_score"] + adj
        fscore = round(max(0.0, min(100.0, fscore)), 1)

        if not row["passes_filters"]:
            ok, reason = False, "failed filters: " + "; ".join(row["filters_failed"])
        elif row["quant_score"] < r["min_quant_score"]:
            ok, reason = False, f"quant score {row['quant_score']} below {r['min_quant_score']}"
        elif sym not in shortlist:
            ok, reason = False, f"outside top {r['deep_dive_top_n']} by quant score"
        elif not use_llm:
            ok, reason = True, "numbers only (no LLM configured)"
        elif v["verdict"] == "error":
            if fallback == "numbers_only":
                ok, reason = True, f"approved on numbers; deep dive unavailable ({v['error']})"
            else:
                ok, reason = False, f"deep dive failed: {v['error']}"
        elif v["red_flags"]:
            ok, reason = False, "red flags: " + "; ".join(v["red_flags"])
        elif v["verdict"] != "approve":
            ok, reason = False, f"analyst verdict: {v['verdict']}"
        else:
            ok, reason = True, "approved"
        if v and v.get("red_flags"):
            red_flag_syms.append(sym)
        if ok:
            approved[sym] = {"fundamental_score": fscore, "next_results_date": (v or {}).get("next_results_date"),
                             "thesis": (v or {}).get("thesis", ""), "sector": row["sector"]}
        rows.append((month, sym, row["sector"], int(ok), fscore, row["quant_score"], row["company_score"],
                     row["sector_score"],
                     json.dumps({p: row[p] for p in ["growth", "quality", "balance_sheet", "valuation", "ownership"]}),
                     json.dumps(row["filters_failed"]), row["data_coverage"],
                     (v or {}).get("verdict"), (v or {}).get("conviction"), (v or {}).get("sector_view"),
                     (v or {}).get("thesis"), json.dumps((v or {}).get("key_risks", [])),
                     json.dumps((v or {}).get("red_flags", [])), (v or {}).get("next_results_date"),
                     json.dumps((v or {}).get("sources", [])), reason))

    done = sum(1 for v in verdicts.values() if v["verdict"] != "error")
    mode = "quant_only" if not use_llm or done == 0 else ("llm" if done == len(verdicts) else "mixed")
    ledger.db.execute("DELETE FROM research WHERE month=?", (month,))
    ledger.db.executemany("INSERT INTO research VALUES (" + ",".join("?" * 20) + ")", rows)
    sector_scores = scores.groupby("sector")["sector_score"].first().to_dict()
    current = {"month": month, "version": datetime.now().isoformat(timespec="seconds"), "mode": mode,
               "approved": approved, "red_flags": red_flag_syms,
               "sector_scores": {k: round(min(100, max(0, v + sector_adj.get(k, 0))), 1)
                                 for k, v in sector_scores.items()},
               "source": import_summary.get("source"),
               "import": {k: import_summary.get(k) for k in ("rows", "unmatched_names", "missing_from_export",
                                                             "coverage")}}
    ledger.set("research_current", current)
    ledger.event(datetime.now().date().isoformat(), "research",
                 f"Monthly research ({mode}, data: {import_summary.get('source')}): "
                 f"{len(approved)} of {len(scores)} stocks approved; "
                 f"{len(review)} deep-dived; {len(red_flag_syms)} with red flags.",
                 data={"approved": list(approved)})
    ledger.commit()
    return current
