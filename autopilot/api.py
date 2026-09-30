"""FastAPI backend for the React dashboard."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from autopilot.config import ROOT, load_settings, resolve
from autopilot.ledger import Ledger

cfg = load_settings()
SOURCES = {"paper": resolve(cfg["paths"]["db"]), "backtest": resolve("data/backtest.db")}
app = FastAPI(title="Portfolio Autopilot")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])


def ledger(source: str) -> Ledger:
    path = SOURCES.get(source)
    if path is None:
        raise HTTPException(400, "source must be paper or backtest")
    if not Path(path).exists():
        raise HTTPException(404, f"No {source} data yet. Run `python -m autopilot.cli "
                                 f"{'backtest' if source == 'backtest' else 'paper'}` first.")
    return Ledger(path)


@app.get("/api/sources")
def sources():
    return {k: Path(v).exists() for k, v in SOURCES.items()}


@app.get("/api/summary")
def summary(source: str = "paper"):
    L = ledger(source)
    eq = L.rows("SELECT * FROM equity ORDER BY date DESC LIMIT 2")
    st = L.get("broker_state") or {"positions": {}, "gtts": {}, "cash": cfg["capital"]}
    latest = eq[0] if eq else None
    prev = eq[1] if len(eq) > 1 else None
    positions = st["positions"]
    protected = sum(1 for p in positions.values() if p.get("gtt_id") in st["gtts"])
    first = L.rows("SELECT equity, benchmark FROM equity ORDER BY date LIMIT 1")
    return {
        "source": source,
        "mode": L.control() if source == "paper" else "backtest",
        "as_of": latest["date"] if latest else None,
        "equity": latest["equity"] if latest else st["cash"],
        "cash": latest["cash"] if latest else st["cash"],
        "invested": latest["invested"] if latest else 0,
        "day_change": round(latest["equity"] - prev["equity"], 2) if prev else 0,
        "total_return": round(latest["equity"] / first[0]["equity"] - 1, 4) if latest and first else 0,
        "benchmark_return": (round(latest["benchmark"] / first[0]["benchmark"] - 1, 4)
                             if latest and first and first[0]["benchmark"] else None),
        "drawdown": latest["drawdown"] if latest else 0,
        "charges_paid": latest["charges_cum"] if latest else 0,
        "positions": len(positions),
        "protected": protected,
        "queued": len(st.get("queue", [])),
        "regime": L.get("regime"),
        "model": L.get("model_meta"),
        "last_tick": L.get("last_tick"),
        "circuit_breaker_dd": cfg["risk"]["circuit_breaker_dd"],
        "p_enter": cfg["strategy"]["p_enter"],
        "max_positions": cfg["risk"]["max_positions"],
        "report": L.get("report") if source == "backtest" else None,
        "research": _research_brief(L.get("research_current")),
    }


def _research_brief(cur):
    if not cur:
        return None
    return {"month": cur["month"], "mode": cur["mode"], "approved": len(cur["approved"]),
            "red_flags": cur["red_flags"], "version": cur["version"], "source": cur.get("source")}


@app.get("/api/news")
def news(source: str = "paper", limit: int = Query(60, le=300), actionable: bool = False):
    L = ledger(source)
    where = "WHERE action NOT LIKE 'none%'" if actionable else ""
    rows = L.rows(f"SELECT * FROM news {where} ORDER BY seen_at DESC, published DESC LIMIT ?", (limit,))
    return {"rows": rows, "blocks": L.get("news_blocks") or {}, "boosts": L.get("news_boosts") or {},
            "llm": cfg["llm"]["provider"] if cfg["llm"].get("enabled") else None}


@app.get("/api/research")
def research(source: str = "paper"):
    L = ledger(source)
    cur = L.get("research_current")
    if not cur:
        return {"month": None, "rows": []}
    rows = L.rows("SELECT * FROM research WHERE month=? ORDER BY approved DESC, fundamental_score DESC",
                  (cur["month"],))
    for r in rows:
        for k in ("pillars_json", "filters_failed", "key_risks", "red_flags", "sources_json"):
            r[k] = json.loads(r[k]) if r[k] else None
    return {"month": cur["month"], "mode": cur["mode"], "source": cur.get("source"),
            "coverage": (cur.get("import") or {}).get("coverage"),
            "sector_scores": cur["sector_scores"], "forecast": cur.get("forecast") or {}, "rows": rows}


@app.get("/api/equity")
def equity(source: str = "paper"):
    return ledger(source).rows("SELECT date, equity, benchmark, drawdown FROM equity ORDER BY date")


@app.get("/api/positions")
def positions(source: str = "paper"):
    L = ledger(source)
    st = L.get("broker_state") or {"positions": {}, "gtts": {}}
    out = []
    for sym, p in st["positions"].items():
        g = st["gtts"].get(p.get("gtt_id") or "", {})
        last = p.get("last_price") or p["avg_price"]
        out.append({**p, "stop": g.get("stop_trigger"), "target": g.get("target_trigger"),
                    "value": round(last * p["qty"], 2),
                    "pnl": round((last - p["avg_price"]) * p["qty"], 2),
                    "pnl_pct": round(last / p["avg_price"] - 1, 4)})
    return sorted(out, key=lambda r: -r["value"])


@app.get("/api/events")
def events(source: str = "paper", limit: int = Query(60, le=500), symbol: str | None = None):
    L = ledger(source)
    if symbol:
        return L.rows("SELECT * FROM events WHERE symbol=? ORDER BY id DESC LIMIT ?", (symbol, limit))
    return L.rows("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))


@app.get("/api/fills")
def fills(source: str = "paper", limit: int = Query(100, le=2000)):
    return ledger(source).rows("SELECT * FROM fills ORDER BY id DESC LIMIT ?", (limit,))


@app.get("/api/signals")
def signals(source: str = "paper"):
    L = ledger(source)
    d = L.rows("SELECT MAX(date) AS d FROM signals")[0]["d"]
    return {"date": d, "rows": L.rows("SELECT * FROM signals WHERE date=? ORDER BY prob DESC", (d,))}


class ControlIn(BaseModel):
    action: str   # pause | resume | exit_all | stop


@app.post("/api/control")
def control(body: ControlIn):
    L = ledger("paper")
    mode = L.control()
    if body.action == "pause":
        L.set_control("paused")
    elif body.action == "resume":
        if mode == "halted":   # fresh peak so the breaker doesn't re-fire instantly
            L.set("peak_equity", L.get("last_equity"), commit=True)
        L.set_control("running")
    elif body.action == "exit_all":
        L.set_control("exit_all")    # engine exits everything at next EOD, then stops
    elif body.action == "stop":
        L.set_control("stopped")
    else:
        raise HTTPException(400, "unknown action")
    L.event(L.get("last_date") or "", "control", f"You set autopilot to {L.control()}.")
    L.commit()
    return {"mode": L.control()}


dist = ROOT / "dashboard" / "dist"
if dist.exists():
    app.mount("/", StaticFiles(directory=dist, html=True), name="dashboard")
