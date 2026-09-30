"""SQLite ledger: the single source of truth the dashboard/API reads."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS fills (
  id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, order_id TEXT, symbol TEXT,
  side TEXT, qty INTEGER, price REAL, value REAL, reason TEXT,
  charges_total REAL, charges_json TEXT, pnl REAL);
CREATE TABLE IF NOT EXISTS orders (
  id TEXT PRIMARY KEY, created TEXT, symbol TEXT, side TEXT, qty INTEGER,
  reason TEXT, status TEXT, note TEXT, prob REAL, limit_price REAL);
CREATE TABLE IF NOT EXISTS equity (
  date TEXT PRIMARY KEY, equity REAL, cash REAL, invested REAL,
  benchmark REAL, drawdown REAL, charges_cum REAL);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, kind TEXT, symbol TEXT,
  message TEXT, data_json TEXT);
CREATE TABLE IF NOT EXISTS signals (
  date TEXT, symbol TEXT, prob REAL, eligible INTEGER, action TEXT, why TEXT,
  PRIMARY KEY (date, symbol));
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS news (
  id TEXT PRIMARY KEY, seen_at TEXT, published TEXT, symbol TEXT, source TEXT, publisher TEXT,
  title TEXT, url TEXT, official INTEGER, event TEXT, direction TEXT, severity INTEGER,
  classifier TEXT, action TEXT);
CREATE TABLE IF NOT EXISTS research (
  month TEXT, symbol TEXT, sector TEXT, approved INTEGER, fundamental_score REAL,
  quant_score REAL, company_score REAL, sector_score REAL, pillars_json TEXT,
  filters_failed TEXT, data_coverage REAL, verdict TEXT, conviction REAL, sector_view TEXT,
  thesis TEXT, key_risks TEXT, red_flags TEXT, next_results_date TEXT, sources_json TEXT,
  reason TEXT, PRIMARY KEY (month, symbol));
"""


class Ledger:
    def __init__(self, path: str | Path = ":memory:"):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.commit()

    # --------------------------------------------------------------- writes
    def fill(self, f) -> None:
        self.db.execute(
            "INSERT INTO fills(date,order_id,symbol,side,qty,price,value,reason,charges_total,charges_json,pnl)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (f.date, f.order_id, f.symbol, f.side, f.qty, f.price, round(f.price * f.qty, 2),
             f.reason, f.charges.get("total", 0), json.dumps(f.charges), f.pnl))

    def order(self, o) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO orders VALUES (?,?,?,?,?,?,?,?,?,?)",
            (o.id, o.created, o.symbol, o.side, o.qty, o.reason, o.status, o.note, o.prob, o.limit_price))

    def equity_point(self, date, equity, cash, invested, benchmark, drawdown, charges_cum) -> None:
        self.db.execute("INSERT OR REPLACE INTO equity VALUES (?,?,?,?,?,?,?)",
                        (date, equity, cash, invested, benchmark, drawdown, charges_cum))

    def event(self, date, kind, message, symbol=None, data=None) -> None:
        self.db.execute("INSERT INTO events(date,kind,symbol,message,data_json) VALUES (?,?,?,?,?)",
                        (date, kind, symbol, message, json.dumps(data or {}, default=str)))

    def signals(self, date, rows: list[dict]) -> None:
        self.db.execute("DELETE FROM signals WHERE date=?", (date,))
        self.db.executemany("INSERT INTO signals VALUES (?,?,?,?,?,?)",
                            [(date, r["symbol"], r["prob"], int(r["eligible"]), r["action"], r["why"])
                             for r in rows])

    def set(self, key: str, value, commit: bool = False) -> None:
        self.db.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (key, json.dumps(value, default=str)))
        if commit:
            self.db.commit()

    def get(self, key: str, default=None):
        row = self.db.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def commit(self) -> None:
        self.db.commit()

    # ---------------------------------------------------------------- reads
    def rows(self, sql: str, args: tuple = ()) -> list[dict]:
        return [dict(r) for r in self.db.execute(sql, args).fetchall()]

    # control state (written by the dashboard, read by the engine)
    def control(self) -> str:
        return self.get("control", "running")

    def set_control(self, mode: str) -> None:
        self.set("control", mode, commit=True)
