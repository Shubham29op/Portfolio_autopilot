import pandas as pd
import pytest
from fastapi.testclient import TestClient

from autopilot.backtest import run_backtest
from autopilot.config import Universe, load_settings
from autopilot.pipeline import prepare


@pytest.fixture(scope="module")
def bt(tmp_path_factory):
    cfg = load_settings(overrides={"data": {"provider": "synthetic"},
                                   "model": {"retrain_every_days": 252}})
    u = Universe.load()
    bars, feats, labels = prepare(cfg, u)
    db = tmp_path_factory.mktemp("bt") / "bt.db"
    report = run_backtest(cfg, u, bars, feats, labels, db, start="2023-01-01")
    return cfg, db, report


def test_report_sane(bt):
    cfg, db, rep = bt
    assert rep["strategy"]["cagr"] is not None
    assert rep["trades"]["total_charges"] > 0


def test_discretionary_orders_capped_per_day(bt):
    cfg, db, _ = bt
    from autopilot.ledger import Ledger
    L = Ledger(db)
    rows = L.rows("SELECT created, COUNT(*) n FROM orders WHERE reason IN "
                  "('entry','model_exit','time_stop','max_hold') GROUP BY created")
    assert max(r["n"] for r in rows) <= cfg["risk"]["max_new_trades_per_day"]


def test_cash_never_negative(bt):
    from autopilot.ledger import Ledger
    eq = pd.DataFrame(Ledger(bt[1]).rows("SELECT * FROM equity"))
    assert (eq["cash"] >= -1).all()
    assert (eq["equity"] - eq["cash"] - eq["invested"]).abs().max() < 1


def test_api_reads_backtest(bt, monkeypatch):
    import autopilot.api as api
    monkeypatch.setitem(api.SOURCES, "backtest", bt[1])
    c = TestClient(api.app)
    s = c.get("/api/summary?source=backtest").json()
    assert s["equity"] > 0 and s["mode"] == "backtest"
    assert isinstance(c.get("/api/equity?source=backtest").json(), list)
    assert "rows" in c.get("/api/signals?source=backtest").json()


def test_resume_does_not_immediately_retrip(bt):
    from autopilot.ledger import Ledger
    ev = Ledger(bt[1]).rows("SELECT date, kind FROM events WHERE kind IN ('halted','resumed')")
    resumed = {e["date"] for e in ev if e["kind"] == "resumed"}
    halted = {e["date"] for e in ev if e["kind"] == "halted"}
    assert not (resumed & halted)
