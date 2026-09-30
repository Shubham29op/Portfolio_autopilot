import numpy as np
import pandas as pd

from autopilot.config import Universe, load_settings
from autopilot.ledger import Ledger
from autopilot.research.monthly import run_monthly_research
from autopilot.research.yf_fundamentals import compute_metrics, finalize

U = Universe.load()
YEARS = pd.to_datetime(["2023-03-31", "2024-03-31", "2025-03-31", "2026-03-31"])
QTRS = pd.to_datetime(["2025-06-30", "2025-09-30", "2025-12-31", "2026-03-31", "2026-06-30"])


def statements(rev0=1000.0, growth=0.15, debt=200.0):
    rev = [rev0 * (1 + growth) ** i for i in range(4)]
    ni = [r * 0.12 for r in rev]
    inc = pd.DataFrame({d: {"Total Revenue": r * 1e7, "Net Income": n * 1e7, "Operating Income": r * 0.18e7,
                            "EBIT": r * 0.18e7, "Interest Expense": -r * 0.01e7, "Diluted EPS": n / 10}
                        for d, r, n in zip(YEARS, rev, ni)})
    bs = pd.DataFrame({d: {"Total Assets": 3000e7, "Current Liabilities": 800e7, "Stockholders Equity": 1500e7,
                           "Total Debt": debt * 1e7} for d in YEARS})
    cf = pd.DataFrame({d: {"Operating Cash Flow": n * 1.1e7} for d, n in zip(YEARS, ni)})
    qrev = [300, 310, 320, 330, 345]
    qinc = pd.DataFrame({d: {"Total Revenue": r * 1e7, "Net Income": r * 0.12e7} for d, r in zip(QTRS, qrev)})
    return inc, qinc, bs, cf


def test_compute_metrics_values():
    inc, qinc, bs, cf = statements()
    prices = pd.Series(np.linspace(100, 250, 1300), index=pd.bdate_range("2021-06-01", periods=1300))
    m = compute_metrics(inc, qinc, bs, cf, {"trailingPE": 25.0, "enterpriseToEbitda": 14.0,
                                             "heldPercentInsiders": 0.5, "longName": "Test Ltd"}, prices)
    assert abs(m["sales_growth_3y"] - 15.0) < 0.01
    assert abs(m["qtr_sales_yoy"] - 15.0) < 0.01               # 345 vs 300
    assert abs(m["debt_to_equity"] - 200 / 1500) < 1e-9
    assert abs(m["opm"] - 18.0) < 1e-9 and abs(m["cfo_last_year"] / m["pat_last_year"] - 1.1) < 1e-9
    assert m["interest_coverage"] == 18.0 and m["pe"] == 25.0
    assert m["pe_5y_median"] > 0 and m["promoter_holding"] == 50.0
    assert np.isnan(m["pledged_pct"])                          # not available from Yahoo


def test_sanity_bounds_and_missing_statements():
    inc, qinc, bs, cf = statements(debt=1e6)                    # absurd debt/equity -> NaN
    m = compute_metrics(inc, qinc, bs, cf, {"trailingPE": 9999}, None)
    assert np.isnan(m["debt_to_equity"]) and np.isnan(m["pe"])
    m2 = compute_metrics(None, None, None, None, {}, None)     # bank with no statements: all NaN, no crash
    assert np.isnan(m2["roce"]) and np.isnan(m2["sales_growth_3y"])


class FakeFetcher:
    def snapshot(self, universe, month, snapshot_dir, bars=None):
        rng = np.random.default_rng(3)
        stocks = [s for s, i in universe.by_symbol.items() if i.type == "stock"]
        rows = {}
        for s in stocks:
            inc, qinc, bs, cf = statements(growth=rng.uniform(0.02, 0.3), debt=rng.uniform(0, 4000))
            rows[s] = compute_metrics(inc, qinc, bs, cf, {"trailingPE": rng.uniform(10, 60)}, None)
        df = finalize(pd.DataFrame.from_dict(rows, orient="index"), universe)
        return df, {"rows": len(df), "unmatched_names": [], "missing_from_export": [], "source": "yfinance",
                    "coverage": {"pe": 1.0}}


def test_monthly_research_from_yfinance_without_llm(tmp_path):
    cfg = load_settings(overrides={"research": {"snapshots": str(tmp_path / "s"), "inbox": str(tmp_path / "in")}})
    L = Ledger(tmp_path / "p.db")
    cur = run_monthly_research(cfg, U, L, month="2026-10", use_llm=False, fetcher=FakeFetcher())
    assert cur["source"] == "yfinance" and cur["mode"] == "quant_only"
    assert cur["approved"]
    rows = L.rows("SELECT symbol, approved, reason FROM research WHERE month='2026-10'")
    assert any("debt/equity" in (r["reason"] or "") for r in rows)   # hard filter still works


def test_live_auto_research_runs_once_per_month(tmp_path, monkeypatch):
    from autopilot import live as live_mod
    cfg = load_settings(overrides={"data": {"provider": "synthetic"}, "news": {"enabled": False},
                                   "paths": {"db": str(tmp_path / "p.db"), "models": str(tmp_path / "m")}})
    lp = live_mod.LivePaper(cfg, U)
    calls = []
    import autopilot.research.monthly as monthly

    def fake_run(cfg_, universe, ledger, month=None, **kw):
        calls.append(month)
        ledger.set("research_current", {"month": month, "approved": {}, "version": month})
        return {"approved": {}}
    monkeypatch.setattr(monthly, "run_monthly_research", fake_run)
    lp.maybe_monthly_research("2026-10-01", None, None)
    lp.maybe_monthly_research("2026-10-05", None, None)        # already done this month
    lp.maybe_monthly_research("2026-11-02", None, None)        # new month
    assert calls == ["2026-10", "2026-11"]
