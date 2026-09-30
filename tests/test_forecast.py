import numpy as np
import pandas as pd

from autopilot.config import Universe, load_settings
from autopilot.ledger import Ledger
from autopilot.research.forecast import forecast_company
from autopilot.research.monthly import run_monthly_research
from autopilot.research.scoring import PILLARS, score_snapshot
from autopilot.research.yf_fundamentals import compute_metrics, finalize

U = Universe.load()
YEARS = pd.to_datetime(["2022-03-31", "2023-03-31", "2024-03-31", "2025-03-31", "2026-03-31"])
SHARES = 100.0


def statements(rev0=1000.0, growth=0.10, margin=0.20, debt=200.0, cash=50.0, years=YEARS):
    """Company with constant growth and ratios: forecast must reproduce them exactly."""
    rev = [rev0 * (1 + growth) ** i for i in range(len(years))]
    rows = {}
    for d, r in zip(years, rev):
        ebitda, da, interest = r * margin, r * 0.04, r * 0.01
        pretax = ebitda - da - interest
        rows[d] = {"Total Revenue": r, "EBITDA": ebitda, "Reconciled Depreciation": da,
                   "Interest Expense": -interest, "Pretax Income": pretax, "Tax Provision": pretax * 0.25,
                   "Net Income": pretax * 0.75, "Operating Income": ebitda - da,
                   "Diluted Average Shares": SHARES}
    inc = pd.DataFrame(rows)
    bs = pd.DataFrame({d: {"Total Debt": debt, "Cash And Cash Equivalents": cash, "Stockholders Equity": 800.0,
                           "Total Assets": 2000.0, "Current Liabilities": 300.0} for d in years})
    cf = pd.DataFrame({d: {"Capital Expenditure": -r * 0.06, "Change In Working Capital": -r * 0.02,
                           "Operating Cash Flow": r * 0.15} for d, r in zip(years, rev)})
    return inc, bs, cf


def flat_prices(level: float, ev_ebitda: float | None = None):
    idx = pd.bdate_range("2021-06-01", "2026-09-30")
    return pd.Series(level, index=idx)


def test_constant_ratios_are_reproduced():
    inc, bs, cf = statements()
    # price such that EV/EBITDA = 10 every year-end would need a growing price; use a flat 30
    # and let the multiple be whatever history says, then check fair value uses that median
    m = forecast_company(inc, bs, cf, {"currentPrice": 30.0, "marketCap": 30.0 * SHARES}, flat_prices(30.0))
    assert abs(m["fc_rev_growth"] - 10.0) < 1e-6
    assert abs(m["fc_ebitda_growth"] - 10.0) < 1e-6
    assert abs(m["fc_eps_growth_1y"] - 10.0) < 1e-6         # constant margins -> profit grows like revenue
    assert abs(m["fc_eps_growth_3y"] - 10.0) < 1e-6
    rev1 = 1000 * 1.1 ** 5
    ni1 = (rev1 * 0.20 - rev1 * 0.04 - rev1 * 0.01) * 0.75
    fcf1 = ni1 + rev1 * 0.04 - rev1 * 0.06 - rev1 * 0.02
    assert abs(m["fc_fcf_yield"] - fcf1 / (30.0 * SHARES) * 100) < 1e-6
    # fair value: median of (30*100 + 200 - 50) / ebitda_year over 5 years, applied to next year's EBITDA
    mult = np.median([(3000 + 150) / (1000 * 1.1 ** i * 0.20) for i in range(5)])
    fair = (rev1 * 0.20 * mult - 150) / SHARES
    assert abs(m["fc_fair_value"] - fair) < 1e-6
    assert abs(m["fc_upside"] - (fair / 30 - 1) * 100) < 1e-6
    assert m["fc_confidence"] == 1.0 and m["fc_method"] == "ev_ebitda"


def test_growth_is_clipped_and_median_not_mean():
    years = YEARS
    inc, bs, cf = statements(growth=0.10)
    # one blow-out year: revenue triples in the last year -> median growth still ~10%, not the mean
    inc.loc["Total Revenue", years[-1]] = inc.loc["Total Revenue", years[-2]] * 3
    m = forecast_company(inc, bs, cf, {"currentPrice": 30.0}, flat_prices(30.0))
    assert abs(m["fc_rev_growth"] - 10.0) < 1e-6
    inc, bs, cf = statements(growth=0.60)                    # 60% a year is clipped to the band's 40%
    m = forecast_company(inc, bs, cf, {"currentPrice": 30.0}, flat_prices(30.0))
    assert abs(m["fc_rev_growth"] - 40.0) < 1e-6


def test_bad_or_short_data_gives_no_forecast():
    inc, bs, cf = statements(years=YEARS[-1:])               # a single year: nothing to trend
    m = forecast_company(inc, bs, cf, {"currentPrice": 30.0}, flat_prices(30.0))
    assert m["fc_confidence"] == 0 and np.isnan(m["fc_eps_growth_1y"]) and m["fc_method"] is None
    m = forecast_company(None, None, None, {}, None)          # no statements at all: no crash
    assert m["fc_confidence"] == 0 and all(np.isnan(m[k]) for k in ("fc_rev_growth", "fc_upside"))
    inc, bs, cf = statements()
    inc.loc["EBITDA"] = -1.0                                   # loss-making at EBITDA level
    m = forecast_company(inc, bs, cf, {"currentPrice": 30.0}, flat_prices(30.0))
    assert m["fc_confidence"] == 0


def test_no_price_history_falls_back_to_current_multiple_with_lower_confidence():
    inc, bs, cf = statements()
    m = forecast_company(inc, bs, cf, {"currentPrice": 30.0, "enterpriseToEbitda": 12.0}, None)
    rev1 = 1000 * 1.1 ** 5
    assert abs(m["fc_fair_value"] - (rev1 * 0.20 * 12.0 - 150) / SHARES) < 1e-6
    assert m["fc_confidence"] == 0.5


def test_financial_uses_book_value_method():
    ni = [100 * 1.12 ** i for i in range(5)]
    eq = [1000 * 1.12 ** i for i in range(5)]                  # ROE 10%, payout 0 -> growth 10% (clipped band ok)
    inc = pd.DataFrame({d: {"Net Income": n, "Total Revenue": n * 3, "Diluted Average Shares": SHARES}
                        for d, n in zip(YEARS, ni)})
    bs = pd.DataFrame({d: {"Stockholders Equity": e} for d, e in zip(YEARS, eq)})
    cf = pd.DataFrame({d: {"Cash Dividends Paid": -n * 0.2} for d, n in zip(YEARS, ni)})
    m = forecast_company(inc, bs, cf, {"currentPrice": 20.0}, flat_prices(20.0), is_financial=True)
    assert m["fc_method"] == "book_value"
    assert abs(m["fc_rev_growth"] - 8.0) < 1e-6              # ROE 10% x (1 - 20% payout)
    assert abs(m["fc_eps_growth_1y"] - (0.10 * eq[-1] * 1.08 / ni[-1] - 1) * 100) < 1e-6
    pb = np.median([20.0 / (e / SHARES) for e in eq])
    assert abs(m["fc_fair_value"] - pb * eq[-1] * 1.08 / SHARES) < 1e-6
    assert np.isnan(m["fc_fcf_yield"])                          # not meaningful for a lender


def test_compute_metrics_carries_forecast_and_pillar_weights_sum_to_one():
    inc, bs, cf = statements()
    m = compute_metrics(inc, None, bs, cf, {"currentPrice": 30.0}, flat_prices(30.0))
    assert "fc_upside" in m and m["fc_confidence"] == 1.0
    cfg = load_settings()
    assert abs(sum(cfg["research"]["pillar_weights"].values()) - 1.0) < 1e-9
    assert set(cfg["research"]["pillar_weights"]) == set(PILLARS)


def _snapshot(rng, with_forecast=True):
    stocks = [s for s, i in U.by_symbol.items() if i.type == "stock"]
    rows = {}
    for s in stocks:
        inc, bs, cf = statements(growth=rng.uniform(0.0, 0.3), margin=rng.uniform(0.1, 0.3))
        price = rng.uniform(10, 60)
        rows[s] = compute_metrics(inc, None, bs, cf, {"trailingPE": rng.uniform(10, 60), "currentPrice": price},
                                  flat_prices(price) if with_forecast else None,
                                  is_financial=U[s].sector == "Financial Services")
    return finalize(pd.DataFrame.from_dict(rows, orient="index"), U)


def test_forecast_pillar_moves_score_and_is_neutral_without_data():
    cfg = load_settings()
    snap = _snapshot(np.random.default_rng(5))
    scored = score_snapshot(snap, cfg)
    assert scored["forecast"].between(0, 100).all() and scored["forecast"].std() > 0
    # Screener-style snapshot: no fc_* columns -> pillar neutral, everything still works
    bare = snap.drop(columns=[c for c in snap if c.startswith("fc_")])
    assert (score_snapshot(bare, cfg)["forecast"] == 50.0).all()
    off = load_settings(overrides={"research": {"pillar_weights": {"forecast": 0.0}}})
    without = score_snapshot(snap, off)
    assert not np.allclose(scored["company_score"], without.reindex(scored.index)["company_score"])


class FakeFetcher:
    def snapshot(self, universe, month, snapshot_dir, bars=None):
        df = _snapshot(np.random.default_rng(7))
        return df, {"rows": len(df), "unmatched_names": [], "missing_from_export": [], "source": "yfinance",
                    "coverage": {"fc_upside": 1.0}}


def test_monthly_research_publishes_forecast(tmp_path):
    cfg = load_settings(overrides={"research": {"snapshots": str(tmp_path / "s"), "inbox": str(tmp_path / "in")}})
    L = Ledger(tmp_path / "p.db")
    cur = run_monthly_research(cfg, U, L, month="2026-10", use_llm=False, fetcher=FakeFetcher())
    assert cur["forecast"] and all("fc_upside" in v for v in cur["forecast"].values())
    row = L.rows("SELECT pillars_json FROM research WHERE month='2026-10'")[0]
    import json
    assert "forecast" in json.loads(row["pillars_json"])
