"""Quantitative fundamental scoring: hard filters, five pillars, sector score."""
from __future__ import annotations

import numpy as np
import pandas as pd

# (metric, higher_is_better, weight within pillar)
PILLARS = {
    "growth": [("sales_growth_3y", True, 0.3), ("profit_growth_3y", True, 0.3),
               ("qtr_sales_yoy", True, 0.2), ("qtr_profit_yoy", True, 0.2)],
    "quality": [("roce", True, 0.4), ("roe_3y_avg", True, 0.4), ("opm_change", True, 0.2)],
    "balance_sheet": [("debt_to_equity", False, 0.4), ("interest_coverage", True, 0.3),
                      ("cfo_to_pat", True, 0.3)],
    "valuation": [("pe_vs_hist", False, 0.4), ("pe_vs_industry", False, 0.4), ("ev_ebitda", False, 0.2)],
    "ownership": [("promoter_change", True, 0.3), ("fii_change", True, 0.25),
                  ("dii_change", True, 0.25), ("pledged_pct", False, 0.2)],
    # top-down forecast (research/forecast.py): next-year profit growth, upside to fair value,
    # free-cash-flow yield. Absent (Screener source, too little history) -> neutral 50.
    "forecast": [("fc_eps_growth_1y", True, 0.4), ("fc_upside", True, 0.4), ("fc_fcf_yield", True, 0.2)],
}
PILLAR_NAMES = list(PILLARS)
# metrics that are meaningless for banks / NBFCs / insurers
NOT_FOR_FINANCIALS = {"roce", "debt_to_equity", "interest_coverage", "cfo_to_pat", "ev_ebitda", "opm_change",
                      "fc_fcf_yield"}


def _pct_rank(s: pd.Series, higher: bool) -> pd.Series:
    r = s.rank(pct=True, ascending=higher) * 100
    return r


def hard_filter(row: pd.Series, cfg: dict, is_financial: bool) -> list[str]:
    f = cfg["research"]["hard_filters"]
    fails = []

    def bad(metric, cond, msg):
        v = row.get(metric)
        if pd.notna(v) and cond(v):
            fails.append(msg.format(v=v))

    if not is_financial:
        bad("debt_to_equity", lambda v: v > f["max_debt_to_equity"], "debt/equity {v:.2f}")
        bad("interest_coverage", lambda v: v < f["min_interest_coverage"], "interest cover {v:.1f}x")
        bad("cfo_to_pat", lambda v: v < f["min_cfo_to_pat"], "cash conversion {v:.2f} (CFO/PAT)")
    bad("pledged_pct", lambda v: v > f["max_pledged_pct"], "promoter pledge {v:.1f}%")
    bad("profit_growth_3y", lambda v: v < f["min_profit_growth_3y"], "3y profit growth {v:.0f}%")
    bad("pat_last_year", lambda v: v <= 0, "loss-making (PAT {v:.0f} Cr)")
    return fails


def score_snapshot(df: pd.DataFrame, cfg: dict, sector_momentum: pd.Series | None = None) -> pd.DataFrame:
    r = cfg["research"]
    fin = df["sector"].isin(r["financial_sectors"])
    out = pd.DataFrame(index=df.index)
    out["sector"] = df["sector"]
    coverage = []

    for pillar, metrics in PILLARS.items():
        parts, weights = [], []
        for m, higher, w in metrics:
            if m not in df or df[m].notna().sum() < 3:
                continue
            ranked = _pct_rank(df[m], higher)
            if m in NOT_FOR_FINANCIALS:
                ranked = ranked.where(~fin)          # NaN for financials -> reweighted away
            parts.append(ranked * w)
            weights.append(ranked.notna() * w)
        if parts:
            num = pd.concat(parts, axis=1).sum(axis=1, min_count=1)
            den = pd.concat(weights, axis=1).sum(axis=1)
            out[pillar] = (num / den.replace(0, np.nan)).fillna(50.0)   # no data -> neutral
            coverage.append(den / sum(w for _, _, w in metrics))
        else:
            out[pillar] = 50.0
            coverage.append(pd.Series(0.0, index=df.index))
    out["data_coverage"] = pd.concat(coverage, axis=1).mean(axis=1).round(2)

    pw = r["pillar_weights"]
    out["company_score"] = sum(out[p] * w for p, w in pw.items()) / sum(pw.values())

    # ---- sector score: fundamentals of the sector's companies + sector price momentum
    sec = df.groupby("sector")[["sales_growth_3y", "profit_growth_3y", "qtr_sales_yoy"]].median()
    sec_score = sec.rank(pct=True).mean(axis=1) * 100
    if sector_momentum is not None and not sector_momentum.empty:
        mom = sector_momentum.reindex(sec.index).rank(pct=True) * 100
        sec_score = (0.6 * sec_score + 0.4 * mom.fillna(50)).where(mom.notna(), sec_score)
    sec_score = sec_score.fillna(50.0)
    out["sector_score"] = out["sector"].map(sec_score).fillna(50.0)

    sw = r["sector_weight"]
    out["quant_score"] = ((1 - sw) * out["company_score"] + sw * out["sector_score"]).round(1)
    out["filters_failed"] = [hard_filter(df.loc[s], cfg, bool(fin.loc[s])) for s in df.index]
    out["passes_filters"] = out["filters_failed"].map(len) == 0
    for c in PILLAR_NAMES + ["company_score", "sector_score"]:
        out[c] = out[c].round(1)
    return out.sort_values("quant_score", ascending=False)


def sector_momentum_from_features(feats_today: pd.DataFrame, universe) -> pd.Series:
    """Median 6-month return of each sector's stocks, from the daily feature panel."""
    if feats_today is None or feats_today.empty:
        return pd.Series(dtype=float)
    rows = [(universe[s].sector, feats_today.at[s, "ret_126"]) for s in feats_today.index
            if s in universe.by_symbol and universe[s].type == "stock"]
    if not rows:
        return pd.Series(dtype=float)
    return pd.DataFrame(rows, columns=["sector", "ret"]).groupby("sector")["ret"].median()
