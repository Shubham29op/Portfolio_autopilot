"""Free, automatic fundamentals from Yahoo Finance (yfinance) for NSE stocks.

Produces the same canonical columns as the Screener importer, so scoring, research and the
dashboard don't care where the numbers came from. Values that fail sanity checks are set to
NaN (never guessed). Not available from Yahoo: promoter pledge and promoter/FII/DII changes;
those stay NaN and the ownership pillar goes neutral. The LLM deep dive still looks for pledges.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
CRORE = 1e7

ROWS = {
    "revenue": ["Total Revenue", "Operating Revenue"],
    "net_income": ["Net Income", "Net Income Common Stockholders",
                   "Net Income From Continuing Operation Net Minority Interest"],
    "op_income": ["Operating Income", "Total Operating Income As Reported"],
    "ebit": ["EBIT", "Operating Income"],
    "interest": ["Interest Expense", "Interest Expense Non Operating"],
    "eps": ["Diluted EPS", "Basic EPS"],
    "total_assets": ["Total Assets"],
    "current_liab": ["Current Liabilities", "Total Current Liabilities"],
    "equity": ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"],
    "total_debt": ["Total Debt"],
    "ocf": ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"],
}

# sanity bounds: outside -> NaN
BOUNDS = {
    "sales_growth_3y": (-80, 300), "profit_growth_3y": (-100, 500), "qtr_sales_yoy": (-90, 500),
    "qtr_profit_yoy": (-500, 1000), "roce": (-100, 200), "roe_3y_avg": (-100, 150),
    "opm": (-100, 100), "opm_last_year": (-100, 100), "debt_to_equity": (0, 30),
    "interest_coverage": (-100, 1000), "pe": (0.5, 400), "pe_5y_median": (0.5, 400),
    "ev_ebitda": (0, 300), "promoter_holding": (0, 100),
}


def _row(df: pd.DataFrame | None, key: str) -> pd.Series | None:
    """Statement row as a date-ascending Series, or None."""
    if df is None or df.empty:
        return None
    for name in ROWS[key]:
        if name in df.index:
            s = pd.to_numeric(df.loc[name], errors="coerce")
            s.index = pd.to_datetime(s.index)
            s = s.sort_index().dropna()
            return s if not s.empty else None
    return None


def _cagr(s: pd.Series | None, years: int = 3) -> float:
    if s is None or len(s) < 2:
        return np.nan
    n = min(years, len(s) - 1)
    start, end = s.iloc[-1 - n], s.iloc[-1]
    if start <= 0 or end <= 0:
        return np.nan
    return ((end / start) ** (1 / n) - 1) * 100


def _yoy_quarter(s: pd.Series | None) -> float:
    """Latest quarter vs the quarter ~1 year earlier."""
    if s is None or len(s) < 2:
        return np.nan
    last_d, last_v = s.index[-1], s.iloc[-1]
    target = last_d - pd.Timedelta(days=365)
    prior = s[(s.index >= target - pd.Timedelta(days=45)) & (s.index <= target + pd.Timedelta(days=45))]
    if prior.empty or prior.iloc[0] == 0:
        return np.nan
    base = prior.iloc[0]
    return (last_v - base) / abs(base) * 100


def _last(s: pd.Series | None, k: int = 1) -> float:
    return float(s.iloc[-k]) if s is not None and len(s) >= k else np.nan


def _price_on(prices: pd.Series | None, d: pd.Timestamp) -> float:
    if prices is None or prices.empty:
        return np.nan
    p = prices[prices.index <= d]
    return float(p.iloc[-1]) if not p.empty else np.nan


def compute_metrics(inc: pd.DataFrame | None, qinc: pd.DataFrame | None, bs: pd.DataFrame | None,
                    cf: pd.DataFrame | None, info: dict, prices: pd.Series | None) -> dict:
    """Pure function: statements (rows x dates, as yfinance returns) -> canonical metrics."""
    rev, ni = _row(inc, "revenue"), _row(inc, "net_income")
    opi, ebit = _row(inc, "op_income"), _row(inc, "ebit")
    interest, eps = _row(inc, "interest"), _row(inc, "eps")
    ta, cl = _row(bs, "total_assets"), _row(bs, "current_liab")
    eq, debt = _row(bs, "equity"), _row(bs, "total_debt")
    ocf = _row(cf, "ocf")
    qrev, qni = _row(qinc, "revenue"), _row(qinc, "net_income")

    m: dict = {}
    m["price"] = info.get("currentPrice") or info.get("regularMarketPrice") or \
        (float(prices.iloc[-1]) if prices is not None and len(prices) else np.nan)
    m["market_cap"] = (info.get("marketCap") or np.nan) / CRORE
    m["sales_growth_3y"] = _cagr(rev)
    m["profit_growth_3y"] = _cagr(ni)
    m["qtr_sales_yoy"] = _yoy_quarter(qrev)
    if np.isnan(m["qtr_sales_yoy"]) and info.get("revenueGrowth") is not None:
        m["qtr_sales_yoy"] = info["revenueGrowth"] * 100
    m["qtr_profit_yoy"] = _yoy_quarter(qni)
    if np.isnan(m["qtr_profit_yoy"]) and info.get("earningsGrowth") is not None:
        m["qtr_profit_yoy"] = info["earningsGrowth"] * 100

    cap_employed = _last(ta) - _last(cl)
    m["roce"] = _last(ebit) / cap_employed * 100 if cap_employed and cap_employed > 0 else np.nan
    if ni is not None and eq is not None:
        j = ni.to_frame("ni").join(eq.to_frame("eq"), how="inner").tail(3)
        j = j[j["eq"] > 0]
        m["roe_3y_avg"] = float((j["ni"] / j["eq"]).mean() * 100) if not j.empty else np.nan
    else:
        m["roe_3y_avg"] = np.nan
    m["roe"] = _last(ni) / _last(eq) * 100 if _last(eq) and _last(eq) > 0 else np.nan
    if opi is not None and rev is not None:
        j = opi.to_frame("op").join(rev.to_frame("rev"), how="inner")
        j = j[j["rev"] > 0]
        margins = (j["op"] / j["rev"] * 100)
        m["opm"] = float(margins.iloc[-1]) if len(margins) else np.nan
        m["opm_last_year"] = float(margins.iloc[-2]) if len(margins) > 1 else np.nan
    else:
        m["opm"] = m["opm_last_year"] = np.nan
    m["debt_to_equity"] = _last(debt) / _last(eq) if _last(eq) and _last(eq) > 0 and not np.isnan(_last(debt)) else np.nan
    ie = abs(_last(interest)) if interest is not None else np.nan
    m["interest_coverage"] = _last(ebit) / ie if ie and ie > 0 else np.nan
    m["cfo_last_year"] = _last(ocf) / CRORE
    m["pat_last_year"] = _last(ni) / CRORE

    m["pe"] = info.get("trailingPE") or np.nan
    if (m["pe"] is None or np.isnan(m["pe"])) and _last(eps) and _last(eps) > 0:
        m["pe"] = m["price"] / _last(eps)
    if eps is not None and prices is not None:
        hist = [(_price_on(prices, d) / e) for d, e in eps.items() if e and e > 0]
        hist = [h for h in hist if h == h]
        m["pe_5y_median"] = float(np.median(hist)) if hist else np.nan
    else:
        m["pe_5y_median"] = np.nan
    m["ev_ebitda"] = info.get("enterpriseToEbitda") or np.nan
    hi = info.get("heldPercentInsiders")
    m["promoter_holding"] = hi * 100 if hi is not None else np.nan   # proxy: Yahoo "insiders"
    for k in ("promoter_change", "pledged_pct", "fii_change", "dii_change"):
        m[k] = np.nan
    m["name"] = info.get("longName") or info.get("shortName") or ""

    for k, (lo, hi_) in BOUNDS.items():
        v = m.get(k)
        if v is not None and v == v and not (lo <= v <= hi_):
            m[k] = np.nan
    return m


def finalize(df: pd.DataFrame, universe) -> pd.DataFrame:
    """Derived columns shared with the Screener importer + industry P/E from the universe."""
    df["sector"] = [universe[s].sector for s in df.index]
    df["industry_pe"] = df.groupby("sector")["pe"].transform("median")
    df["cfo_to_pat"] = df["cfo_last_year"] / df["pat_last_year"].where(df["pat_last_year"] > 0)
    df["opm_change"] = df["opm"] - df["opm_last_year"]
    df["pe_vs_hist"] = df["pe"] / df["pe_5y_median"].where(df["pe_5y_median"] > 0)
    df["pe_vs_industry"] = df["pe"] / df["industry_pe"].where(df["industry_pe"] > 0)
    df["nse_code"] = df.index
    return df


class YFinanceFundamentals:
    def __init__(self, pause_s: float = 0.5, retries: int = 2):
        self.pause_s, self.retries = pause_s, retries

    def fetch_one(self, symbol: str, prices: pd.Series | None) -> dict | None:
        import yfinance as yf

        for attempt in range(self.retries + 1):
            try:
                t = yf.Ticker(f"{symbol}.NS")
                info = t.info or {}
                m = compute_metrics(t.income_stmt, t.quarterly_income_stmt, t.balance_sheet,
                                    t.cashflow, info, prices)
                return m
            except Exception as exc:
                log.warning("fundamentals %s attempt %d failed: %s", symbol, attempt + 1, exc)
                time.sleep(2 * (attempt + 1))
        return None

    def snapshot(self, universe, month: str, snapshot_dir: Path,
                 bars: dict[str, pd.DataFrame] | None = None) -> tuple[pd.DataFrame, dict]:
        stocks = [s for s, i in universe.by_symbol.items() if i.type == "stock"]
        rows, failed = {}, []
        for i, sym in enumerate(stocks, 1):
            prices = bars[sym]["close"] if bars and sym in bars else None
            m = self.fetch_one(sym, prices)
            if m is None:
                failed.append(sym)
            else:
                rows[sym] = m
            if i % 20 == 0:
                log.info("fundamentals %d/%d", i, len(stocks))
            time.sleep(self.pause_s)
        df = finalize(pd.DataFrame.from_dict(rows, orient="index"), universe) if rows else pd.DataFrame()
        snapshot_dir = Path(snapshot_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        if not df.empty:
            df.to_csv(snapshot_dir / f"{month}.csv")
        metrics = [k for k in BOUNDS] + ["cfo_last_year", "pat_last_year"]
        coverage = {k: round(float(df[k].notna().mean()), 2) for k in metrics if k in df} if not df.empty else {}
        summary = {"rows": len(df), "unmatched_names": [], "missing_from_export": failed,
                   "source": "yfinance", "coverage": coverage}
        return df, summary
