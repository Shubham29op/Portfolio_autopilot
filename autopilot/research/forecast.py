"""Top-down forecast from statement lines only (revenue first, then costs as a share of revenue).

Non-financials: revenue grows at the median of its past yearly growth (clipped to a sane band);
EBITDA, depreciation, interest, tax, capex and working capital are medians of their historical
share of revenue (medians, so one odd year doesn't set the forecast). That gives forecast
EBITDA, net profit and free cash flow; fair value = forecast EBITDA x the stock's own median
EV/EBITDA, less net debt.

Banks/NBFCs have no meaningful revenue or capex lines, so book value is rolled forward at the
sustainable growth rate (ROE x retention) and valued at the stock's own median price/book.

No per-company operating drivers (rooms, stores, subscribers) are available from free feeds,
so this is steps 2-5 of the sell-side method with step 1 approximated by the revenue trend.
Every output is bounded; anything outside the bounds is NaN, never a guess.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from autopilot.research.yf_fundamentals import _last, _price_on, _row

PCT = 100.0

# statement rows used only by the forecast (the scoring rows live in yf_fundamentals.ROWS)
FC_ROWS = {
    "ebitda": ["EBITDA", "Normalized EBITDA"],
    "da": ["Reconciled Depreciation", "Depreciation And Amortization", "Depreciation"],
    "pretax": ["Pretax Income"],
    "tax": ["Tax Provision"],
    "capex": ["Capital Expenditure", "Capital Expenditure Reported"],
    "chg_wc": ["Change In Working Capital"],
    "shares": ["Diluted Average Shares", "Basic Average Shares", "Ordinary Shares Number"],
    "dividends": ["Cash Dividends Paid", "Common Stock Dividend Paid"],
    "cash": ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments",
             "Cash Financial"],
}

DEFAULTS = {"rev_growth_band": [-20, 40], "fin_growth_band": [0, 30], "min_years": 2}

# sanity bounds on outputs (percent): outside -> NaN
BOUNDS = {
    "fc_rev_growth": (-20, 40), "fc_ebitda_growth": (-60, 150), "fc_eps_growth_1y": (-80, 200),
    "fc_eps_growth_3y": (-50, 100), "fc_fcf_yield": (-30, 40), "fc_upside": (-80, 300),
}
OUTPUTS = list(BOUNDS) + ["fc_fair_value", "fc_confidence", "fc_method"]


def _stmt_row(df, key):
    """Like yf_fundamentals._row but for the forecast-only rows."""
    if df is None or df.empty:
        return None
    for name in FC_ROWS[key]:
        if name in df.index:
            s = pd.to_numeric(df.loc[name], errors="coerce")
            s.index = pd.to_datetime(s.index)
            s = s.sort_index().dropna()
            return s if not s.empty else None
    return None


def _first(*series):
    """First non-None Series (pandas objects can't be chained with `or`)."""
    for s in series:
        if s is not None:
            return s
    return None


def _share_median(num: pd.Series | None, rev: pd.Series, absolute: bool = False) -> float:
    """Median of num / revenue over the years both exist."""
    if num is None:
        return np.nan
    j = num.to_frame("n").join(rev.to_frame("r"), how="inner")
    j = j[j["r"] > 0]
    if j.empty:
        return np.nan
    ratio = (j["n"].abs() if absolute else j["n"]) / j["r"]
    return float(ratio.median())


def _nan_out() -> dict:
    out = {k: np.nan for k in OUTPUTS}
    out["fc_confidence"], out["fc_method"] = 0.0, None
    return out


def _bounded(out: dict) -> dict:
    for k, (lo, hi) in BOUNDS.items():
        v = out.get(k)
        if v is not None and v == v and not (lo <= v <= hi):
            out[k] = np.nan
    if out.get("fc_upside") != out.get("fc_upside"):        # upside NaN -> fair value meaningless
        out["fc_fair_value"] = np.nan
    return out


def _shares_latest(shares: pd.Series | None, info: dict) -> float:
    s = _last(shares)
    if s == s and s > 0:
        return s
    so = info.get("sharesOutstanding")
    return float(so) if so else np.nan


def forecast_company(inc, bs, cf, info: dict, prices: pd.Series | None,
                     is_financial: bool = False, cfg: dict | None = None) -> dict:
    """Pure function: yfinance-shaped statements -> fc_* metrics (percent unless noted)."""
    c = {**DEFAULTS, **(cfg or {})}
    out = _nan_out()
    rev, ni = _row(inc, "revenue"), _row(inc, "net_income")
    eq, debt = _row(bs, "equity"), _row(bs, "total_debt")
    shares = _first(_stmt_row(inc, "shares"), _stmt_row(bs, "shares"))
    price = info.get("currentPrice") or info.get("regularMarketPrice") or \
        (float(prices.iloc[-1]) if prices is not None and len(prices) else np.nan)
    n_shares = _shares_latest(shares, info)

    if is_financial:
        return _bounded(_forecast_financial(out, ni, eq, _stmt_row(cf, "dividends"), shares, n_shares,
                                            price, prices, c))

    if rev is None or ni is None or len(rev) < c["min_years"] or _last(rev) <= 0:
        return out
    growth = (rev / rev.shift(1) - 1).dropna() * PCT
    growth = growth[rev.shift(1).reindex(growth.index) > 0]
    if growth.empty:
        return out
    lo, hi = c["rev_growth_band"]
    g = float(np.clip(growth.median(), lo, hi)) / PCT

    ebitda = _stmt_row(inc, "ebitda")
    da = _first(_stmt_row(inc, "da"), _stmt_row(cf, "da"))
    if ebitda is None:
        opi = _row(inc, "op_income")
        if opi is not None and da is not None:
            ebitda = (opi.to_frame("o").join(da.to_frame("d"), how="inner").sum(axis=1))
    if ebitda is None or _last(ebitda) <= 0:
        return out
    m_ebitda = _share_median(ebitda, rev)
    m_da = _share_median(da, rev, absolute=True)
    m_int = _share_median(_row(inc, "interest"), rev, absolute=True)
    m_capex = _share_median(_stmt_row(cf, "capex"), rev, absolute=True)
    m_wc = _share_median(_stmt_row(cf, "chg_wc"), rev)               # cash-flow sign: + is a source
    pretax, tax = _stmt_row(inc, "pretax"), _stmt_row(inc, "tax")
    tax_rate = np.nan
    if pretax is not None and tax is not None:
        j = pretax.to_frame("p").join(tax.to_frame("t"), how="inner")
        j = j[j["p"] > 0]
        if not j.empty:
            tax_rate = float(np.clip((j["t"] / j["p"]).median(), 0, 0.40))
    if tax_rate != tax_rate:
        tax_rate = 0.25
    m_da = 0.0 if m_da != m_da else m_da
    m_int = 0.0 if m_int != m_int else m_int
    m_capex = m_da if m_capex != m_capex else m_capex                # no capex line: maintenance = D&A
    m_wc = 0.0 if m_wc != m_wc else m_wc

    rev0, ni0, ebitda0 = _last(rev), _last(ni), _last(ebitda)

    def year(n: int) -> dict:
        r = rev0 * (1 + g) ** n
        e = r * m_ebitda
        n_i = (e - r * m_da - r * m_int) * (1 - tax_rate)
        fcf = n_i + r * m_da - r * m_capex + r * m_wc
        return {"rev": r, "ebitda": e, "ni": n_i, "fcf": fcf}

    y1, y3 = year(1), year(3)
    out["fc_rev_growth"] = g * PCT
    out["fc_ebitda_growth"] = (y1["ebitda"] / ebitda0 - 1) * PCT
    if ni0 > 0:
        out["fc_eps_growth_1y"] = (y1["ni"] / ni0 - 1) * PCT
        out["fc_eps_growth_3y"] = ((y3["ni"] / ni0) ** (1 / 3) - 1) * PCT if y3["ni"] > 0 else np.nan
    mcap = info.get("marketCap") or (price * n_shares if price == price and n_shares == n_shares else np.nan)
    if mcap == mcap and mcap > 0:
        out["fc_fcf_yield"] = y1["fcf"] / mcap * PCT

    # fair value: forecast EBITDA x own median EV/EBITDA, less net debt
    cash = _stmt_row(bs, "cash")
    multiple = _median_multiple(ebitda, shares, debt, cash, prices)
    if multiple != multiple:
        multiple = info.get("enterpriseToEbitda") or np.nan
        conf_pen = 0.5                                                 # current multiple only
    else:
        conf_pen = 1.0
    if multiple == multiple and multiple > 0 and n_shares == n_shares and price == price and price > 0:
        net_debt = (0.0 if _last(debt) != _last(debt) else _last(debt)) - \
                   (0.0 if _last(cash) != _last(cash) else _last(cash))
        fair = (y1["ebitda"] * multiple - net_debt) / n_shares
        out["fc_fair_value"] = fair
        out["fc_upside"] = (fair / price - 1) * PCT
    out["fc_confidence"] = round(min(1.0, len(rev) / 5) * conf_pen, 2)
    out["fc_method"] = "ev_ebitda"
    return _bounded(out)


def _median_multiple(ebitda, shares, debt, cash, prices) -> float:
    """Median historical EV/EBITDA using year-end prices. NaN if price or share history is missing."""
    if prices is None or prices.empty or shares is None or ebitda is None:
        return np.nan
    vals = []
    for d, e in ebitda.items():
        if e <= 0:
            continue
        sh = shares[shares.index <= d]
        if sh.empty:
            continue
        p = _price_on(prices, d)
        if p != p:
            continue
        dd = debt[debt.index <= d] if debt is not None else pd.Series(dtype=float)
        cc = cash[cash.index <= d] if cash is not None else pd.Series(dtype=float)
        ev = p * sh.iloc[-1] + (dd.iloc[-1] if not dd.empty else 0) - (cc.iloc[-1] if not cc.empty else 0)
        if ev > 0:
            vals.append(ev / e)
    return float(np.median(vals)) if len(vals) >= 2 else np.nan


def _forecast_financial(out, ni, eq, dividends, shares, n_shares, price, prices, c) -> dict:
    """Banks/NBFCs: book value grows at ROE x retention; valued at own median price/book."""
    if ni is None or eq is None or len(ni) < c["min_years"]:
        return out
    j = ni.to_frame("ni").join(eq.to_frame("eq"), how="inner")
    j = j[(j["eq"] > 0) & (j["ni"] > 0)]
    if len(j) < c["min_years"]:
        return out
    roe = float((j["ni"] / j["eq"]).median())
    payout = 0.0
    if dividends is not None:
        d = dividends.abs().to_frame("d").join(ni.to_frame("ni"), how="inner")
        d = d[d["ni"] > 0]
        if not d.empty:
            payout = float(np.clip((d["d"] / d["ni"]).median(), 0, 1))
    lo, hi = c["fin_growth_band"]
    g = float(np.clip(roe * (1 - payout) * PCT, lo, hi)) / PCT
    bv0, ni0 = _last(eq), _last(ni)
    bv1, bv3 = bv0 * (1 + g), bv0 * (1 + g) ** 3
    ni1, ni3 = roe * bv1, roe * bv3
    out["fc_rev_growth"] = g * PCT                                    # book value growth for a lender
    out["fc_eps_growth_1y"] = (ni1 / ni0 - 1) * PCT
    out["fc_eps_growth_3y"] = ((ni3 / ni0) ** (1 / 3) - 1) * PCT
    pb = _median_pb(eq, shares, prices)
    if pb == pb and n_shares == n_shares and price == price and price > 0:
        fair = pb * bv1 / n_shares
        out["fc_fair_value"] = fair
        out["fc_upside"] = (fair / price - 1) * PCT
    out["fc_confidence"] = round(min(1.0, len(j) / 5) * (1.0 if pb == pb else 0.5), 2)
    out["fc_method"] = "book_value"
    return out


def _median_pb(eq, shares, prices) -> float:
    if prices is None or prices.empty or shares is None:
        return np.nan
    vals = []
    for d, e in eq.items():
        sh = shares[shares.index <= d]
        p = _price_on(prices, d)
        if e > 0 and not sh.empty and p == p and sh.iloc[-1] > 0:
            vals.append(p / (e / sh.iloc[-1]))
    return float(np.median(vals)) if len(vals) >= 2 else np.nan
