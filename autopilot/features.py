"""Feature engineering. Every feature at date t uses data up to and including
the close of t only - no lookahead. Tests enforce this."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    # returns / momentum
    "ret_5", "ret_10", "ret_21", "ret_63", "ret_126", "ret_252", "mom_risk_adj", "mom_12_1",
    # trend
    "dist_sma20", "dist_sma50", "dist_sma200", "sma200_slope", "sma50_slope", "sma50_above_200",
    # volatility / range
    "vol_21", "vol_63", "vol_ratio_5_20", "atr_pct", "atr_slope", "bb_z", "pos_in_range_20",
    "gap_abs_5", "max_ret_21",
    # oscillators / volume
    "rsi_14", "volume_ratio", "turnover_log",
    # drawdown / highs
    "dd_252", "dist_high_63",
    # relative to market and sector
    "rel_strength_21", "rel_strength_63", "beta_63", "corr_63", "sector_rel_21", "sector_mom_63",
    # cross-sectional ranks (today)
    "rank_ret_21", "rank_ret_126", "rank_mom_risk_adj", "rank_vol_21",
    # market state
    "mkt_ret_21", "mkt_dist_sma200", "mkt_vol_21", "mkt_vol_ratio", "breadth_sma200", "breadth_ret_21",
]


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(),
                    (df["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def symbol_features(df: pd.DataFrame) -> pd.DataFrame:
    c, v = df["close"], df["volume"]
    r = np.log(c).diff()
    f = pd.DataFrame(index=df.index)
    for n in (5, 10, 21, 63, 126, 252):
        f[f"ret_{n}"] = c / c.shift(n) - 1
    f["mom_12_1"] = c.shift(21) / c.shift(252) - 1          # classic 12-1 momentum
    f["vol_21"] = r.rolling(21).std() * np.sqrt(252)
    f["vol_63"] = r.rolling(63).std() * np.sqrt(252)
    vol_126 = r.rolling(126).std() * np.sqrt(252)
    f["vol_ratio_5_20"] = r.rolling(5).std() / r.rolling(20).std().replace(0, np.nan)
    f["mom_risk_adj"] = f["ret_126"] / vol_126.replace(0, np.nan)
    sma20, sma50, sma200 = c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200).mean()
    f["dist_sma20"] = c / sma20 - 1
    f["dist_sma50"] = c / sma50 - 1
    f["dist_sma200"] = c / sma200 - 1
    f["sma200_slope"] = sma200 / sma200.shift(21) - 1
    f["sma50_slope"] = sma50 / sma50.shift(10) - 1
    f["sma50_above_200"] = (sma50 > sma200).astype(float)
    std20 = c.rolling(20).std()
    f["bb_z"] = (c - sma20) / std20.replace(0, np.nan)
    hi20, lo20 = df["high"].rolling(20).max(), df["low"].rolling(20).min()
    f["pos_in_range_20"] = (c - lo20) / (hi20 - lo20).replace(0, np.nan)
    a = atr(df)
    f["atr"] = a
    f["atr_pct"] = a / c
    f["atr_slope"] = a / a.shift(21) - 1
    f["gap_abs_5"] = (df["open"] / c.shift(1) - 1).abs().rolling(5).mean()
    f["max_ret_21"] = r.rolling(21).max()                     # lottery-ness
    f["rsi_14"] = rsi(c)
    f["volume_ratio"] = v.rolling(20).mean() / v.rolling(60).mean().replace(0, np.nan)
    f["turnover_log"] = np.log1p((v * c).rolling(20).mean())
    f["dd_252"] = c / c.rolling(252).max() - 1
    f["dist_high_63"] = c / df["high"].rolling(63).max() - 1
    f["logret"] = r
    f["close"] = c
    return f


def build_features(bars: dict[str, pd.DataFrame], benchmark: str,
                   sectors: dict[str, str] | None = None) -> pd.DataFrame:
    """Panel of features indexed by (date, symbol)."""
    per = {s: symbol_features(df) for s, df in bars.items()}
    panel = pd.concat(per, names=["symbol", "date"]).swaplevel().sort_index()

    bench = per[benchmark]
    mkt = pd.DataFrame({
        "mkt_ret_21": bench["ret_21"], "mkt_ret_63": bench["ret_63"],
        "mkt_dist_sma200": bench["dist_sma200"], "mkt_vol_21": bench["vol_21"],
        "mkt_vol_ratio": bench["vol_21"] / bench["vol_63"].replace(0, np.nan),
        "mkt_logret": bench["logret"],
    })
    panel = panel.join(mkt, on="date")
    panel["rel_strength_21"] = panel["ret_21"] - panel["mkt_ret_21"]
    panel["rel_strength_63"] = panel["ret_63"] - panel["mkt_ret_63"]

    # rolling beta / correlation vs benchmark (63d), computed per symbol
    def _beta_corr(g):
        x, y = g["mkt_logret"], g["logret"]
        cov = y.rolling(63).cov(x)
        var = x.rolling(63).var()
        return pd.DataFrame({"beta_63": cov / var.replace(0, np.nan), "corr_63": y.rolling(63).corr(x)},
                            index=g.index)
    bc = panel.groupby(level="symbol", group_keys=False).apply(_beta_corr)
    panel[["beta_63", "corr_63"]] = bc[["beta_63", "corr_63"]]

    # sector relatives
    sectors = sectors or {}
    panel["sector"] = panel.index.get_level_values("symbol").map(lambda s: sectors.get(s, "none"))
    gs = panel.groupby([panel.index.get_level_values("date"), panel["sector"]])
    panel["sector_mom_63"] = gs["ret_63"].transform("median")
    panel["sector_rel_21"] = panel["ret_21"] - gs["ret_21"].transform("median")
    panel = panel.drop(columns=["sector"])

    g = panel.groupby(level="date")
    panel["rank_ret_21"] = g["ret_21"].rank(pct=True)
    panel["rank_ret_126"] = g["ret_126"].rank(pct=True)
    panel["rank_mom_risk_adj"] = g["mom_risk_adj"].rank(pct=True)
    panel["rank_vol_21"] = g["vol_21"].rank(pct=True)
    panel["breadth_sma200"] = g["dist_sma200"].transform(lambda x: (x > 0).mean())
    panel["breadth_ret_21"] = g["ret_21"].transform(lambda x: (x > 0).mean())
    return panel.replace([np.inf, -np.inf], np.nan)
