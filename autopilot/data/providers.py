"""Market data providers.

All providers return {symbol: DataFrame[open, high, low, close, volume]}
indexed by tz-naive daily DatetimeIndex, adjusted for splits/dividends.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
COLS = ["open", "high", "low", "close", "volume"]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[COLS].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[(df["close"] > 0) & (df["open"] > 0)]
    # Repair occasional bad highs/lows from free feeds
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df


SPLIT_FACTORS = [2, 3, 4, 5, 10, 20, 25, 50, 100, 1.5, 1.25, 4 / 3,
                 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 10, 2 / 3]


def repair_splits(df: pd.DataFrame, symbol: str = "", jump: float = 1.8) -> tuple[pd.DataFrame, list[dict]]:
    """Free feeds sometimes miss split/bonus adjustments (common for Indian ETFs), which shows up
    as a one-day 'crash' that never recovers (e.g. -90% on a 1:10 split). Detect jumps beyond
    `jump`x that persist for a week AND match a standard split ratio, then rescale history
    before the jump. Real crashes rarely match a clean ratio; every repair is logged for review."""
    df = df.copy()
    fixes = []
    for _ in range(6):
        c = df["close"]
        ratio = c / c.shift(1)
        hits = ratio[(ratio > jump) | (ratio < 1 / jump)].index
        done = False
        for d in hits:
            i = df.index.get_loc(d)
            if i < 5 or i + 5 > len(df):
                continue
            before = c.iloc[i - 5:i].median()
            after = c.iloc[i:i + 5].median()
            k = before / after                     # >1: split (price fell), <1: consolidation
            f = min(SPLIT_FACTORS, key=lambda x: abs(np.log(k / x)))
            if abs(np.log(k / f)) > 0.05:
                continue                           # not a clean ratio: treat as real move
            cols = ["open", "high", "low", "close"]
            df.loc[df.index < d, cols] = df.loc[df.index < d, cols] / f
            df.loc[df.index < d, "volume"] = df.loc[df.index < d, "volume"] * f
            fixes.append({"symbol": symbol, "date": str(d.date()), "factor": round(f, 4),
                          "one_day_move": round(float(ratio.loc[d] - 1), 3)})
            log.warning("%s: repaired unadjusted split/bonus on %s (factor %.3g)", symbol, d.date(), f)
            done = True
            break
        if not done:
            break
    return df, fixes


def suspicious_moves(df: pd.DataFrame, limit: float = 0.25) -> list[str]:
    r = df["close"].pct_change().abs()
    return [str(d.date()) for d in r[r > limit].index]


class YFinanceProvider:
    """Free EOD data via yfinance (NSE symbols get the .NS suffix).

    Good enough for research and paper trading. Swap for Kite historical
    data before going live (see broker/kite.py).
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.repairs: list[dict] = []

    @staticmethod
    def ticker(symbol: str) -> str:
        return f"{symbol}.NS"

    def _cache_path(self, symbol: str) -> Path:
        return self.cache_dir / f"{symbol.replace('&', '_')}.csv"

    def load(self, symbols: list[str], start: str, refresh: bool = False) -> dict[str, pd.DataFrame]:
        import yfinance as yf  # local import: optional dependency at runtime

        out: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            path = self._cache_path(sym)
            cached = None
            if path.exists():
                cached = pd.read_csv(path, index_col=0, parse_dates=True)
            fetch_from = start
            if cached is not None and not cached.empty and not refresh:
                fetch_from = (cached.index.max() - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
            try:
                raw = yf.Ticker(self.ticker(sym)).history(start=fetch_from, auto_adjust=True)
                raw = raw.rename(columns=str.lower)
                fresh = _clean(raw) if not raw.empty else None
            except Exception as exc:  # network / symbol errors
                log.warning("yfinance failed for %s: %s", sym, exc)
                fresh = None
            frames = [f for f in (cached, fresh) if f is not None and not f.empty]
            if not frames:
                log.warning("no data for %s, skipping", sym)
                continue
            df = _clean(pd.concat(frames))
            df, fixes = repair_splits(df, sym)
            self.repairs.extend(fixes)
            df.to_csv(path)
            out[sym] = df
        return out

    def last_prices(self, symbols: list[str]) -> dict[str, float]:
        """Latest traded price (delayed on free feeds - fine for paper)."""
        import yfinance as yf

        prices = {}
        for sym in symbols:
            try:
                prices[sym] = float(yf.Ticker(self.ticker(sym)).fast_info["last_price"])
            except Exception as exc:
                log.warning("LTP failed for %s: %s", sym, exc)
        return prices


def traded_today(symbol: str, date_iso: str) -> bool:
    """True if NSE printed 1-minute bars for `symbol` today (holiday / closed-market guard)."""
    import yfinance as yf

    try:
        h = yf.Ticker(f"{symbol}.NS").history(period="1d", interval="1m")
        return not h.empty and h.index.max().date().isoformat() == date_iso
    except Exception as exc:
        log.warning("market-live check failed: %s", exc)
        return False


class SyntheticProvider:
    """Deterministic fake market for tests and offline development.

    Regime-switching GBM with a shared market factor, sector factors and some
    persistent drift so a momentum/ML signal has something to find.
    """

    def __init__(self, seed: int = 7, start: str = "2014-01-01", end: str = "2026-09-25"):
        self.seed, self.start, self.end = seed, start, end

    def load(self, symbols: list[str], start: str | None = None, refresh: bool = False,
             sectors: dict[str, str] | None = None) -> dict[str, pd.DataFrame]:
        rng = np.random.default_rng(self.seed)
        dates = pd.bdate_range(self.start, self.end)
        n = len(dates)
        # Market regimes: bull / bear / chop
        regime = np.zeros(n, dtype=int)
        i = 0
        while i < n:
            r = rng.choice([0, 1, 2], p=[0.55, 0.2, 0.25])
            length = int(rng.integers(60, 300))
            regime[i:i + length] = r
            i += length
        mu = np.array([0.0006, -0.0008, 0.0])[regime]
        sig = np.array([0.009, 0.018, 0.012])[regime]
        market = mu + sig * rng.standard_normal(n)
        sectors = sectors or {}
        sector_names = sorted(set(sectors.values())) or ["x"]
        sector_ret = {s: 0.004 * rng.standard_normal(n) for s in sector_names}
        out = {}
        for k, sym in enumerate(symbols):
            beta = rng.uniform(0.6, 1.3)
            # slowly varying idiosyncratic drift -> momentum persistence
            drift = np.cumsum(rng.standard_normal(n)) * 0.00002
            drift = drift - drift.mean() + rng.normal(0.0002, 0.0002)
            idio = rng.uniform(0.008, 0.016) * rng.standard_normal(n)
            sec = sector_ret.get(sectors.get(sym, "x"), 0.0)
            ret = beta * market + sec + drift + idio
            close = 100 * rng.uniform(0.5, 20) * np.exp(np.cumsum(ret))
            gap = 0.004 * rng.standard_normal(n)
            open_ = close * np.exp(-ret + gap)  # open near prior close with gap
            spread = np.abs(0.01 * rng.standard_normal(n)) + 0.003
            high = np.maximum(open_, close) * (1 + spread)
            low = np.minimum(open_, close) * (1 - spread)
            vol = rng.integers(100_000, 5_000_000, n).astype(float)
            df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                               "volume": vol}, index=dates)
            out[sym] = _clean(df)
        return out


def make_provider(cfg: dict, root: Path):
    kind = cfg["data"]["provider"]
    if kind == "yfinance":
        return YFinanceProvider(root / cfg["data"]["cache_dir"])
    if kind == "synthetic":
        return SyntheticProvider()
    raise ValueError(f"unknown provider {kind}")
