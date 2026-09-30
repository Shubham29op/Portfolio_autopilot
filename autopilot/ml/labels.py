"""Triple-barrier labels aligned with how the strategy actually trades:
enter at next day's open, stop at k*ATR below, target at m*ATR above,
give up after H days. Label = 1 only if the target is hit before the stop."""
from __future__ import annotations

import numpy as np
import pandas as pd


def triple_barrier(df: pd.DataFrame, atr: pd.Series, horizon: int,
                   stop_atr: float, target_atr: float) -> pd.DataFrame:
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    a = atr.reindex(df.index).to_numpy()
    n = len(df)
    label = np.full(n, np.nan)
    fwd_ret = np.full(n, np.nan)
    days = np.full(n, np.nan)

    entry = np.full(n, np.nan)
    entry[:-1] = o[1:]                       # buy at next open
    stop = entry - stop_atr * a
    target = entry + target_atr * a
    done = np.zeros(n, dtype=bool)
    valid = np.arange(n) + horizon < n       # need full horizon of future bars
    valid &= ~np.isnan(a) & ~np.isnan(entry)

    for k in range(1, horizon + 1):
        idx = np.arange(n) + k
        ok = valid & ~done & (idx < n)
        if not ok.any():
            continue
        j = idx[ok]
        hit_stop = l[j] <= stop[ok]           # stop checked first (conservative)
        hit_tgt = (h[j] >= target[ok]) & ~hit_stop
        pos = np.flatnonzero(ok)
        s_pos, t_pos = pos[hit_stop], pos[hit_tgt]
        label[s_pos], fwd_ret[s_pos], days[s_pos] = 0, stop[s_pos] / entry[s_pos] - 1, k
        label[t_pos], fwd_ret[t_pos], days[t_pos] = 1, target[t_pos] / entry[t_pos] - 1, k
        done[s_pos] = True
        done[t_pos] = True

    timeout = valid & ~done
    t_idx = np.flatnonzero(timeout)
    label[t_idx] = 0
    fwd_ret[t_idx] = c[t_idx + horizon] / entry[t_idx] - 1
    days[t_idx] = horizon
    return pd.DataFrame({"label": label, "fwd_ret": fwd_ret, "days": days}, index=df.index)


def build_labels(bars: dict[str, pd.DataFrame], features: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One triple-barrier label per horizon (label_h20, label_h40, ...) plus the main
    horizon's forward return (fwd_ret) used to train the expected-return model."""
    m = cfg["model"]
    horizons = sorted(set(m.get("horizons", [m["horizon_days"]])) | {m["horizon_days"]})
    out = {}
    for sym, df in bars.items():
        a = features.xs(sym, level="symbol")["atr"]
        cols = {}
        for h in horizons:
            tb = triple_barrier(df, a, h, m["label_stop_atr"], m["label_target_atr"])
            cols[f"label_h{h}"] = tb["label"]
            if h == m["horizon_days"]:
                cols["label"] = tb["label"]
                cols["fwd_ret"] = tb["fwd_ret"]
                cols["days"] = tb["days"]
        out[sym] = pd.DataFrame(cols, index=df.index)
    return pd.concat(out, names=["symbol", "date"]).swaplevel().sort_index()
