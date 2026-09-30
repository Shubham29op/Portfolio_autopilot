"""Model training, purged walk-forward evaluation and a simple model registry."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import brier_score_loss, roc_auc_score

from autopilot.features import FEATURES

log = logging.getLogger(__name__)


def make_model(params: dict) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(random_state=42, early_stopping=False, **params)


def label_cols(cfg: dict) -> list[str]:
    m = cfg["model"]
    hs = sorted(set(m.get("horizons", [m["horizon_days"]])) | {m["horizon_days"]})
    return [f"label_h{h}" for h in hs]


def max_horizon(cfg: dict) -> int:
    m = cfg["model"]
    return max(set(m.get("horizons", [m["horizon_days"]])) | {m["horizon_days"]})


def training_frame(features: pd.DataFrame, labels: pd.DataFrame, cfg: dict,
                   symbols: list[str] | None = None) -> pd.DataFrame:
    cols = label_cols(cfg) + ["label", "fwd_ret"]
    df = features[FEATURES].join(labels[cols], how="inner")
    if symbols is not None:
        df = df[df.index.get_level_values("symbol").isin(symbols)]
    return df.dropna(subset=cols + ["ret_252", "dist_sma200"])


class Ensemble:
    """One classifier per horizon (averaged) + one expected-return regressor."""

    def __init__(self, cfg: dict):
        self.cfg, self.params = cfg, cfg["model"]["params"]
        self.clfs: dict[str, HistGradientBoostingClassifier] = {}
        self.reg: HistGradientBoostingRegressor | None = None

    def fit(self, tr: pd.DataFrame) -> "Ensemble":
        X = tr[FEATURES]
        for col in label_cols(self.cfg):
            self.clfs[col] = make_model(self.params).fit(X, tr[col].astype(int))
        # winsorised forward return keeps outliers from dominating the regressor
        y = tr["fwd_ret"].clip(tr["fwd_ret"].quantile(0.01), tr["fwd_ret"].quantile(0.99))
        self.reg = HistGradientBoostingRegressor(random_state=42, early_stopping=False, **self.params).fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        Xf = X[FEATURES]
        probs = np.column_stack([c.predict_proba(Xf)[:, 1] for c in self.clfs.values()])
        return pd.DataFrame({"prob": probs.mean(axis=1), "prob_min": probs.min(axis=1),
                             "exp_ret": self.reg.predict(Xf)}, index=X.index)


def _fold_metrics(y: np.ndarray, p: np.ndarray, fwd: np.ndarray, exp_ret: np.ndarray) -> dict:
    out = {"n": int(len(y)), "base_rate": float(y.mean()) if len(y) else None}
    if len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, p))
        out["brier"] = float(brier_score_loss(y, p))
    top = p >= np.quantile(p, 0.8) if len(p) >= 10 else np.ones_like(p, bool)
    out["top_quintile_hit_rate"] = float(y[top].mean()) if top.any() else None
    out["top_quintile_fwd_ret"] = float(np.nanmean(fwd[top])) if top.any() else None
    if len(fwd) > 10 and np.nanstd(exp_ret) > 0:
        out["exp_ret_ic"] = float(pd.Series(exp_ret).corr(pd.Series(fwd), method="spearman"))
    return out


def walk_forward(features: pd.DataFrame, labels: pd.DataFrame, cfg: dict,
                 symbols: list[str]) -> tuple[pd.DataFrame, list[dict]]:
    """Out-of-sample predictions (prob, prob_min, exp_ret) for every date after the first
    training window. Purge = longest label horizon + 1 so no label leaks into training."""
    m = cfg["model"]
    horizon, every = max_horizon(cfg), m["retrain_every_days"]
    all_rows = features[FEATURES].dropna(subset=["ret_252", "dist_sma200"])
    all_rows = all_rows[all_rows.index.get_level_values("symbol").isin(symbols)]
    train_df = training_frame(features, labels, cfg, symbols)

    dates = all_rows.index.get_level_values("date").unique().sort_values()
    preds, folds = [], []
    start_i = 0
    for i in range(len(dates)):
        cutoff = dates[max(0, i - horizon - 1)]
        if (train_df.index.get_level_values("date") <= cutoff).sum() >= m["min_train_rows"]:
            start_i = i
            break
    else:
        raise ValueError("not enough history for walk-forward training")

    for i in range(start_i, len(dates), every):
        r = dates[i]
        cutoff = dates[i - horizon - 1]
        tr = train_df[train_df.index.get_level_values("date") <= cutoff]
        model = Ensemble(cfg).fit(tr)
        test_dates = dates[i: i + every]
        te = all_rows[all_rows.index.get_level_values("date").isin(test_dates)]
        p = model.predict(te)
        preds.append(p)
        lab = train_df.reindex(te.index).dropna(subset=["label"])
        pp = p.reindex(lab.index)
        fm = _fold_metrics(lab["label"].to_numpy(), pp["prob"].to_numpy(),
                           lab["fwd_ret"].to_numpy(), pp["exp_ret"].to_numpy())
        fm.update(train_end=str(cutoff.date()), test_start=str(test_dates[0].date()),
                  test_end=str(test_dates[-1].date()), train_rows=int(len(tr)))
        folds.append(fm)
        log.info("fold %s: auc=%s ic=%s n=%s", fm["test_start"], fm.get("auc"), fm.get("exp_ret_ic"), fm["n"])
    return pd.concat(preds).sort_index(), folds


class ModelRegistry:
    """models/<version>/{model.joblib, meta.json}; champion.json points to the live one."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def train_and_register(self, features: pd.DataFrame, labels: pd.DataFrame, cfg: dict,
                           symbols: list[str], wf_folds: list[dict] | None = None) -> str:
        m = cfg["model"]
        df = training_frame(features, labels, cfg, symbols)
        model = Ensemble(cfg).fit(df)
        version = datetime.now().strftime("%Y%m%d-%H%M%S")
        vdir = self.root / version
        vdir.mkdir(parents=True)
        joblib.dump(model, vdir / "model.joblib")
        aucs = [f["auc"] for f in (wf_folds or []) if f.get("auc") is not None]
        ics = [f["exp_ret_ic"] for f in (wf_folds or []) if f.get("exp_ret_ic") is not None]
        meta = {
            "version": version,
            "features": FEATURES,
            "horizons": label_cols(cfg),
            "train_rows": int(len(df)),
            "train_start": str(df.index.get_level_values("date").min().date()),
            "train_end": str(df.index.get_level_values("date").max().date()),
            "base_rate": float(df["label"].mean()),
            "walk_forward_auc_mean": float(np.mean(aucs)) if aucs else None,
            "walk_forward_ic_mean": float(np.mean(ics)) if ics else None,
            "walk_forward_folds": len(aucs),
            "params": m["params"],
            "label": {k: m[k] for k in ("horizon_days", "label_stop_atr", "label_target_atr")},
        }
        (vdir / "meta.json").write_text(json.dumps(meta, indent=2))
        (self.root / "champion.json").write_text(json.dumps({"version": version}))
        return version

    def champion(self):
        ptr = self.root / "champion.json"
        if not ptr.exists():
            return None, None
        version = json.loads(ptr.read_text())["version"]
        vdir = self.root / version
        return joblib.load(vdir / "model.joblib"), json.loads((vdir / "meta.json").read_text())

    @staticmethod
    def predict(model, features_today: pd.DataFrame) -> pd.DataFrame:
        return model.predict(features_today)
