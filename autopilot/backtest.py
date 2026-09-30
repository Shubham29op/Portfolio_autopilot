"""Event-driven backtest: replays daily bars through the same Engine + PaperBroker
used for live paper trading, with out-of-sample (walk-forward) model probabilities."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from autopilot.broker.paper import PaperBroker
from autopilot.engine import Engine
from autopilot.ledger import Ledger
from autopilot.ml.model import walk_forward

log = logging.getLogger(__name__)


def perf_stats(values: pd.Series, periods: int = 252) -> dict:
    values = values.dropna()
    if len(values) < 2:
        return {}
    rets = values.pct_change().dropna()
    years = (values.index[-1] - values.index[0]).days / 365.25
    cagr = (values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    dd = values / values.cummax() - 1
    vol = rets.std() * np.sqrt(periods)
    return {
        "start": str(values.index[0].date()), "end": str(values.index[-1].date()),
        "total_return": round(values.iloc[-1] / values.iloc[0] - 1, 4),
        "cagr": round(cagr, 4), "max_drawdown": round(dd.min(), 4),
        "volatility": round(vol, 4),
        "sharpe_rf0": round(rets.mean() / rets.std() * np.sqrt(periods), 2) if rets.std() else None,
        "calmar": round(cagr / abs(dd.min()), 2) if dd.min() < 0 else None,
    }


def run_backtest(cfg: dict, universe, bars, feats, labels, db_path: str | Path,
                 start: str | None = None, preds: pd.DataFrame | None = None,
                 folds: list[dict] | None = None) -> dict:
    from autopilot.config import deep_merge
    # simulate you pressing Resume ~a month after a circuit-breaker halt
    cfg = deep_merge(cfg, {"risk": {"auto_resume_days": cfg["risk"].get("auto_resume_days") or 30}})
    db_path = Path(db_path)
    if db_path.exists():
        db_path.unlink()
    ledger = Ledger(db_path)
    broker = PaperBroker(cfg, universe, cfg["capital"])
    engine = Engine(cfg, universe, broker, ledger,
                    use_research=cfg.get("research", {}).get("apply_in_backtest", False))

    if preds is None:
        preds, folds = walk_forward(feats, labels, cfg, universe.tradable)
    folds = folds or []
    dates = preds.index.get_level_values("date").unique().sort_values()
    if start:
        dates = dates[dates >= pd.Timestamp(start)]

    long = pd.concat(bars, names=["symbol", "date"]).swaplevel().sort_index()
    long = long[long.index.get_level_values("date").isin(dates)]
    by_date = {d: g.droplevel("date") for d, g in long.groupby(level="date")}
    feats_by_date = {d: g.droplevel("date") for d, g in
                     feats[feats.index.get_level_values("date").isin(dates)].groupby(level="date")}
    preds_by_date = {d: g.droplevel("date") for d, g in preds.groupby(level="date")}

    ledger.event(str(dates[0].date()), "info", "Backtest started with out-of-sample model probabilities.")
    for d in dates:
        day = by_date.get(d)
        if day is None:
            continue
        bars_today = day[["open", "high", "low", "close"]].to_dict("index")
        engine.run_bar_day(str(d.date()), bars_today, feats_by_date.get(d, pd.DataFrame()),
                           preds_by_date.get(d, pd.DataFrame(columns=["prob", "exp_ret"])))

    report = build_report(ledger, folds, cfg)
    ledger.set("report", report)
    ledger.set("model_meta", {"version": "walk-forward", "walk_forward_auc_mean": report["model"]["auc_mean"],
                              "walk_forward_ic_mean": report["model"]["exp_ret_ic_mean"],
                              "walk_forward_folds": report["model"]["folds"]})
    ledger.commit()
    return report


def build_report(ledger: Ledger, folds: list[dict], cfg: dict) -> dict:
    eq = pd.DataFrame(ledger.rows("SELECT * FROM equity ORDER BY date"))
    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq.set_index("date")
    fills = pd.DataFrame(ledger.rows("SELECT * FROM fills"))
    sells = fills[fills["side"] == "sell"] if not fills.empty else fills
    aucs = [f["auc"] for f in folds if f.get("auc") is not None]
    ics = [f["exp_ret_ic"] for f in folds if f.get("exp_ret_ic") is not None]
    yearly = (eq["equity"].resample("YE").last() / eq["equity"].resample("YE").first() - 1)
    bench_yearly = (eq["benchmark"].resample("YE").last() / eq["benchmark"].resample("YE").first() - 1)
    trades = {
        "round_trips": int(len(sells)),
        "win_rate": round(float((sells["pnl"] > 0).mean()), 3) if len(sells) else None,
        "avg_pnl": round(float(sells["pnl"].mean()), 2) if len(sells) else None,
        "total_charges": round(float(fills["charges_total"].sum()), 2) if not fills.empty else 0,
        "exit_reasons": sells["reason"].value_counts().to_dict() if len(sells) else {},
        "avg_exposure": round(float((eq["invested"] / eq["equity"]).mean()), 3),
        "trades_per_year": round(len(fills) / max(1e-9, (eq.index[-1] - eq.index[0]).days / 365.25), 1),
    }
    return {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "strategy": perf_stats(eq["equity"]),
        "benchmark_buy_hold": perf_stats(eq["benchmark"]),
        "trades": trades,
        "yearly": {str(d.year): {"strategy": round(float(v), 4),
                                 "benchmark": round(float(bench_yearly.get(d, np.nan)), 4)}
                   for d, v in yearly.items()},
        "model": {"folds": len(folds), "auc_mean": round(float(np.mean(aucs)), 4) if aucs else None,
                  "auc_min": round(float(np.min(aucs)), 4) if aucs else None,
                  "exp_ret_ic_mean": round(float(np.mean(ics)), 4) if ics else None,
                  "fold_detail": folds},
        "notes": ["Tax excluded (v1).", "Stock universe = today's large caps: survivorship bias.",
                  "Charges per config/charges.yaml; idle cash earns cash.idle_yield_annual."],
    }


def tune(cfg: dict, universe, bars, feats, labels, out_dir: Path) -> list[dict]:
    """Grid search over strategy/risk knobs using ONE set of walk-forward predictions
    (the model doesn't depend on them). Selects by cfg['tune']['select_by'].
    Beware: every extra combination is a chance to overfit history."""
    import itertools

    from autopilot.config import deep_merge

    t = cfg["tune"]
    preds, folds = walk_forward(feats, labels, cfg, universe.tradable)
    keys = list(t["grid"])
    results = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for combo in itertools.product(*(t["grid"][k] for k in keys)):
        over = {}
        for k, v in zip(keys, combo):
            sec, name = k.split(".")
            over.setdefault(sec, {})[name] = v
        c = deep_merge(cfg, over)
        rep = run_backtest(c, universe, bars, feats, labels, out_dir / "tune_tmp.db", t.get("start"), preds, folds)
        row = {"params": dict(zip(keys, combo)), **rep["strategy"],
               "round_trips": rep["trades"]["round_trips"], "charges": rep["trades"]["total_charges"]}
        results.append(row)
        log.info("tune %s -> cagr %.3f dd %.3f calmar %s", row["params"], row["cagr"], row["max_drawdown"], row["calmar"])
    key = t.get("select_by", "calmar")
    results.sort(key=lambda r: (r.get(key) or -1e9), reverse=True)
    (out_dir / "tune.json").write_text(json.dumps(results, indent=2, default=str))
    return results
