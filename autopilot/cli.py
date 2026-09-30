"""Command line entry point.

  python -m autopilot.cli fetch                 # download/refresh EOD data (yfinance)
  python -m autopilot.cli train                 # walk-forward evaluate + register model
  python -m autopilot.cli backtest [--start]    # full backtest -> data/backtest.db + report
  python -m autopilot.cli tune                  # grid-search strategy knobs on walk-forward predictions
  python -m autopilot.cli paper                 # live paper loop (runs until killed)
  python -m autopilot.cli paper-eod             # run today's end-of-day step once
  python -m autopilot.cli research              # monthly research now (auto-runs on the 1st anyway)
  python -m autopilot.cli research --no-llm     # quant scoring only
  python -m autopilot.cli research --check FILE # show which export columns were recognised
  python -m autopilot.cli serve                 # API + dashboard on :8000
Add --synthetic to any command to use the offline fake market.
"""
from __future__ import annotations

import argparse
import json
import logging

from autopilot.config import Universe, load_settings, resolve


def main() -> None:
    ap = argparse.ArgumentParser(prog="autopilot")
    ap.add_argument("command", choices=["fetch", "train", "backtest", "tune", "paper", "paper-eod", "research", "serve"])
    ap.add_argument("--synthetic", action="store_true", help="offline fake market data")
    ap.add_argument("--start", help="backtest start date YYYY-MM-DD")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--month", help="research month YYYY-MM (default: current)")
    ap.add_argument("--no-llm", action="store_true", help="research without Claude deep dives")
    ap.add_argument("--check", help="research: inspect one Screener export and exit")
    ap.add_argument("--source", choices=["auto", "screener", "yfinance"],
                    help="research data source (default from settings: auto)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    overrides = {"data": {"provider": "synthetic"}} if a.synthetic else None
    cfg, universe = load_settings(overrides=overrides), Universe.load()

    if a.command == "fetch":
        from autopilot.data.providers import make_provider, suspicious_moves
        from autopilot.config import ROOT
        prov = make_provider(cfg, ROOT)
        # refresh=True re-downloads full history so split repairs apply to clean data
        bars = prov.load(universe.symbols, cfg["data"]["history_start"], refresh=True) \
            if hasattr(prov, "repairs") else prov.load(universe.symbols)
        missing = [s for s in universe.symbols if s not in bars]
        for s, df in sorted(bars.items()):
            sus = suspicious_moves(df)
            flag = f"  CHECK one-day moves >25% on {', '.join(sus[:3])}" if sus else ""
            print(f"{s:12s} {df.index.min().date()} -> {df.index.max().date()}  {len(df)} bars{flag}")
        for r in getattr(prov, "repairs", []):
            print(f"REPAIRED {r['symbol']} {r['date']}: unadjusted split/bonus, factor {r['factor']} "
                  f"(raw one-day move {r['one_day_move']:+.0%})")
        if missing:
            print("NO DATA:", ", ".join(missing))
    elif a.command == "train":
        from autopilot.live import LivePaper
        meta = LivePaper(cfg, universe).train()
        print(json.dumps(meta, indent=2))
    elif a.command == "backtest":
        from autopilot.backtest import run_backtest
        from autopilot.pipeline import prepare
        bars, feats, labels = prepare(cfg, universe)
        report = run_backtest(cfg, universe, bars, feats, labels, resolve("data/backtest.db"), a.start)
        rdir = resolve(cfg["paths"]["reports"])
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / "backtest.json").write_text(json.dumps(report, indent=2, default=str))
        report["model"].pop("fold_detail", None)
        print(json.dumps(report, indent=2, default=str))
    elif a.command == "tune":
        from autopilot.backtest import tune
        from autopilot.pipeline import prepare
        bars, feats, labels = prepare(cfg, universe)
        res = tune(cfg, universe, bars, feats, labels, resolve(cfg["paths"]["reports"]))
        for r in res[:10]:
            print(f"{r['params']}  cagr {r['cagr']:+.1%}  maxdd {r['max_drawdown']:+.1%}  "
                  f"calmar {r['calmar']}  trades {r['round_trips']}")
        print("Full table: data/reports/tune.json. Copy the winner into config/settings.yaml yourself;"
              " prefer a combination whose neighbours also do well.")
    elif a.command == "paper":
        from autopilot.live import LivePaper
        LivePaper(cfg, universe).run_forever()
    elif a.command == "paper-eod":
        from autopilot.live import LivePaper
        lp = LivePaper(cfg, universe)
        if lp.model is None:
            lp.train()
        print("processed" if lp.eod() else "today's bar not available yet")
    elif a.command == "research":
        run_research(a, cfg, universe)
    elif a.command == "serve":
        import uvicorn
        uvicorn.run("autopilot.api:app", host="127.0.0.1", port=a.port)


def run_research(a, cfg, universe) -> None:
    from pathlib import Path

    from autopilot.ledger import Ledger
    from autopilot.research.importer import canonicalize, load_column_map, read_export
    from autopilot.research.monthly import run_monthly_research

    if a.check:
        _, rep = canonicalize(read_export(Path(a.check)), load_column_map())
        print("Recognised:")
        for k, v in rep["matched"].items():
            print(f"  {k:18s} <- {v}")
        print("Missing metrics:", ", ".join(rep["missing"]) or "none")
        print("Unused columns:", ", ".join(map(str, rep["unmapped_columns"])) or "none")
        return
    feats_today, bars = None, None
    try:
        from autopilot.features import build_features
        from autopilot.pipeline import load_bars
        bars = load_bars(cfg, universe)
        feats = build_features(bars, cfg["benchmark"], {s: universe[s].sector for s in universe.symbols})
        feats_today = feats.xs(feats.index.get_level_values("date").max(), level="date")
    except Exception as exc:
        print(f"(no price data for sector momentum / P/E history: {exc})")
    ledger = Ledger(resolve(cfg["paths"]["db"]))
    held = set((ledger.get("broker_state") or {}).get("positions", {}))
    use_llm = False if a.no_llm else None
    cur = run_monthly_research(cfg, universe, ledger, month=a.month, use_llm=use_llm,
                               feats_today=feats_today, held=held, source=a.source, bars=bars)
    print(f"{cur['month']} ({cur['mode']}, data: {cur['source']}): {len(cur['approved'])} approved")
    cov = cur["import"].get("coverage") or {}
    if cov:
        weak = {k: v for k, v in cov.items() if v < 0.8}
        print("Data coverage below 80%:", ", ".join(f"{k} {v:.0%}" for k, v in weak.items()) or "none")
    for sym, info in sorted(cur["approved"].items(), key=lambda kv: -kv[1]["fundamental_score"]):
        print(f"  {sym:12s} {info['fundamental_score']:5.1f}  results: {info['next_results_date'] or '?'}")
    if cur["red_flags"]:
        print("Red flags:", ", ".join(cur["red_flags"]))
    if cur["import"]["missing_from_export"]:
        print("No data for:", ", ".join(cur["import"]["missing_from_export"]))


if __name__ == "__main__":
    main()
