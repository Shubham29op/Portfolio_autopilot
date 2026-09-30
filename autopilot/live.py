"""Live paper trading loop: real (delayed) prices, simulated execution."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from autopilot.broker.paper import PaperBroker
from autopilot.config import resolve
from autopilot.data.providers import make_provider
from autopilot.config import ROOT
from autopilot.engine import Engine
from autopilot.features import build_features
from autopilot.ledger import Ledger
from autopilot.ml.labels import build_labels
from autopilot.ml.model import ModelRegistry, walk_forward
from autopilot.pipeline import load_bars

log = logging.getLogger(__name__)


class LivePaper:
    def __init__(self, cfg: dict, universe):
        self.cfg, self.universe = cfg, universe
        self.tz = ZoneInfo(cfg["schedule"]["timezone"])
        self.ledger = Ledger(resolve(cfg["paths"]["db"]))
        self.broker = PaperBroker(cfg, universe, cfg["capital"])
        st = self.ledger.get("broker_state")
        if st:
            self.broker.load_state(st)
        self.engine = Engine(cfg, universe, self.broker, self.ledger, use_research=True)
        self.registry = ModelRegistry(resolve(cfg["paths"]["models"]))
        self.provider = make_provider(cfg, ROOT)
        self.model, self.meta = self.registry.champion()
        from autopilot.llm import make_provider as make_llm
        from autopilot.news.guard import NewsGuard
        self.llm = make_llm(cfg["llm"])
        self.news = NewsGuard(cfg, universe, self.ledger, self.broker, self.llm) \
            if cfg.get("news", {}).get("enabled") else None
        self._live_day = None

    # --------------------------------------------------------------- model
    def train(self, bars=None) -> dict:
        bars = bars or load_bars(self.cfg, self.universe, refresh=False)
        feats = build_features(bars, self.cfg["benchmark"], {s: self.universe[s].sector for s in self.universe.symbols})
        labels = build_labels(bars, feats, self.cfg)
        _, folds = walk_forward(feats, labels, self.cfg, self.universe.tradable)
        self.registry.train_and_register(feats, labels, self.cfg, self.universe.tradable, folds)
        self.model, self.meta = self.registry.champion()
        self.ledger.set("model_meta", self.meta, commit=True)
        self.ledger.event(datetime.now(self.tz).date().isoformat(), "model",
                          f"Trained model {self.meta['version']} "
                          f"(walk-forward AUC {self.meta['walk_forward_auc_mean']:.3f}, "
                          f"expected-return IC {self.meta.get('walk_forward_ic_mean') or 0:.3f}).")
        self.ledger.commit()
        return self.meta

    def _model_stale(self) -> bool:
        if not self.meta:
            return True
        trained = datetime.strptime(self.meta["version"], "%Y%m%d-%H%M%S")
        days = self.cfg["model"]["retrain_every_days"] * 7 / 5   # trading -> calendar days
        return datetime.now() - trained > timedelta(days=days)

    # ------------------------------------------------------------- market
    def now(self) -> datetime:
        return datetime.now(self.tz)

    def _hm(self, key: str):
        return datetime.strptime(key, "%H:%M").time()

    def is_trading_day(self, d: datetime) -> bool:
        return d.weekday() < 5 and d.date().isoformat() not in self.cfg["schedule"]["holidays"]

    def market_open(self, d: datetime) -> bool:
        s = self.cfg["schedule"]
        return self.is_trading_day(d) and self._hm(s["market_open"]) <= d.time() <= self._hm(s["market_close"])

    def in_exec_window(self, d: datetime) -> bool:
        a, b = self.cfg["schedule"]["execution_window"]
        return self._hm(a) <= d.time() <= self._hm(b)

    # ----------------------------------------------------------- actions
    def market_live(self, now: datetime) -> bool:
        """Weekday + not a listed holiday + NSE actually printing prices today."""
        date = now.date().isoformat()
        if self._live_day == date:
            return True
        from autopilot.data.providers import traded_today
        if self.cfg["data"]["provider"] != "yfinance" or traded_today(self.cfg["benchmark"], date):
            self._live_day = date
            return True
        log.info("no NSE prices today yet (holiday or feed delay); skipping tick")
        return False

    def news_check(self, now: datetime) -> None:
        if self.news is None:
            return
        try:
            for a in self.news.check(now):
                log.info("news %s: %s -> %s", a["symbol"], a["title"][:80], a["action"])
        except Exception:
            log.exception("news check failed (trading continues)")

    def tick(self) -> None:
        now = self.now()
        date = now.date().isoformat()
        if not self.market_live(now):
            return
        if self.broker.today != date:
            self.broker.start_day(date)
        mode = self.ledger.control()
        if mode not in ("stopped",):
            self.news_check(now)
        syms = set(self.broker.positions) | {o.symbol for o in self.broker.queue}
        if syms:
            prices = self.provider.last_prices(sorted(syms))
            # GTT stops/targets live "at the broker": simulated even if you stopped the service
            self.broker.process_tick(prices)
            # protective orders (news exits, stop repairs, circuit breaker) run any time the market is open;
            # discretionary orders only inside the execution window and only when running
            allow_discretionary = mode == "running" and self.in_exec_window(now)
            keep = [o for o in self.broker.queue if not (o.protective or allow_discretionary)]
            self.broker.queue = [o for o in self.broker.queue if o.protective or allow_discretionary]
            self.broker.execute_queue_at(prices)
            self.broker.queue += keep
        self.engine.record_fills(date)
        self.ledger.set("broker_state", self.broker.to_state())
        self.ledger.set("last_tick", now.isoformat(timespec="seconds"))
        self.ledger.commit()

    def preopen(self) -> None:
        now = self.now()
        date = now.date().isoformat()
        if self.ledger.get("preopen_done") == date:
            return
        self.news_check(now)
        self.ledger.set("broker_state", self.broker.to_state())
        self.ledger.set("preopen_done", date, commit=True)

    def eod(self) -> bool:
        """Returns True if today's bar was available and processed."""
        now = self.now()
        date = now.date().isoformat()
        if self.ledger.get("last_eod") == date:
            return True
        bars = load_bars(self.cfg, self.universe, refresh=False)
        bench = bars[self.cfg["benchmark"]]
        if bench.index.max().date().isoformat() != date:
            log.info("today's bar not available yet")
            return False
        if self._model_stale():
            self.train(bars)
        feats = build_features(bars, self.cfg["benchmark"], {s: self.universe[s].sector for s in self.universe.symbols})
        ts = pd.Timestamp(date)
        today = feats.xs(ts, level="date")
        self.maybe_monthly_research(date, bars, today)
        preds = self.registry.predict(self.model, today.dropna(subset=["ret_252", "dist_sma200"]))
        closes = {s: float(df.loc[ts, "close"]) for s, df in bars.items() if ts in df.index}
        if self.broker.today != date:
            self.broker.start_day(date)
        self.engine.close_day(date, closes, today, preds)
        self.ledger.set("last_eod", date, commit=True)
        return True

    def maybe_monthly_research(self, date: str, bars, feats_today) -> None:
        """First EOD of each month: fetch fundamentals, score, deep-dive, publish the approved list.
        Runs BEFORE that evening's decisions so they use the new list. On failure the previous
        month's list stays in force and it retries at the next EOD."""
        r = self.cfg.get("research", {})
        if not (r.get("enabled") and r.get("auto_monthly")):
            return
        month = date[:7]
        cur = self.ledger.get("research_current")
        if (cur and cur.get("month") == month) or self.ledger.get("research_attempt") == date:
            return
        self.ledger.set("research_attempt", date, commit=True)
        from autopilot.research.monthly import run_monthly_research
        log.info("monthly research for %s starting (takes several minutes)", month)
        try:
            res = run_monthly_research(self.cfg, self.universe, self.ledger, month=month,
                                       feats_today=feats_today, held=set(self.broker.positions), bars=bars)
            log.info("monthly research done: %d approved", len(res["approved"]))
        except Exception as exc:
            log.exception("monthly research failed")
            self.ledger.event(date, "research", f"Monthly research failed ({str(exc)[:150]}). Keeping the "
                                                "previous list; will retry at the next end-of-day run.")
            self.ledger.commit()

    def run_forever(self) -> None:
        s = self.cfg["schedule"]
        log.info("live paper started; mode=%s", self.ledger.control())
        if self.model is None:
            self.train()
        while True:
            now = self.now()
            try:
                pre = self._hm(s.get("preopen_check", self.cfg.get("news", {}).get("preopen_check", "08:45")))
                if self.market_open(now):
                    self.tick()
                elif self.is_trading_day(now) and pre <= now.time() < self._hm(s["market_open"]):
                    self.preopen()
                elif self.is_trading_day(now) and now.time() >= self._hm(s["eod_run"]):
                    self.eod()
            except Exception:
                log.exception("cycle failed; will retry next tick (no orders are sent on failure)")
            time.sleep(s["tick_minutes"] * 60)
