"""Data -> features -> labels, shared by train / backtest / live."""
from __future__ import annotations

import logging

from autopilot.config import ROOT, Universe
from autopilot.data.providers import SyntheticProvider, make_provider
from autopilot.features import build_features
from autopilot.ml.labels import build_labels

log = logging.getLogger(__name__)


def load_bars(cfg: dict, universe: Universe, refresh: bool = False):
    provider = make_provider(cfg, ROOT)
    syms = universe.symbols
    if isinstance(provider, SyntheticProvider):
        bars = provider.load(syms, sectors={s: universe[s].sector for s in syms})
    else:
        bars = provider.load(syms, cfg["data"]["history_start"], refresh=refresh)
    if cfg["benchmark"] not in bars:
        raise RuntimeError(f"benchmark {cfg['benchmark']} has no data")
    log.info("loaded %d symbols", len(bars))
    return bars


def prepare(cfg: dict, universe: Universe, refresh: bool = False):
    bars = load_bars(cfg, universe, refresh)
    feats = build_features(bars, cfg["benchmark"], {s: universe[s].sector for s in universe.symbols})
    labels = build_labels(bars, feats, cfg)
    return bars, feats, labels
