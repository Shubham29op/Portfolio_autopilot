"""Load YAML config into plain dicts with a small typed universe helper."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"


@dataclass(frozen=True)
class Instrument:
    symbol: str
    type: str
    sector: str
    tradable: bool = True
    name: str = ""          # company name (from NSE list) - used for news search

    @property
    def is_equity_risk(self) -> bool:
        return self.type in ("stock", "equity_etf")

    @property
    def needs_research(self) -> bool:
        """Only individual stocks go through the monthly fundamental gate."""
        return self.type == "stock"


class Universe:
    def __init__(self, instruments: list[Instrument]):
        self.by_symbol = {i.symbol: i for i in instruments}

    @classmethod
    def load(cls, path: Path | None = None, nse_list: Path | None = None) -> "Universe":
        raw = yaml.safe_load((path or CONFIG_DIR / "universe.yaml").read_text())
        items = [Instrument(**r) for r in raw["instruments"]]
        nse_list = nse_list or CONFIG_DIR / "ind_nifty100list.csv"
        if nse_list.exists():
            stocks = load_nse_index_list(nse_list)
            items = [i for i in items if i.type != "stock"] + stocks
        return cls(items)

    @property
    def symbols(self) -> list[str]:
        return list(self.by_symbol)

    @property
    def tradable(self) -> list[str]:
        return [s for s, i in self.by_symbol.items() if i.tradable]

    def __getitem__(self, symbol: str) -> Instrument:
        return self.by_symbol[symbol]


def load_nse_index_list(path: Path) -> list[Instrument]:
    """Parse NSE's index constituent CSV (niftyindices.com download):
    columns Company Name, Industry, Symbol, Series, ISIN Code."""
    import csv

    out = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            row = {k.strip(): (v or "").strip() for k, v in row.items() if k}
            if row.get("Series", "EQ") != "EQ" or not row.get("Symbol"):
                continue
            out.append(Instrument(symbol=row["Symbol"], type="stock",
                                  sector=row.get("Industry") or "Unknown",
                                  name=row.get("Company Name", "")))
    return out


def load_settings(path: Path | None = None, overrides: dict | None = None) -> dict[str, Any]:
    cfg = yaml.safe_load((path or CONFIG_DIR / "settings.yaml").read_text())
    cfg["charges"] = yaml.safe_load((CONFIG_DIR / "charges.yaml").read_text())
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p
