"""Import a Screener.in export into a canonical monthly fundamentals snapshot."""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from autopilot.config import CONFIG_DIR

log = logging.getLogger(__name__)
NUMERIC_EXCLUDE = {"nse_code", "name"}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def load_column_map(path: Path | None = None) -> dict[str, list[str]]:
    return yaml.safe_load((path or CONFIG_DIR / "fundamentals.yaml").read_text())["columns"]


def read_export(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        # screener exports sometimes put a title row first; find the header row
        raw = pd.read_excel(path, header=None)
        header_row = next(i for i in range(min(10, len(raw)))
                          if raw.iloc[i].astype(str).str.contains("Name", case=False).any())
        df = pd.read_excel(path, header=header_row)
    else:
        df = pd.read_csv(path)
    return df.dropna(how="all")


def canonicalize(df: pd.DataFrame, colmap: dict[str, list[str]]) -> tuple[pd.DataFrame, dict]:
    lookup = {_norm(c): c for c in df.columns}
    out, report = pd.DataFrame(index=df.index), {"matched": {}, "missing": []}
    for key, aliases in colmap.items():
        src = next((lookup[_norm(a)] for a in aliases if _norm(a) in lookup), None)
        if src is None:
            report["missing"].append(key)
            out[key] = np.nan
            continue
        report["matched"][key] = src
        col = df[src]
        if key not in NUMERIC_EXCLUDE:
            col = pd.to_numeric(col.astype(str).str.replace(",", "").str.replace("%", "")
                                .str.strip().replace({"": np.nan, "nan": np.nan}), errors="coerce")
        out[key] = col
    report["unmapped_columns"] = [c for c in df.columns if c not in report["matched"].values()]
    return out, report


def attach_symbols(df: pd.DataFrame, universe) -> tuple[pd.DataFrame, list[str]]:
    """Match rows to universe symbols via the NSE Code column (required)."""
    if not df["nse_code"].notna().any():
        raise ValueError("Export has no 'NSE Code' column. Add it to your Screener screen "
                         "(name matching is too error-prone for money decisions).")
    stocks = {s for s, i in universe.by_symbol.items() if i.type == "stock"}
    df["symbol"] = df["nse_code"].astype(str).str.strip().str.upper()
    unmatched = df.loc[~df["symbol"].isin(stocks), "symbol"].tolist()
    return df[df["symbol"].isin(stocks)].drop_duplicates("symbol").set_index("symbol"), unmatched


def import_snapshot(files: list[Path], universe, month: str, snapshot_dir: Path) -> tuple[pd.DataFrame, dict]:
    colmap = load_column_map()
    frames, reports = [], []
    for f in files:
        canon, rep = canonicalize(read_export(f), colmap)
        rep["file"] = str(f)
        frames.append(canon)
        reports.append(rep)
    df, unmatched = attach_symbols(pd.concat(frames, ignore_index=True), universe)
    df["cfo_to_pat"] = df["cfo_last_year"] / df["pat_last_year"].where(df["pat_last_year"] > 0)
    df["opm_change"] = df["opm"] - df["opm_last_year"]
    df["pe_vs_hist"] = df["pe"] / df["pe_5y_median"].where(df["pe_5y_median"] > 0)
    df["pe_vs_industry"] = df["pe"] / df["industry_pe"].where(df["industry_pe"] > 0)
    df["sector"] = [universe[s].sector for s in df.index]
    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(snapshot_dir / f"{month}.csv")
    stocks = [s for s, i in universe.by_symbol.items() if i.type == "stock"]
    summary = {"rows": len(df), "unmatched_names": unmatched,
               "missing_from_export": sorted(set(stocks) - set(df.index)),
               "columns": reports}
    if summary["missing_from_export"]:
        log.warning("%d universe stocks not in export: %s", len(summary["missing_from_export"]),
                    summary["missing_from_export"][:10])
    return df, summary
