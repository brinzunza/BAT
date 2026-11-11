"""Local OHLCV dataset discovery, loading and quick statistics (pure pandas; no UI imports)."""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIRS = [ROOT / "data" / "raw", ROOT / "data" / "processed", ROOT / "data" / "external",
                ROOT / "experiments" / "research" / "datasets", ROOT / "research" / "datasets"]
STD_COLS = ["Open", "High", "Low", "Close", "Volume"]


@dataclass
class DatasetInfo:
    path: Path
    name: str
    size: int
    rows: int
    start: Optional[pd.Timestamp]
    end: Optional[pd.Timestamp]
    bar_seconds: Optional[float]
    mtime: float

    @property
    def label(self) -> str:
        return self.name

    @property
    def timeframe(self) -> str:
        s = self.bar_seconds
        if not s:
            return "?"
        for unit, secs in (("d", 86400), ("h", 3600), ("m", 60), ("s", 1)):
            if s >= secs and abs(s / secs - round(s / secs)) < 1e-6:
                return f"{int(round(s / secs))}{unit}"
        return f"{s:.0f}s"


def _count_lines(path: Path) -> int:
    n = 0
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            n += chunk.count(b"\n")
    return n


def _read_edges(path: Path):
    """First/last data rows without parsing the whole file."""
    with open(path, "rb") as f:
        header = f.readline().decode("utf-8", "ignore").strip().split(",")
        first = f.readline().decode("utf-8", "ignore").strip().split(",")
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(size - 4096, 0))
        tail = f.read().decode("utf-8", "ignore").strip().splitlines()
    last = tail[-1].split(",") if tail else []
    second = None
    return header, first, last


def inspect_csv(path: Path) -> Optional[DatasetInfo]:
    try:
        header, first, last = _read_edges(path)
        cols = [h.strip().lower() for h in header]
        if "close" not in cols:
            return None
        ts_idx = cols.index("timestamp") if "timestamp" in cols else 0
        start = pd.to_datetime(first[ts_idx], errors="coerce") if len(first) > ts_idx else None
        end = pd.to_datetime(last[ts_idx], errors="coerce") if len(last) > ts_idx else None
        rows = max(_count_lines(path) - 1, 0)
        bar = None
        try:  # median spacing of the first rows: robust to gaps (weekends, outages)
            head = pd.read_csv(path, nrows=500, usecols=[ts_idx])
            dts = pd.to_datetime(head.iloc[:, 0]).diff().dropna()
            if len(dts):
                bar = float(dts.median().total_seconds())
        except Exception:
            pass
        if bar is None and rows > 1 and start is not None and end is not None and pd.notna(start) and pd.notna(end):
            bar = (end - start).total_seconds() / (rows - 1)
        st = path.stat()
        return DatasetInfo(path, path.name, st.st_size, rows, start if pd.notna(start) else None,
                           end if pd.notna(end) else None, bar, st.st_mtime)
    except Exception:
        return None


def scan_datasets(dirs: Optional[List[Path]] = None) -> List[DatasetInfo]:
    out = []
    for d in dirs or DATASET_DIRS:
        if d.is_dir():
            for p in sorted(d.glob("*.csv")):
                info = inspect_csv(p)
                if info:
                    out.append(info)
    return out


def load_dataset(path, *, start=None, end=None, resample: Optional[str] = None,
                 limit: Optional[int] = None) -> pd.DataFrame:
    """Load a CSV into BAT's standard frame: Open/High/Low/Close/Volume + ``timestamp`` column."""
    df = pd.read_csv(path)
    df.columns = [("timestamp" if c.strip().lower() in ("timestamp", "time", "date", "datetime")
                   else c.strip().capitalize()) for c in df.columns]
    if "timestamp" not in df:
        df.insert(0, "timestamp", pd.RangeIndex(len(df)))
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "Volume" not in df:
        df["Volume"] = 0.0
    missing = [c for c in ("Open", "High", "Low", "Close") if c not in df]
    if missing:
        raise ValueError(f"{Path(path).name}: missing columns {missing}")
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    if start:
        df = df[df["timestamp"] >= pd.Timestamp(start)]
    if end:
        df = df[df["timestamp"] <= pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)]
    if resample:
        df = resample_bars(df, resample)
    if limit:
        df = df.tail(int(limit))
    return df[["timestamp"] + STD_COLS + [c for c in df.columns if c not in ["timestamp"] + STD_COLS]].reset_index(drop=True)


def resample_bars(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    g = df.set_index("timestamp").resample(rule, label="left", closed="left")
    out = g.agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna(subset=["Open"])
    return out.reset_index()


def dataset_stats(df: pd.DataFrame) -> Dict[str, object]:
    """Quality + descriptive stats for a frame returned by ``load_dataset``."""
    n = len(df)
    if n == 0:
        return dict(rows=0)
    close = df["Close"].to_numpy(float)
    ts = pd.to_datetime(df["timestamp"]) if "timestamp" in df else None
    out: Dict[str, object] = dict(rows=n, low=float(df["Low"].min()), high=float(df["High"].max()),
                                  first=float(close[0]), last=float(close[-1]),
                                  change=float(close[-1] / close[0] - 1) if close[0] else np.nan,
                                  avg_volume=float(df["Volume"].mean()))
    rets = np.diff(np.log(np.where(close > 0, close, np.nan)))
    out["vol_per_bar"] = float(np.nanstd(rets)) if len(rets) > 1 else np.nan
    out["nan_cells"] = int(df[STD_COLS].isna().sum().sum())
    out["bad_ohlc"] = int(((df["High"] < df["Low"]) | (df["High"] < df[["Open", "Close"]].max(axis=1) - 1e-9)
                            | (df["Low"] > df[["Open", "Close"]].min(axis=1) + 1e-9)).sum())
    if ts is not None and pd.api.types.is_datetime64_any_dtype(ts) and n > 2:
        dt = ts.diff().dropna()
        med = dt.median()
        out["start"], out["end"] = ts.iloc[0], ts.iloc[-1]
        out["bar"] = med
        out["gaps"] = int((dt > med * 2).sum()) if med > pd.Timedelta(0) else 0
        out["largest_gap"] = dt.max()
        out["duplicates"] = int(ts.duplicated().sum())
    return out


class DatasetCache:
    """Tiny mtime-keyed cache so the UI doesn't re-parse CSVs on every tab switch."""

    def __init__(self, maxsize: int = 4):
        self.maxsize = maxsize
        self._store: Dict[tuple, pd.DataFrame] = {}

    def get(self, path, **kw) -> pd.DataFrame:
        p = Path(path)
        key = (str(p), p.stat().st_mtime, tuple(sorted(kw.items())))
        if key not in self._store:
            if len(self._store) >= self.maxsize:
                self._store.pop(next(iter(self._store)))
            self._store[key] = load_dataset(p, **kw)
        return self._store[key]


def save_dataset(df: pd.DataFrame, name: str, directory: Optional[Path] = None) -> Path:
    directory = Path(directory or ROOT / "data" / "raw")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    df.to_csv(path, index=False)
    return path


def short_name(name: str, width: int = 30) -> str:
    """Middle-ellipsis for long dataset file names."""
    if len(name) <= width:
        return name
    keep = (width - 1) // 2
    return name[:keep] + "…" + name[-(width - 1 - keep):]


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def symbol_from_name(name: str) -> str:
    """``X_BTCUSD_minute_...csv`` -> ``X:BTCUSD`` (Polygon naming; the engine's forex-spread logic keys off ``C:``)."""
    m = re.match(r"^([A-Z])_([A-Z0-9]+)_", name)
    return f"{m.group(1)}:{m.group(2)}" if m else name.split("_")[0].split(".")[0]
