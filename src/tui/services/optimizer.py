"""Generic parameter grid search with a chronological train/test split."""
from __future__ import annotations

import itertools
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.core.application.backtest import BacktestService
from src.core.domain.models import BacktestConfig
from src.tui.report import _periodic_returns, equity_curve
from src.tui.strategies import ParamSpec, StrategySpec

OBJECTIVES = {
    "total_return": "Total return %",
    "sharpe": "Sharpe",
    "profit_factor": "Profit factor",
    "calmar": "Return / max drawdown",
    "win_rate": "Win rate %",
}
MAX_COMBOS = 3000


def parse_range(text: str, spec: ParamSpec) -> List[Any]:
    """``"10:40:5"`` (start:stop:step, inclusive), ``"10,20,30"`` or a single value."""
    text = (text or "").strip()
    if not text:
        return [spec.default]
    cast = int if spec.kind == "int" else float
    if spec.kind == "choice":
        vals = [v.strip() for v in text.split(",") if v.strip()]
        return [spec.parse(v) for v in vals]
    if ":" in text:
        parts = [float(x) for x in text.split(":")]
        if len(parts) != 3 or parts[2] <= 0:
            raise ValueError(f"{spec.label}: use start:stop:step with step > 0")
        start, stop, step = parts
        n = int(np.floor((stop - start) / step + 1e-9)) + 1
        vals = [start + i * step for i in range(max(n, 1))]
    else:
        vals = [float(x) for x in text.split(",") if x.strip()]
    out = []
    for v in vals:
        out.append(spec.parse(cast(round(v)) if spec.kind == "int" else round(v, 6)))
    return list(dict.fromkeys(out))


def build_grid(spec: StrategySpec, ranges: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Cartesian product, dropping combinations the strategy itself rejects."""
    names = [p.name for p in spec.params]
    combos = []
    for values in itertools.product(*(ranges.get(n, [spec.defaults()[n]]) for n in names)):
        combo = dict(zip(names, values))
        try:
            spec.parse(combo)
        except ValueError:
            continue
        combos.append(combo)
    return combos


def quick_metrics(trades: pd.DataFrame, bars: pd.DataFrame, initial: float) -> Dict[str, float]:
    """Cheap, fully vectorised scorecard used inside the optimisation loop."""
    eq = equity_curve(trades, bars, initial)
    if len(trades):
        closed = trades.dropna(subset=["Profit"])
        pnl = closed["Profit"].to_numpy(float)
    else:
        pnl = np.array([])
    ret = float(eq.iloc[-1] / initial - 1) * 100
    dd = float((eq / eq.cummax() - 1).min()) * 100
    wins, losses = pnl[pnl > 0].sum(), -pnl[pnl <= 0].sum()
    rets, ppy, _ = _periodic_returns(eq)
    sharpe = float(rets.mean() / rets.std() * np.sqrt(ppy)) if len(rets) > 2 and rets.std() > 0 and np.isfinite(ppy) else np.nan
    return dict(total_return=ret, max_dd=dd, trades=int(len(pnl)),
                win_rate=float((pnl > 0).mean() * 100) if len(pnl) else np.nan,
                profit_factor=float(wins / losses) if losses > 0 else (np.inf if wins > 0 else np.nan),
                sharpe=sharpe, calmar=ret / abs(dd) if dd < 0 else np.nan)


def run_grid(df: pd.DataFrame, spec: StrategySpec, combos: Sequence[Dict[str, Any]], config: BacktestConfig, *,
             split: float = 0.7, objective: str = "sharpe", progress: Optional[Callable[[int, int, float], None]] = None,
             cancel: Optional[threading.Event] = None) -> pd.DataFrame:
    """Backtest every combo on train and on the held-out test segment."""
    cut = int(len(df) * split)
    train, test = df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)
    svc = BacktestService()
    rows, t0 = [], time.time()
    for i, combo in enumerate(combos):
        if cancel is not None and cancel.is_set():
            break
        try:
            strat = spec.build(combo)
            row = dict(combo)
            for tag, part in (("train", train), ("test", test)):
                res = svc.run(part, strat, config)
                m = quick_metrics(res.trades, part, config.initial_cash)
                row.update({f"{tag}_{k}": v for k, v in m.items()})
            rows.append(row)
        except Exception as e:  # noqa: BLE001 - one bad combo must not stop the sweep
            rows.append({**combo, "error": str(e)})
        if progress:
            progress(i + 1, len(combos), time.time() - t0)
    out = pd.DataFrame(rows)
    if len(out) and f"train_{objective}" in out:
        out = out.sort_values(f"train_{objective}", ascending=False, na_position="last").reset_index(drop=True)
    return out
