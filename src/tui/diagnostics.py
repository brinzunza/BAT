"""Plain-language checks for a backtest run.

Each function returns a list of :class:`Finding`. The Backtest page prints them in its run log so a user can tell
*why* a run produced nothing (bad data, no signals, a broken indicator, ...) instead of staring at an empty screen.
Pure functions: no Textual, no I/O.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.tui.datasets import dataset_stats

REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume", "timestamp")


@dataclass(frozen=True)
class Finding:
    level: str            # ok | info | warn | error
    text: str
    blocking: bool = False


def check_config(symbol: str, spread: float, mode: str, position_pct: float) -> List[Finding]:
    out: List[Finding] = []
    forex = symbol.startswith("C:")
    if forex:
        out.append(Finding("info", f"Forex symbol: a spread of {spread:g} pips is applied to every fill."))
    elif spread:
        out.append(Finding("info", f"Spread ({spread:g} pips) only applies to forex (C: symbols) — ignored for {symbol}."))
    if mode == "long_short":
        out.append(Finding("info", "Long/short: a Sell signal while flat opens a short; shorts add proceeds to cash, "
                                   "so position sizes compound after short entries."))
    return out


def check_data(df: Optional[pd.DataFrame], lookback: int = 0) -> List[Finding]:
    """Problems with the input bars. Findings with ``blocking=True`` stop the run."""
    if df is None or len(df) == 0:
        return [Finding("error", "No data was returned for that selection.", True)]
    out: List[Finding] = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return [Finding("error", f"Data is missing required column(s): {', '.join(missing)}. "
                                 f"Expected {', '.join(REQUIRED_COLUMNS)}.", True)]
    n = len(df)
    if lookback and n <= lookback:
        return [Finding("error", f"Only {n:,} bars, but this strategy needs more than {lookback:,} bars of history "
                                 f"before it can signal. Use more data or smaller windows.", True)]
    st = dataset_stats(df)
    span = f"{st['start']:%Y-%m-%d %H:%M} → {st['end']:%Y-%m-%d %H:%M}" if "start" in st else "no timestamps"
    bar = f" · bar {st['bar']}" if "bar" in st else ""
    out.append(Finding("ok", f"Loaded {n:,} bars · {span}{bar}"))
    if n < 100:
        out.append(Finding("warn", f"Very short sample ({n} bars): results will not be statistically meaningful."))
    if st.get("nan_cells"):
        out.append(Finding("warn", f"{st['nan_cells']} missing (NaN) price/volume cells — indicators will be blank around them."))
    if st.get("bad_ohlc"):
        out.append(Finding("warn", f"{st['bad_ohlc']} bars have inconsistent OHLC (high below low/open/close)."))
    if st.get("duplicates"):
        out.append(Finding("warn", f"{st['duplicates']} duplicate timestamps."))
    if st.get("gaps"):
        out.append(Finding("info", f"{st['gaps']} time gap(s) larger than 2× the bar size (largest {st['largest_gap']}) — "
                                   f"quiet periods or missing data; indicators treat bars as consecutive."))
    nonpos = int((df["Close"] <= 0).sum())
    if nonpos:
        out.append(Finding("warn", f"{nonpos} bars with a non-positive close are skipped by the simulator."))
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    if ts.notna().all() and not ts.is_monotonic_increasing:
        out.append(Finding("warn", "Timestamps are not in increasing order."))
    return out


def check_signals(sig: pd.DataFrame, names: Dict[str, str], indicators: Sequence[str], mode: str,
                  lookback: int = 0) -> List[Finding]:
    """Did the strategy compute sensible indicators and produce tradeable signals?"""
    out: List[Finding] = []
    present = [c for c in indicators if c in sig.columns]
    text_cols = [c for c in present if sig[c].dtype == object]
    if text_cols:
        sample = str(sig[text_cols[0]].iloc[-1])
        out.append(Finding("error", f"Indicator column(s) {', '.join(text_cols)} hold text (e.g. '{sample}') instead of "
                                    f"numbers, so no signal can ever fire. This is a bug in the strategy code, not in "
                                    f"your parameters."))
    numeric = [c for c in present if c not in text_cols and pd.api.types.is_numeric_dtype(sig[c])
               and sig[c].dtype != bool]
    dead = [c for c in numeric if sig[c].isna().all()]
    if dead:
        out.append(Finding("warn", f"Indicator(s) {', '.join(dead)} are all NaN — a window is probably longer than the data."))
    live = [c for c in numeric if c not in dead]
    if live:
        warm = int(max((sig[c].notna().to_numpy().argmax() for c in live), default=0))
        if warm > 0:
            out.append(Finding("info", f"Warm-up: the first {warm:,} bars have no indicator values, so no trades there."))
    buy = sig[names["buy"]].astype(bool)
    sell = sig[names["sell"]].astype(bool)
    nb, ns, both = int(buy.sum()), int(sell.sum()), int((buy & sell).sum())
    if nb + ns == 0:
        out.append(Finding("warn", "The strategy produced 0 buy/sell signals on this data. The parameters may be too "
                                   "strict (wider thresholds, shorter windows) or the data too quiet."))
    else:
        out.append(Finding("ok", f"Signals: {nb:,} buy · {ns:,} sell"))
        if mode == "long_only" and nb == 0:
            out.append(Finding("warn", "Long-only mode needs Buy signals to open a position, but there are none."))
        elif mode == "long_only" and ns == 0:
            out.append(Finding("info", "No Sell signals: a position, once opened, is held to the end of the data."))
    if both:
        out.append(Finding("warn", f"Buy and Sell fire together on {both:,} bars (the live engine rejects such bars)."))
    return out


def check_result(summary: Dict[str, Any], n_buy: int, n_sell: int, mode: str,
                 open_trade: Optional[Dict[str, Any]]) -> List[Finding]:
    """Interpretation of the finished simulation."""
    out: List[Finding] = []
    trades = int(summary.get("trades", 0))
    if trades == 0:
        if n_buy + n_sell == 0:
            out.append(Finding("warn", "No trades because there were no signals (see above)."))
        elif open_trade:
            out.append(Finding("warn", "A position was opened but never closed, so there are no completed round trips."))
        else:
            out.append(Finding("warn", "Signals fired but none turned into a trade (e.g. only Sell signals in long-only "
                                       "mode while flat)."))
    if open_trade:
        out.append(Finding("info", f"A {open_trade['side']} position is still open at the end "
                                   f"(unrealised {open_trade.get('unrealized', 0.0):+,.2f}); it is in the equity curve "
                                   f"but not counted as a round trip."))
    days = summary.get("days", float("nan"))
    if days == days and days < 60:
        out.append(Finding("warn", f"Short sample ({days:.0f} days): CAGR, Sharpe, Sortino and Calmar are annualised from "
                                   f"a small window and can look unrealistically good."))
    out.append(Finding("info", "Model: fills at the signal bar's close, no commissions or slippage (only a forex spread). "
                               "Real-world results will be lower."))
    return out


def explain_exception(e: BaseException) -> str:
    """A hint a non-developer can act on."""
    name = type(e).__name__
    msg = str(e) or name
    low = msg.lower()
    if isinstance(e, FileNotFoundError):
        return f"{msg} — the file is gone; open Research and press Rescan."
    if isinstance(e, KeyError):
        return f"Missing key {msg}: a required column or setting is absent."
    if "timeout" in low or "connection" in low or name in ("ConnectionError", "Timeout", "ReadTimeout"):
        return f"{msg} — the provider could not be reached; check your connection and credentials (Connections page)."
    if "401" in msg or "403" in msg or "unauthor" in low or "api key" in low:
        return f"{msg} — the provider rejected the credentials (Connections page)."
    if isinstance(e, ValueError):
        return msg
    return f"{name}: {msg}"
