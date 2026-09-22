"""The canonical trade-record schema every strategy's signal-generation code must produce, and
the only contract `quant_engine.portfolio` requires to run any strategy. Everything upstream of
this (how a signal is found) is strategy-specific; everything downstream (sizing, margin,
settlement, Monte Carlo) is generic and lives in this package.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_DTYPES: dict[str, str] = {
    "instrument": "object",
    "is_buy": "bool",
    "entry_time": "datetime64[ns]",
    "exit_time": "datetime64[ns]",
    "entry_price": "float64",
    "stop_risk": "float64",  # standardized name -- some older scripts call this `stop_dist`
    "net_r": "float64",
}
CALLER_METADATA_DTYPES: dict[str, str] = {
    "risk_pct": "float64",
    "account": "object",  # sizing-key override; defaults to `instrument` if absent
}
OPTIONAL_PYRAMID_COLUMNS = ["units", "unit_rs", "unit_entry_times", "unit_entry_prices"]
# The three list-valued columns are what actually signals "this trade has real pyramid detail"
# (mirrors portfolio.py::to_engine_arrays's own `has_pyramid` check). `units` is NOT part of
# this set: it's legitimately present alone (or defaulted to 1 by normalize_trades) for every
# ordinary single-unit trade, so treating it as pyramid evidence in the all-or-nothing check
# below would reject every non-pyramid strategy -- exactly the common case.
PYRAMID_DETAIL_COLUMNS = ["unit_rs", "unit_entry_times", "unit_entry_prices"]


class SchemaError(ValueError):
    """Raised by validate_trades() with every violation found, not just the first."""


def normalize_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Non-destructive: returns a copy. Fills `account` from `instrument` when absent, and
    fills a default `units`=1 (no pyramid columns) when none of the optional pyramid columns
    are present at all."""
    df = df.copy()
    if "account" not in df.columns and "instrument" in df.columns:
        df["account"] = df["instrument"]
    has_any_pyramid_col = any(c in df.columns for c in OPTIONAL_PYRAMID_COLUMNS)
    if not has_any_pyramid_col and "units" not in df.columns:
        df["units"] = 1
    return df


def validate_trades(df: pd.DataFrame, *, strategy_name: str = "") -> pd.DataFrame:
    """Gate function: `portfolio.run_portfolio()`'s first line. Runs normalize_trades() first,
    then checks dtypes, temporal/positivity invariants, and pyramid-column consistency.
    Collects every violation into one SchemaError so an author sees all problems in one pass.
    """
    df = normalize_trades(df)
    errors: list[str] = []
    prefix = f"[{strategy_name}] " if strategy_name else ""

    for col, dtype in {**REQUIRED_DTYPES, **CALLER_METADATA_DTYPES}.items():
        if col not in df.columns:
            errors.append(f"{prefix}missing required column {col!r}")
            continue
        if dtype == "datetime64[ns]":
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                errors.append(f"{prefix}column {col!r} must be datetime64, got {df[col].dtype}")
        elif dtype == "bool":
            if df[col].dtype != bool and not set(df[col].unique()) <= {True, False}:
                errors.append(f"{prefix}column {col!r} must be bool-like, got {df[col].dtype}")
        elif dtype == "float64":
            if not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"{prefix}column {col!r} must be numeric, got {df[col].dtype}")
        elif dtype == "object":
            pass  # str-like, no strict dtype check

    if not errors:
        if "entry_time" in df.columns and "exit_time" in df.columns:
            bad = df["exit_time"] < df["entry_time"]
            if bad.any():
                errors.append(f"{prefix}{int(bad.sum())} row(s) have exit_time < entry_time")
        if "stop_risk" in df.columns:
            bad = df["stop_risk"] <= 0
            if bad.any():
                errors.append(f"{prefix}{int(bad.sum())} row(s) have stop_risk <= 0")
        if "risk_pct" in df.columns:
            bad = (df["risk_pct"] <= 0) | (df["risk_pct"] > 1)
            if bad.any():
                errors.append(f"{prefix}{int(bad.sum())} row(s) have risk_pct outside (0, 1]")

        present_detail = [c for c in PYRAMID_DETAIL_COLUMNS if c in df.columns]
        if present_detail and len(present_detail) != len(PYRAMID_DETAIL_COLUMNS):
            missing = set(PYRAMID_DETAIL_COLUMNS) - set(present_detail)
            errors.append(f"{prefix}pyramid columns must be all-or-nothing; missing {sorted(missing)}")
        elif present_detail and "units" not in df.columns:
            errors.append(f"{prefix}pyramid detail columns present but 'units' is missing")
        elif present_detail:
            for i, row in df.iterrows():
                units = int(row["units"])
                if len(row["unit_rs"]) != units:
                    errors.append(f"{prefix}row {i}: len(unit_rs)={len(row['unit_rs'])} != units={units}")
                    break
                if len(row["unit_entry_times"]) != units or len(row["unit_entry_prices"]) != units:
                    errors.append(f"{prefix}row {i}: unit_entry_times/unit_entry_prices length mismatch")
                    break
                if units > 0 and pd.Timestamp(row["unit_entry_times"][0]) != pd.Timestamp(row["entry_time"]):
                    errors.append(f"{prefix}row {i}: unit_entry_times[0] != entry_time")
                    break

    if errors:
        raise SchemaError("\n".join(errors))
    return df
