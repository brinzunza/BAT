"""Unified instrument-keyed position sizing / margin registry -- merges
`research_scripts/beid_goal_hunt/combined_fx_gold_mc.py` (floor-free fractional units; XAUUSD/
BTCUSD/ETHUSD/ESF dispatch) and `research_scripts/goal20_speed_king_margin_report.py::
position_size_and_margin` (generic FX + USDJPY's $1-per-unit convention + EURGBP's fixed
cross-rate approximation) into one registry.

Both existing modules keep working unchanged (they get a thin re-export shim, not a deletion);
new code should use `quant_engine.sizing.size_and_margin` directly.

Design decision, made deliberately rather than left open: fractional (floor-free) unit sizing
is the DEFAULT for every instrument, including FX, fixing a real historic bug by construction.
`goal20_speed_king_margin_report.py::position_size_and_margin`'s `units = max(min(...), 1.0)`
whole-unit floor exists only to match OANDA's live minimum order granularity (it is "an exact
port of PRODLIVETRADING/broker/oanda_client.py::position_size") -- it was never a backtest
correctness requirement, and reusing it for gold once rejected ~99.85% of candidates for
"insufficient margin" that was actually a sizing-floor artifact (1 unit of gold/BTC/ETH is
thousands of dollars, not the ~$1 the floor assumes for FX). The floor is now an explicit,
opt-in `live_parity=True` for anyone specifically reproducing what the live bot's whole-unit
rounding would have done -- never the default.
"""
from __future__ import annotations

from dataclasses import dataclass

GBP_USD_APPROX = 1.36  # matches PRODLIVETRADING/broker/oanda_client.py's EURGBP sizing approximation


@dataclass(frozen=True)
class InstrumentSpec:
    asset_class: str
    max_leverage: float
    family: str = "default"  # "default" | "usdjpy" | "eurgbp" -- which risk/notional formula applies
    whole_unit_floor: bool = False  # only ever checked when live_parity=True is passed explicitly
    pip_size: float | None = None


INSTRUMENT_REGISTRY: dict[str, InstrumentSpec] = {
    "USDJPY": InstrumentSpec("fx", 50.0, family="usdjpy", whole_unit_floor=True, pip_size=0.01),
    "EURGBP": InstrumentSpec("fx", 50.0, family="eurgbp", whole_unit_floor=True, pip_size=0.0001),
    "EURUSD": InstrumentSpec("fx", 50.0, whole_unit_floor=True, pip_size=0.0001),
    "GBPUSD": InstrumentSpec("fx", 50.0, whole_unit_floor=True, pip_size=0.0001),
    "XAUUSD": InstrumentSpec("metal", 20.0, pip_size=0.1),
    "BTCUSD": InstrumentSpec("crypto", 10.0, pip_size=1.0),
    "ETHUSD": InstrumentSpec("crypto", 10.0, pip_size=1.0),
    "ESF": InstrumentSpec("index_future", 15.0, pip_size=0.25),
    "ES": InstrumentSpec("index_future", 15.0, pip_size=0.25),
}
_DEFAULT_FX_SPEC = InstrumentSpec("fx", 50.0, whole_unit_floor=True)


def size_and_margin(instrument: str, equity: float, entry_price: float, stop_risk: float,
                     risk_pct: float, *, cap: float | None = None,
                     spec_overrides: dict[str, InstrumentSpec] | None = None,
                     live_parity: bool = False) -> tuple[float, float, float]:
    """Returns (units, risk_dollars, margin). Floor-free by default for every instrument (see
    module docstring); pass live_parity=True to reproduce the live FX bot's whole-unit
    rounding for instruments whose InstrumentSpec has whole_unit_floor=True."""
    registry = {**INSTRUMENT_REGISTRY, **(spec_overrides or {})}
    spec = registry.get(instrument, _DEFAULT_FX_SPEC)

    risk_base = min(equity, cap) if cap is not None else equity
    risk_dollars = risk_base * risk_pct
    max_notional = risk_base * spec.max_leverage

    if spec.family == "usdjpy":
        risk_per_unit_usd = stop_risk / entry_price
        risk_based_units = risk_dollars / risk_per_unit_usd
        leverage_capped_units = max_notional  # 1 unit = $1 notional
        notional_fn = lambda u: u
    elif spec.family == "eurgbp":
        risk_per_unit_usd = stop_risk * GBP_USD_APPROX
        risk_based_units = risk_dollars / risk_per_unit_usd
        eur_usd_approx = entry_price * GBP_USD_APPROX
        leverage_capped_units = max_notional / eur_usd_approx
        notional_fn = lambda u: u * eur_usd_approx
    else:
        risk_per_unit_usd = stop_risk
        risk_based_units = risk_dollars / risk_per_unit_usd
        leverage_capped_units = max_notional / entry_price
        notional_fn = lambda u: u * entry_price

    units = min(risk_based_units, leverage_capped_units)
    if live_parity and spec.whole_unit_floor:
        units = max(units, 1.0)

    actual_risk_dollars = units * risk_per_unit_usd
    notional = notional_fn(units)
    margin = notional / spec.max_leverage
    return units, actual_risk_dollars, margin
