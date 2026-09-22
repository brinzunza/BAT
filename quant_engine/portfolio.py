"""The shared portfolio + Monte Carlo layer: validates a strategy's trade DataFrame against the
canonical schema, dispatches an `account_mode`, and wraps `dtfx_engine.run_portfolio_mc` -- the
most general simulation primitive in this codebase (N arbitrary accounts, margin computed
entirely in Python so it's instrument-agnostic, real per-unit pyramid margin, event-driven
exit-time settlement, two-tier ruin). The array-conversion logic below (leverage_risk_cap =
leverage * stop_risk / entry_price, the four account_mode groupings) is promoted, not
reinvented, from `research_scripts/beid_goal_hunt/corrected_event_search.py::run_cpp_mc` --
already proven against two independent strategy families' real trade data.

Two-stage target1/target2 is deliberately NOT implemented in the C++ engine (see
cpp_engine/README.md's rationale). Two calls with identical seed/block/max_trades and
different `target` values are exact continuations of the same simulated paths -- BUT ONLY
per-sim-index when each simulated path has its own independent RNG stream, i.e. only when
n_threads == n_sims (one thread per sim). The C++ worker creates one std::mt19937_64 PER
THREAD, outside its per-sim loop, so with fewer threads than sims that stream keeps advancing
across multiple sims on the same thread -- and since a lower target makes a path stop (and
therefore stop consuming random draws) earlier, two calls at different targets diverge for
every sim after the first one on a shared thread. `cpp_engine/validate_portfolio_mc.py`'s
`validate_target_invariance` found this empirically (real counter-examples with n_threads=1,
n_sims>1) before this constraint was enforced here. Because "one thread per sim" isn't
practical at the thousands-of-sims scale this codebase's production MC runs use,
`extra_targets` is a tool for small, focused two-stage checks -- not a general substitute for
`run_mc_capped`'s native two-stage tracking at scale.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/brunoinzunza/Documents/GitHub/tradingResearch/dtfxStrategy/research_scripts")
sys.path.insert(0, "/Users/brunoinzunza/Documents/GitHub/tradingResearch/cpp_engine/build")

from .schema import validate_trades
from .sizing import INSTRUMENT_REGISTRY, _DEFAULT_FX_SPEC
from .stats import true_median_and_stats

try:
    import dtfx_engine
    HAVE_CPP = True
except ImportError:
    HAVE_CPP = False

ACCOUNT_MODES = ("pooled", "instrument", "side", "instrument_side")


def _leverage_for(instrument: str) -> float:
    spec = INSTRUMENT_REGISTRY.get(instrument, _DEFAULT_FX_SPEC)
    return spec.max_leverage


def _account_grouping(t: pd.DataFrame, account_mode: str) -> tuple[np.ndarray, int]:
    if account_mode == "instrument_side":
        col = t.get("account", t.instrument)
        keys = list(dict.fromkeys(col.tolist()))
        base_map = {k: 2 * i for i, k in enumerate(keys)}
        account = col.map(base_map).to_numpy(np.int32) + t.is_buy.to_numpy(np.int32)
        return account, 2 * len(keys)
    if account_mode == "instrument":
        col = t.get("account", t.instrument)
        keys = list(dict.fromkeys(col.tolist()))
        key_map = {k: i for i, k in enumerate(keys)}
        return col.map(key_map).to_numpy(np.int32), len(keys)
    if account_mode == "side":
        return t.is_buy.to_numpy(np.int32), 2
    if account_mode == "pooled":
        return np.zeros(len(t), dtype=np.int32), 1
    raise ValueError(f"unknown account_mode={account_mode!r}; must be one of {ACCOUNT_MODES}")


def to_engine_arrays(trades: pd.DataFrame, *, account_mode: str = "instrument_side",
                      leverage_overrides: dict[str, float] | None = None) -> dict:
    """Converts a schema-valid trade DataFrame into the plain numpy arrays
    `dtfx_engine.run_portfolio_mc` needs. Public so `validate_portfolio_mc.py` and any direct
    caller can build inputs without going through the full `run_portfolio()` convenience
    wrapper."""
    t = trades.sort_values("entry_time", kind="stable").reset_index(drop=True)
    entry_ns = t.entry_time.values.astype("datetime64[ns]").astype(np.int64)
    exit_ns = t.exit_time.values.astype("datetime64[ns]").astype(np.int64)
    account, n_accounts = _account_grouping(t, account_mode)

    leverage_overrides = leverage_overrides or {}
    leverage = t.instrument.map(lambda i: leverage_overrides.get(i, _leverage_for(i))).to_numpy(float)
    leverage_risk_cap = leverage * t.stop_risk.to_numpy(float) / t.entry_price.to_numpy(float)

    unit_count = t.get("units", pd.Series(np.ones(len(t)), index=t.index)).to_numpy(np.int32)
    max_units = max(int(unit_count.max()), 1)
    unit_r = np.zeros((len(t), max_units), dtype=np.float64)
    unit_offset = np.zeros((len(t), max_units), dtype=np.int64)
    unit_ratio = np.ones((len(t), max_units), dtype=np.float64)
    has_pyramid = {"unit_rs", "unit_entry_times", "unit_entry_prices"}.issubset(t.columns)
    for i in range(len(t)):
        count = int(unit_count[i])
        if has_pyramid:
            rs = np.asarray(t.unit_rs.iloc[i], dtype=float)[:count]
            prices = np.asarray(t.unit_entry_prices.iloc[i], dtype=float)[:count]
            times = pd.to_datetime(list(t.unit_entry_times.iloc[i])).values.astype("datetime64[ns]").astype(np.int64)[:count]
            unit_r[i, :count] = rs
            unit_offset[i, :count] = times - entry_ns[i]
            unit_ratio[i, :count] = prices / t.entry_price.iloc[i]
        else:
            unit_r[i, 0] = t.net_r.iloc[i]

    return dict(
        entry_ns=entry_ns, exit_ns=exit_ns, account_code=account, n_accounts=n_accounts,
        risk_pct=t.risk_pct.to_numpy(float), leverage_risk_cap=leverage_risk_cap,
        unit_count=unit_count, unit_r=unit_r, unit_offset_ns=unit_offset, unit_price_ratio=unit_ratio,
    )


@dataclass
class PortfolioMCResult:
    true_median_years: float
    p_reach_ever: float
    p_reach_within_1yr: float
    p_ruin: float
    median_of_reached_only: float
    median_max_dd_pct: float
    avg_skipped_per_path: float
    by_target: dict = field(default_factory=dict)  # populated when extra_targets is used


def run_portfolio(trades: pd.DataFrame, *, account_mode: str = "instrument_side",
                   n_sims: int = 2000, block: int = 20, max_trades: int = 10000, seed: int = 1,
                   start_per_account: float = 100.0, cap: float = 500_000.0, target: float = 1e6,
                   leg_floor: float = 2.0, total_floor: float = 10.0,
                   year_ns: float = 365.25 * 86400 * 1e9, boundary_gap_ns: int = 1,
                   n_threads: int = 0, strategy_name: str = "",
                   leverage_overrides: dict[str, float] | None = None,
                   extra_targets: list[float] | None = None) -> PortfolioMCResult:
    """The one function a new strategy needs to know about. Validates `trades` against the
    canonical schema first (schema.validate_trades raises with every violation if it doesn't
    conform), then runs the block-bootstrap Monte Carlo via dtfx_engine.run_portfolio_mc.
    `extra_targets`, when given, makes one additional identically-seeded call per extra target
    and returns each as its own PortfolioMCResult in `.by_target`. Only valid per-sim when
    n_threads == n_sims (see module docstring) -- this function enforces that by forcing
    n_threads = n_sims whenever extra_targets is non-empty, overriding whatever n_threads was
    passed, rather than silently computing wrong two-stage statistics. This makes
    extra_targets impractical for large n_sims (one OS thread per simulated path); use it for
    small, focused checks, not a production-scale two-stage run.
    """
    if not HAVE_CPP:
        raise ImportError(
            "dtfx_engine is not importable -- build it first with `bash cpp_engine/build.sh`"
        )
    if extra_targets:
        n_threads = n_sims
    trades = validate_trades(trades, strategy_name=strategy_name)
    arrays = to_engine_arrays(trades, account_mode=account_mode, leverage_overrides=leverage_overrides)

    def _call(tgt):
        target_years, max_dd_pct, ruined, processed, accepted, skipped = dtfx_engine.run_portfolio_mc(
            arrays["entry_ns"], arrays["exit_ns"], arrays["account_code"], arrays["risk_pct"],
            arrays["leverage_risk_cap"], arrays["unit_count"], arrays["unit_r"],
            arrays["unit_offset_ns"], arrays["unit_price_ratio"], arrays["n_accounts"],
            n_sims, block, max_trades, seed, start_per_account, cap, tgt, leg_floor,
            total_floor, year_ns, boundary_gap_ns, n_threads,
        )
        stats = true_median_and_stats(target_years, ruined, n_sims)
        return PortfolioMCResult(
            true_median_years=stats["true_median_years"], p_reach_ever=stats["p_reach_ever"],
            p_reach_within_1yr=stats["p_reach_within_1yr"], p_ruin=stats["p_ruin"],
            median_of_reached_only=stats["median_of_reached_only"],
            median_max_dd_pct=float(np.median(max_dd_pct)),
            avg_skipped_per_path=float(np.mean(skipped)),
        )

    result = _call(target)
    if extra_targets:
        result.by_target = {target: result, **{t: _call(t) for t in extra_targets}}
    return result
