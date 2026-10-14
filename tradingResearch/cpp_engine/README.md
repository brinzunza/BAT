# dtfx_engine — C++ backend for the DTFX research pipeline

C++ port of the three pieces almost every script in `research_scripts/` builds on:

| Component | Status | Python original | C++ source |
|---|---|---|---|
| Zone detection (pivots + structure FSM) | Active | `dtfx_robust_search.py::_prep_zones` (pure Python, not JIT'd) | `src/pivots.cpp`, `src/structure_fsm.cpp`, `src/zone_detect.cpp` |
| Backtest engine | Active | `prod_filter_grid_search.py::_simulate_generic` (numba) | `src/backtest.cpp` |
| Monte Carlo engine | **Legacy — frozen** | `goal10_cana_nt_mc_capped.py::run_mc_capped` (numba) | `src/monte_carlo.cpp`, multi-threaded |
| Regime-feature engine (new, 2026-09-04) | Active | none — new construct, no Python original | `src/features.cpp`, multi-threaded |
| Two-account backtest + Monte Carlo (new, 2026-09-15) | **Legacy — frozen** | `goal30_apex_king_deployment.py::simulate_two_account` / `mc_path_two_account`+`run_mc` (pure Python, not JIT'd) | `src/two_account.cpp`, multi-threaded |
| Portfolio Monte Carlo (new, 2026-09-21) | **Active — shared core** | none — new construct, generalizes the two-account model to N accounts | `src/portfolio_mc.cpp`, multi-threaded |

Every strategy/account parameter (stop/target mult, fill age, bounce requirement, touches,
probe/excursion, account start/risk/cap/targets) is a **runtime argument** in all three —
nothing about any specific strategy is precomputed or baked in. Changing a strategy never
requires touching this C++ code, only the arguments passed to it.

### Two-account engine (`two_account.cpp`) — Legacy, frozen

**Status: Legacy — frozen.** `two_account.cpp` and `monte_carlo.cpp`'s `run_mc_capped` remain
exactly as-is because they back already-published results (`goal30_apex_king_deployment.py`,
`goal10_cana_nt_mc_capped.py` and everything downstream of it). Neither gets new work: no new
strategy should build on them. `two_account.cpp`'s fixed two-sub-account model is losslessly
reproduced by `run_portfolio_mc` (below) with `account_mode="side"`, and `run_mc_capped`'s
no-margin, no-exit-time-settlement model is a structural mismatch with the "settle P&L only at
a trade's real exit time" correctness rule the newer engine enforces. Use `run_portfolio_mc`
for any new work; these two stay in place only so old results keep reproducing exactly.

Distinct from `monte_carlo.cpp` above: that one ports a single-account, position-cap-only
model (`goal10_cana_nt_mc_capped.py`). This one ports the TWO separate buy-only/sell-only
$-per-side sub-account model with real per-instrument margin sizing and a max-1-position-per-
side cap — the engine actually used for BEID strategy validation in `research_scripts/
beid_goal_hunt/`, and the module `goal30_apex_king_deployment.py` is wired to delegate to it
automatically (`simulate_two_account`/`run_mc`, with `_python` suffixed originals kept as the
fallback when the extension isn't built). Measured speedup on a real ~26k-trade BEID set:
**~70-90x** for the deterministic backtest, **~400-500x** for the 500-2000-path Monte Carlo
(pure-Python MC runs that took 100-1000+ seconds now take well under a second).

Two real bugs were found and fixed while building and validating this port (both affect the
Python original too, not just the C++ port):
1. **Non-stable sort**: `simulate_two_account`/`run_mc` re-sort `trades` by `entry_time`
   internally without `kind="stable"`. 1-minute-bar trade sets routinely have tied
   `entry_time` values (~16% of rows in a representative BEID set), and pandas' default
   quicksort doesn't guarantee a consistent tie order — so two calls with the *same* input
   data could silently produce different results. Fixed with `kind="stable"` in both the
   Python original and (necessarily, to match) the C++ port's own sort key.
2. **Heap tie-break**: Python's `heapq` on `(exit_ns, risk_dollars, margin, r, side)` tuples
   breaks ties in exit time by comparing the later tuple fields; the C++ `std::priority_queue`
   needed an explicit comparator doing the same (simultaneous exits are also routine at 1-min
   granularity, ~11% of rows) to match exactly rather than resolve ties arbitrarily.

Validate with `python3 validate_two_account.py` (uses real BEID trade data, not synthetic) —
exact match required for the deterministic backtest (down to the per-event equity trace),
statistical agreement for the Monte Carlo (different RNG by construction, same convention as
`monte_carlo.cpp`).

### Portfolio Monte Carlo engine (`portfolio_mc.cpp`) — Active, shared core

**Status: Active.** This is the general-purpose simulation primitive new strategy work should
build on, wrapped by `quant_engine.portfolio.run_portfolio()` rather than called directly.
Unlike `two_account.cpp`'s hardcoded two-sub-account, three-instrument-aware model, this engine
has **zero instrument-specific code**: every account's margin, leverage cap, and per-unit
pyramid sizing is computed entirely in Python (`quant_engine.sizing`) and passed in as plain
arrays, so adding a new instrument or a new account grouping never touches C++.

Capabilities:
- **N arbitrary accounts** (not fixed at two) via a caller-supplied `account_code` per trade —
  `quant_engine.portfolio` dispatches this as `account_mode` ∈
  `{"pooled", "instrument", "side", "instrument_side"}`.
- **Real per-unit pyramid margin**, checked at each add's own timestamp (`unit_r`/
  `unit_offset_ns`/`unit_price_ratio`, `[n_trades, max_units]` arrays) — not just an aggregate
  per-trade margin figure.
- **Event-driven settlement**: P&L and freed margin apply only at a trade's real exit time (a
  time-ordered heap of pending exits), never at entry — required for correct margin
  availability at high trade-overlap.
- **Two-tier ruin**: a per-account `leg_floor` and a portfolio-wide `total_floor`, both checked
  after every settled exit.
- **Block-bootstrap Monte Carlo** with one `std::mt19937_64` per **thread** (not per sim),
  seeded via `seed_seq{seed, thread_idx}` — see the constraint below before relying on
  per-sim RNG independence.

Signature (`src/bindings.cpp`):
```python
dtfx_engine.run_portfolio_mc(entry_ns, exit_ns, account_code, risk_pct, leverage_risk_cap,
                              unit_count, unit_r, unit_offset_ns, unit_price_ratio,
                              n_accounts, n_sims, block, max_trades, seed, start_per_account,
                              cap, target, leg_floor, total_floor, year_ns, boundary_gap_ns,
                              n_threads)
    -> (target_years, max_dd_pct, ruined, processed, accepted, skipped)
```

**Two-stage `target1`/`target2` is deliberately not implemented in C++.** Because a path's
block draws don't depend on `target` (only the early-stop check does), two separate calls with
identical `seed`/`block`/`max_trades` and different `target` are exact continuations of the
same simulated paths — *but only when each simulated path has its own independent RNG stream*,
i.e. only when `n_threads == n_sims` (one thread per sim). With fewer threads than sims, a
thread's single persistent RNG keeps advancing across every sim it processes, so a lower target
(which makes a path stop, and therefore stop consuming random draws, earlier) causes the two
calls' streams to diverge after the first sim on each thread — confirmed with real
counter-examples (`n_threads=1`) in `validate_portfolio_mc.py::validate_target_invariance`
before this constraint was documented. `quant_engine.portfolio.run_portfolio(...,
extra_targets=[...])` gets two-stage tracking by making a second identically-seeded call, and
enforces `n_threads = n_sims` whenever `extra_targets` is used — which makes it a tool for
small, focused two-stage checks, not a general substitute for `run_mc_capped`'s native
two-stage tracking at the thousands-of-sims scale production runs actually use.

Validate with `python3 validate_portfolio_mc.py` (two real datasets: a BEID zone-fade set with
no pyramid columns, and a Donchian-pyramid gold+BTC set with real `unit_rs`/
`unit_entry_times`) — exact match for the deterministic single-path replay, statistical
agreement for the Monte Carlo, plus the target-invariance and incremental-pyramid-margin checks
described above. All checks pass as of the last run; measured speedup 100-1100x over the
pure-Python reference depending on dataset and check.

`avg_skipped_per_path` (a diagnostic-only summary stat) disagrees numerically between the
Python reference and the C++ engine (e.g. ~300 vs ~700 on the Donchian set) despite every
outcome-critical statistic (`true_median`, reach%, ruin%) agreeing — most plausibly a
definitional difference in what each side's skip-counter counts rather than a functional bug.
Documented here rather than silently tightened or hidden; worth a closer look before leaning on
this specific number.

**Fixed incident (2026-09-28): int64 nanosecond overflow in the simulated clock.** The worker's
`clock`/`et`/`xt` are `int64_t` nanoseconds anchored to real epoch-0 entry timestamps and grow
without bound as the simulated clock advances block by block — a strategy needing longer than
`int64_t`'s ~292-year range to reach its target silently wrapped `clock` to a garbage value,
which `settle_through` could then read as spuriously satisfying `total >= target` at a
fabricated (sometimes sub-1-year, sometimes negative) "years" result. Confirmed on a real
241-trade small-sample FX config that needed ~572 simulated years: a from-scratch Python
reimplementation and a hard lower bound on elapsed time (inter-block gaps alone) both proved the
reported <1-year result was impossible; a debug trace of the actual event sequence showed
`event.time_ns` going negative and wrapping through zero at the point of failure. Fixed by
detecting the real signed-overflow condition on every timestamp addition (`__builtin_add_overflow`
/`__builtin_sub_overflow`) and treating overflow as "target not reached" rather than an arbitrary
year-count cutoff — this can only ever correct an already-broken path to `-1.0`, never change a
path that stays within the safe range. Re-validated all-PASS via `validate_portfolio_mc.py`.

## Build

```bash
bash build.sh
```

Compiles with CMake + pybind11 (both already installed for this project — no new
dependencies) and copies the resulting extension into `../dtfxStrategy/research_scripts/`, so
any script there can do:

```python
import dtfx_engine
```

exactly like it already does `import dtfx_robust_search as rs`.

Re-run `build.sh` after any change under `src/`.

## Validate

```bash
python3 validate.py
```

Cross-checks every function against the existing Python/numba implementations on real
instrument data (EURUSD/GBPUSD/EURGBP/USDJPY):

- **Zone detection** and **backtest engine** are pure deterministic functions of their
  inputs — validated with exact array equality. Any mismatch here is a real bug.
- **Monte Carlo** uses `std::mt19937_64`, not numba/NumPy's Mersenne Twister, so it will
  never bit-match path-for-path. Validated statistically instead: percentile/summary
  agreement at 100k sims (median/P5/P95 years-to-$1M, median drawdown, %ruined) within a 5%
  relative tolerance.

Run this after any change to `src/` before trusting the C++ output.

## Performance

Measured on EURUSD (6.3M bars, Hybrid-Ultra config, `benchmark.py`), this Mac (8 cores):

| Component | Python/numba | C++ | Speedup |
|---|---|---|---|
| Zone detection | 3.3s (never JIT'd) | 0.05s | ~68x |
| Backtest engine | 350-430ms (numba, warm) | ~90ms (8 threads) | ~4.7x |
| Monte Carlo (25k sims) | 0.36-0.39s (numba, 1 thread) | 0.06-0.07s (8 threads) | ~5.6x |

Zone detection's win is almost entirely "was never JIT-accelerated, now is." The backtest
engine and Monte Carlo were already numba-JIT'd (near-native speed per-thread already, so a
straight single-threaded C++ port alone lands close to parity, not dramatically faster) — the
real gain there comes from parallelizing across zones/sims with `std::thread`, something the
single-threaded numba originals don't do. Run `python3 benchmark.py` to reproduce on your
machine (core count changes the last two rows).

## API

```python
detect_zones(open, high, low, close, times_ns, min_retracement)
    -> (top, bottom, mid, is_buy, created_at_ns, start_idx)

simulate_generic(zone_top, zone_bottom, zone_mid, zone_is_buy, zone_start_idx,
                  zone_created_ns, times_ns, lows, highs, min_fill_age_ns,
                  stop_mult, target_mult, require_bounce, probe_depth, exc_mult,
                  max_touches, max_window)
    -> (touch_num, entry_idx, is_win, mid, stop_risk)

run_mc_capped(net_r, n_sims, block, max_trades, seed,
              start=200.0, risk_frac=0.05, cap=2_000_000.0,
              target1=1e6, target2=1e7, ruin_floor=1.0, n_threads=0)
    -> (trades_to_1m, trades_to_10m, max_drawdown, ruined)

compute_regime_features(high, low, close, window=500, stride=50,
                         vol_lookback_points=200, n_threads=0)
    -> (efficiency_ratio, hurst_est, vol_percentile, realized_vol)
```

`compute_regime_features` is new (2026-09-04, `/goal` — hyper-creative synthetic
indicators): per-bar market-regime features used as zone-touch GATES, not zone-detection
inputs, by `research_scripts/goal62_regime_feature_strategies.py`. Recomputed only every
`stride` bars from the trailing `window` bars, forward-filled between grid points.
Multi-threaded (one thread per contiguous output range); validated exact-match against a
pure-Python reference (see `validate.py`) — 6.34M EURUSD bars in 0.13s. See
`REALISTIC_FILL_RANKINGS.md`'s "Hyper-creative synthetic indicators" section for what each
of the three outputs means and the strategies built on them.

All arrays are numpy; signatures mirror the Python originals as closely as possible so
they're near drop-in replacements.

## Integration

`dtfx_robust_search.py::_prep_zones`, `prod_filter_grid_search.py::_simulate_generic`, and
`goal10_cana_nt_mc_capped.py::run_mc_capped` each try `dtfx_engine` first and fall back to
their original numba/pure-Python implementation (kept in the same file, suffixed `_numba`/
`_py`) if the compiled extension isn't present. Every script downstream of these three —
which is most of the active research chain (`goal5` → `goal9`/`10`/`11`/`12`/`13` →
`STRATEGY_RANKINGS.md`) — gets the C++ engine automatically, with no per-script changes.

## Why not port the other ~17 one-off `@njit` zone simulators too?

Older audit/research scripts each wrote their own numba variant for a specific,
already-answered question (asymmetric-target studies, penetration-depth audits, max-age
audits, etc.). `simulate_generic`'s knobs (asymmetric stop/target, bounce-gating, touches,
window) already cover what most of them varied; new research should build on it rather than
spinning up another one-off numba function. Those old scripts still run exactly as before —
nothing about them changed.
