# dtfx_engine — C++ backend for the DTFX research pipeline

C++ port of the three pieces almost every script in `research_scripts/` builds on:

| Component | Python original | C++ source |
|---|---|---|
| Zone detection (pivots + structure FSM) | `dtfx_robust_search.py::_prep_zones` (pure Python, not JIT'd) | `src/pivots.cpp`, `src/structure_fsm.cpp`, `src/zone_detect.cpp` |
| Backtest engine | `prod_filter_grid_search.py::_simulate_generic` (numba) | `src/backtest.cpp` |
| Monte Carlo engine | `goal10_cana_nt_mc_capped.py::run_mc_capped` (numba) | `src/monte_carlo.cpp`, multi-threaded |
| Regime-feature engine (new, 2026-09-04) | none — new construct, no Python original | `src/features.cpp`, multi-threaded |

Every strategy/account parameter (stop/target mult, fill age, bounce requirement, touches,
probe/excursion, account start/risk/cap/targets) is a **runtime argument** in all three —
nothing about any specific strategy is precomputed or baked in. Changing a strategy never
requires touching this C++ code, only the arguments passed to it.

## Build

```bash
bash build.sh
```

Compiles with CMake + pybind11 (both already installed for this project — no new
dependencies) and copies the resulting extension into `../research_scripts/`, so any script
there can do:

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
