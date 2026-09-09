# Causal DTFX exploratory research

Current status: **no verified leaderboard winner**. Goal remains open.
Research results: `trading-data/causal_discovery/RESEARCH_REPORT.md`.
Read `PROTOCOL.md` before modifying the experiment or opening held-back results.

Build and test:

```sh
python3 build.py
python3 test_native.py
```

Run from this directory. `prepare.py` builds binary opportunity tables and causal
features; `search.py` performs discovery/risk selection followed by validation.
The tables contain later outcomes, but search explicitly excludes all rows on/after
2018-03-09 before any fitting/scoring. `evaluate.py` requires a frozen candidate.

```sh
# Causal zone availability with legacy fills: comparison only, NOT a valid winner.
python3 prepare.py
python3 search.py

# Bid/ask narrow-target execution audit.
CAUSAL_EXECUTION=1 python3 prepare.py
CAUSAL_EXECUTION=1 python3 search.py

# Different completed-candle structural entry mechanisms.
CAUSAL_STRUCTURAL=1 python3 prepare.py
CAUSAL_STRUCTURAL=1 python3 search.py
CAUSAL_STRUCTURAL=1 python3 learned_search.py

# Scale transfer. Use 5, 15 or 60, each in its own output directory.
CAUSAL_STRUCTURAL=1 CAUSAL_TIMEFRAME=15 python3 prepare.py
CAUSAL_STRUCTURAL=1 CAUSAL_TIMEFRAME=15 python3 search.py
```

Do not rerun selection over an already evaluated frozen candidate: preserve its
version and record a new research round. Generated `core/` is an isolated copy of
the original C++ sources with confirmation timestamps added; original production
and research engines are not changed. `research.cpp` implements causal indicators,
as-of joins, accounting, stricter execution, aggregation, and histogram learning.
Python handles I/O and result tables. No broker APIs are used.

Additional commands:

```sh
python3 market_cache.py
python3 test_nav.py
python3 pooled_search.py
python3 nav_validation.py
python3 basis_search.py
CAUSAL_PATH=0 python3 reconstruct_confluence.py
CAUSAL_PATH=1 python3 reconstruct_confluence.py
```

`nav_account()` now implements marked-to-market sizing, causal FX conversion and
boundary-position valuation for next-open structural signals. `account()` retains
the legacy balance-based screening convention. `path_trades()` implements two explicit
OHLC event paths for limit fills, cancellation and exits. Remaining limits include
unobserved intraminute order, latency, and pending-order reservation. First-stage
metrics do not establish a winner. See the updated research report for full results.
