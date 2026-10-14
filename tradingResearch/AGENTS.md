# Agent instructions for this repo

This is a monorepo of independent research projects (`dtfxStrategy/`, `coinbaseResearch/`,
`flightDataResearch/`, and others — each its own thing, no shared code between them by
default) **except** for two shared infrastructure packages at the repo root:

- **`quant_engine/`** — the shared Python interface for trade-record validation, position
  sizing, portfolio settlement, and Monte Carlo. Read `quant_engine/docs/ADDING_A_STRATEGY.md`
  before writing or porting ANY backtest in this repo, in any project folder.
- **`cpp_engine/`** — the C++ acceleration layer `quant_engine` wraps. Read
  `cpp_engine/README.md` for what's Active vs. Legacy.

## The rule that matters most

If you are backtesting a trading strategy, validating one, or running a Monte Carlo on trade
results anywhere in this repo: **do not write your own position sizing, margin math, P&L
settlement, or Monte Carlo loop.** Use `quant_engine.portfolio.run_portfolio()`. This applies
regardless of which project folder you're working in — `quant_engine` is intentionally generic,
not tied to `dtfxStrategy`.

This rule exists because duplicating that logic has repeatedly produced real, silently-wrong
results in this codebase's history: a sizing floor built for FX once rejected ~99.85% of gold
trades for "insufficient margin," P&L applied at trade-entry instead of real exit time once
invalidated a batch of published results, two copy-pasted settlement loops silently drifted
from each other, and a lost vendor-specific price cache got silently re-fetched from a
different vendor under the same filename, flipping a strategy's sign. `quant_engine` is the
fix for all four, built and validated specifically because they happened. See
`quant_engine/docs/ADDING_A_STRATEGY.md` for the full rationale and a verified worked example.

## Before you trust or report a number

1. Your trades DataFrame passed `quant_engine.schema.validate_trades()` (or went through
   `run_portfolio()`, which calls it for you).
2. Any Monte Carlo statistic you report came from `quant_engine.portfolio.run_portfolio()`,
   not a hand-rolled loop.
3. If you changed anything under `cpp_engine/src/`, `python3 cpp_engine/validate_portfolio_mc.py`
   is all-PASS before you rely on its output.
4. If you're porting an existing script's already-published numbers, you reproduced them
   (exact match for deterministic paths, ~5-20% relative tolerance for Monte Carlo stats) before
   treating the port as correct.

## Data

Never hardcode a raw price-data file path or guess which vendor a file came from. Use
`quant_engine.data.catalog.Catalog` to resolve `instrument -> (vendor, path)`, and
`quant_engine.data.clean.load_clean()` for the standard bad-day/bad-tick filter.

## Project folders

Each top-level folder other than `quant_engine/` and `cpp_engine/` is an independent research
project (its own strategy, market, or dataset). Work inside one doesn't need to touch, know
about, or be consistent with another — except that any strategy backtest, anywhere, routes
through `quant_engine` as described above.
