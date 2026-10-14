# Adding a strategy to `quant_engine`

Read this before writing a new backtest, porting an existing one, or validating a strategy's
results anywhere in this repo. It is the one thing a new strategy needs to know about.

## The one rule

**Never hand-roll position sizing, margin math, P&L settlement timing, or Monte Carlo.** Use
`quant_engine` for all of it. This isn't a style preference — every one of those categories has
already produced a real, silent bug in this codebase when duplicated ad hoc:

- A whole-unit sizing floor built for FX, reused for gold, once rejected ~99.85% of candidates
  for "insufficient margin" that was actually a sizing-floor artifact.
- P&L applied at trade entry instead of real exit time (a lookahead bug) once invalidated a
  large chunk of results.
- Two near-identical settlement loops (copy-pasted between scripts) drifted: one silently
  stopped honoring an `account` override column the other still respected.
- A lost vendor-specific price cache got silently re-fetched from a different vendor under the
  same filename, producing a strategy-profitability sign flip.

`quant_engine` exists specifically so none of that can happen again. Signal generation (finding
entries/exits) is still all yours, per-strategy, in whatever language/logic fits — only the
**output** needs to conform to the schema below, and only sizing/settlement/MC route through
this package.

## The trade-record schema

Your signal-generation code's only job is to produce a `pandas.DataFrame` with these columns.
`quant_engine.schema.validate_trades()` checks it before anything else runs.

**Required:**

| column | dtype | meaning |
|---|---|---|
| `instrument` | str | e.g. `"EURUSD"`, `"XAUUSD"` |
| `is_buy` | bool | long or short |
| `entry_time` | datetime64[ns] | UTC-naive |
| `exit_time` | datetime64[ns] | UTC-naive, `>= entry_time` |
| `entry_price` | float64 | |
| `stop_risk` | float64 | distance from entry to stop, in price units, `> 0` (this is the standardized name — some older scripts call it `stop_dist`) |
| `net_r` | float64 | realized P&L in R-multiples (this trade's return / its own risk) |

**Caller-supplied metadata:**

| column | dtype | meaning |
|---|---|---|
| `risk_pct` | float64 | fraction of equity risked per trade, `(0, 1]` |
| `account` | str | sizing-key override; defaults to `instrument` when absent |

**Optional, pyramid strategies only** (all four below or none — partial sets are rejected):

| column | meaning |
|---|---|
| `units` | number of pyramid adds for this trade |
| `unit_rs` | list, one R-multiple per unit |
| `unit_entry_times` | list, one datetime per unit; `unit_entry_times[0]` must equal `entry_time` |
| `unit_entry_prices` | list, one price per unit |

Any other column you add (`forced`, `touch_num`, whatever your strategy needs for its own
analysis) passes through untouched — the schema only constrains the columns above.

## The four steps

```python
import sys
sys.path.insert(0, "/Users/brunoinzunza/Documents/GitHub/tradingResearch")  # repo root

from quant_engine import Catalog, load_clean
from quant_engine.portfolio import run_portfolio

# 1. Get clean data.
cat = Catalog.build()
entry = cat.lookup("EURUSD")               # vendor-tagged, deepest-history file for this instrument
df = load_clean(entry.path, asset_class=entry.asset_class)

# 2. Generate signals -- this part is entirely yours.
trades = my_strategy.generate_trades(df)    # must return the schema above

# 3. Validate (run_portfolio does this for you too, but fail fast during development).
from quant_engine.schema import validate_trades
trades = validate_trades(trades, strategy_name="my_strategy")

# 4. Run the shared Monte Carlo.
result = run_portfolio(
    trades,
    account_mode="instrument_side",         # see table below
    n_sims=2000, block=20, seed=1,
    start_per_account=100.0, cap=500_000.0, target=1_000_000.0,
    strategy_name="my_strategy",
)
print(result.true_median_years, result.p_reach_ever, result.p_ruin)
```

That's the whole interface. `run_portfolio()` validates the schema, converts your DataFrame to
the arrays the C++ engine needs, runs the block-bootstrap Monte Carlo, and returns a
`PortfolioMCResult` (`true_median_years`, `p_reach_ever`, `p_reach_within_1yr`, `p_ruin`,
`median_of_reached_only`, `median_max_dd_pct`, `avg_skipped_per_path`).

## `account_mode`

Controls how trades are grouped into separate C++ "accounts" (each with its own equity, own
margin, own ruin check):

| mode | grouping | when to use |
|---|---|---|
| `"pooled"` | everything, one account | simplest case; no per-instrument or per-side isolation |
| `"instrument"` | one account per `instrument`/`account` value | independent capital per instrument |
| `"side"` | one account per `is_buy` | separate buy-only / sell-only books |
| `"instrument_side"` | one account per (`instrument`/`account`, `is_buy`) | the default for most real strategies here — matches how `margin_aware_mc.py` keys equity |

`start_per_account` is the starting equity **per account**, not the portfolio total — if you
want a $200 portfolio split across 4 accounts, pass `start_per_account=50.0`.

## Sizing

`quant_engine.sizing.size_and_margin(instrument, equity, entry_price, stop_risk, risk_pct, ...)`
computes `(units, risk_dollars, margin)` from `INSTRUMENT_REGISTRY`
(`quant_engine/sizing.py`). Most strategies don't need to call this directly — `run_portfolio`
handles margin internally via each instrument's registered `max_leverage`. Add a new
instrument by adding an `InstrumentSpec` to `INSTRUMENT_REGISTRY`; leave `whole_unit_floor`
at its default (`False`) unless you are specifically reproducing OANDA's live minimum-order
rounding (`live_parity=True`) — fractional, floor-free sizing is the deliberate default for
every instrument, including FX, and is what backtest correctness actually requires.

## Data

`quant_engine.data.catalog.Catalog` resolves `instrument -> (vendor, path, pip_size,
asset_class)` from the seed registry in `catalog.py` — never guess a raw file path yourself;
a prior vendor mixup (see "The one rule" above) is exactly what this prevents.
`quant_engine.data.clean.load_clean()` is the one bad-day/bad-tick filter every script should
use instead of rolling its own threshold.

## Validating a new strategy's results

- If you're **porting** an existing script's already-published numbers onto `run_portfolio()`,
  the bar is reproducing those numbers: exact match for anything deterministic, ~5-20% relative
  tolerance for Monte Carlo statistics (different RNG by construction). See
  `cpp_engine/validate_portfolio_mc.py` for the reference pattern — real trade data in, a
  pure-Python reference path compared against the C++ path, deterministic + statistical checks.
- If you're validating a **brand-new** strategy, call `run_portfolio(..., n_sims=1, block=len(trades))`
  first — this runs exactly one simulation over your trades in their real chronological order
  (no resampling), giving you the actual historical equity path and ruin/reach outcome before
  you trust any Monte Carlo statistic built on top of it.
- Before touching `cpp_engine/src/*.cpp`, re-run `python3 cpp_engine/validate_portfolio_mc.py`
  after your change — it must stay all-PASS.

## Active vs. Legacy engines

`run_portfolio_mc` (wrapped by this package) is the **Active** engine — it has zero
instrument-specific C++ code and supports N arbitrary accounts. `two_account.cpp` and
`run_mc_capped` are **Legacy — frozen**: they back already-published results and stay exactly
as-is, but no new strategy should build on them. Full detail in `cpp_engine/README.md`.

## Worked example

This exact script runs end-to-end (verified against real EURUSD data as part of writing this
guide — not a hypothetical):

```python
import sys
sys.path.insert(0, "/Users/brunoinzunza/Documents/GitHub/tradingResearch")

import numpy as np
import pandas as pd
from quant_engine import Catalog, load_clean
from quant_engine.portfolio import run_portfolio

cat = Catalog.build()
entry = cat.lookup("EURUSD")
df = load_clean(entry.path, asset_class=entry.asset_class)
df = df.loc["2020-01-01":"2020-06-30"]

# Toy signal: go long at 00:00 each day, exit 12 hours later. Fixed 20-pip stop.
daily = df.between_time("00:00", "00:01")
daily_opens = daily.groupby(daily.index.date).first()

rows = []
for date, row in daily_opens.iterrows():
    entry_time = pd.Timestamp(date)
    exit_time = entry_time + pd.Timedelta(hours=12)
    if exit_time not in df.index:
        continue
    entry_price = row["close"]
    exit_price = df.loc[exit_time, "close"]
    stop_risk = 20 * entry.pip_size
    net_r = (exit_price - entry_price) / stop_risk
    rows.append(dict(
        instrument="EURUSD", is_buy=True, entry_time=entry_time, exit_time=exit_time,
        entry_price=entry_price, stop_risk=stop_risk, net_r=net_r, risk_pct=0.01,
    ))

trades = pd.DataFrame(rows)
trades["entry_time"] = pd.to_datetime(trades["entry_time"])
trades["exit_time"] = pd.to_datetime(trades["exit_time"])

result = run_portfolio(trades, account_mode="pooled", n_sims=500, block=10, seed=1,
                        start_per_account=200.0, cap=50_000.0, target=1_000_000.0,
                        strategy_name="toy_daily_open")
print(result)
```

(This particular toy signal has no real edge — it's here to demonstrate the interface, not as
a strategy worth trusting. A real strategy's `generate_trades` step is the part you bring.)
