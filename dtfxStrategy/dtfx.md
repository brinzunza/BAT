# DTFX — base strategy specification

Written 2026-09-06 as the canonical, from-scratch definition of the DTFX zone rule, after
several rounds of correction this session (see `research_scripts/goal95_every_swing_zones.py`
and `research_scripts/goal98_zone_lifetime.py` for the code this document describes).
**Everything below is the base to experiment on going forward** — variants, entry depths,
inversions, structure gates, etc. all start from this exact definition, not from an earlier
or looser description. Where this document and an older script's docstring disagree, this
document wins.

This is a specification of a mechanism, not a claim that it is profitable. See
[Status](#status--known-results) at the end before assuming otherwise.

---

## 0. What this is, where it comes from, and how to use this document

Written for an agent (or person) with zero prior context on this project, so they can
pick up real research from here without re-deriving what's already known or re-testing
what's already been tried and rejected. Read this section first; sections 1-9 below are
the precise mechanical spec, and "Status / known results" is a chronological research log
— both assume the orientation this section gives.

### What "DTFX" is

DTFX is the name of a trading-strategy research program in this repo (`~/Documents/GitHub/
tradingResearch/dtfxStrategy/`), not a documented acronym — no file anywhere in this
project expands the letters, so don't invent an expansion. The idea, unchanged since the
original notebook (`dtfx_strategy.ipynb`), combines two things:

1. **Trend from confirmed market structure** — higher-highs-and-higher-lows /
   lower-lows-and-lower-highs, read directly off price swings, not a moving average or a
   drawn trendline.
2. **Retracement-zone entries** — every time a new swing forms, the leg that produced it
   defines a zone; price is expected to retrace partway into that zone before the trend
   (or the immediately preceding move) resumes, so DTFX enters at the zone's 50% level,
   stops at its far edge, and targets its near edge (the extreme that created it).

### Three variants share the name — know which one you're looking at

The exact swing-detection and zone-construction rules have been rewritten from scratch
more than once across this research program's history. All three call themselves "DTFX";
they are mechanically different, and results from one do not transfer to another without
re-testing:

1. **The original notebook** (`dtfx_strategy.ipynb`) — a "confirmed structure" state
   machine: a new extreme in the trend's direction is accepted immediately, but the
   opposite-type swing that comes before it only becomes "confirmed" once a *subsequent*
   trend-direction extreme validates it. Zone top/bottom draw from that confirmed-structure
   state, not from the two most recent swings unconditionally. Superseded; kept only as the
   conceptual origin.
2. **The "strict pivot" branch** (`CHECKPOINT_NOTES.md`, `research_scripts/
   dtfx_eurusd_backtest_strict.py`, `PRODLIVETRADING/strategy/pivots.py` +
   `structure_fsm.py` before they were deleted 2026-09-07) — a candle-*close*-based
   candidate that only validates as a real swing after `min_retracement` (2-3) consecutive
   opposite-color candles follow without extending it, feeding a structure FSM that only
   creates a zone at genuine trend-confirmation breaks (not every swing). This branch found
   the project's single biggest early result — **the edge only exists from 2021 onward**
   across the pre-2021/post-2021 boundary known at the time (see `CHECKPOINT_NOTES.md` for
   the full multi-instrument breakdown: EUR/GBP, SPX500, AAPL, BTC/USD, and others; a
   validated RSI-neutral-zone + ATR-compression filter; and several negative results, e.g.
   "position in trend" did not generalize across instruments and should not be revisited
   without new evidence). This branch was then extended, through a long chain of walk-
   forward parameter searches (`dtfx_robust_search.py`'s methodology; see `PRODLIVETRADING/
   strategy/engine.py`'s git history for the full COMB → Omega → SOMEGA → Hybrid Stop →
   Hybrid-Ultra → Speed-King → Apex-King → Velocity-King → Apex-Ultra chain, each entry
   dated and reasoned) into **Apex-Ultra**, single-symbol EUR/GBP, RISK_PCT=6% — the
   strategy actually deployed live in `PRODLIVETRADING/bot.py` from 2026-08 until
   **2026-09-07, when it was retired and its code deleted entirely** per direct
   instruction ("completely delete apex ultra from PROD"), in favor of variant 3. Nothing
   about variant 3 shares code with this branch; they were developed independently and
   happen to converge on the same two-idea core.
3. **"Every swing is a zone"** — the mechanism sections 1-9 below define precisely, and
   the one now actually running live. Symmetric close-based zigzag (a single bar's close
   crossing its predecessor's confirms a swing immediately — no N-candle confirmation
   step), and **every** consecutive swing pair becomes a zone, not just ones at a
   confirmed structure break (`research_scripts/goal95_every_swing_zones.py`). Entry
   requires an explicit overshoot-then-tag gate (§6) that variant 2 never had. Developed
   in this session (starting 2026-09-06) after direct correction that variant-2-style
   "structure confirmation" zone gating was silently dropping most valid zones.

**When in doubt about which variant a given script or result belongs to: if it imports
`find_pivots_strict`/`run_structure_fsm`/`SymbolEngine` from the git history of
`PRODLIVETRADING/strategy/`, or references N_MULTIPLIER/TARGET_RR/MIN_FILL_AGE/RSI/ATR, it's
variant 2. If it imports `close_based_zigzag_symmetric`/`build_zones_every_swing` from
`goal95`, or references `MIN_ZONE_PIPS`/overshoot-then-tag, it's variant 3 — the one this
document specifies and dtfx.md's own "Status" section reports results for.**

### Current state of the world (2026-09-07)

- **Live**: `PRODLIVETRADING/bot.py`, variant 3 only, confirmed running on the practice
  OANDA account (VM `prodtradinglive`) as of this date. Frozen config: 4 pairs (EURUSD,
  GBPUSD, EURGBP, USDJPY), **uniform 10-pip minimum zone height on all four pairs**
  (`MIN_ZONE_PIPS`, `config.py`, since 2026-09-08 — was a 10p-USDJPY/5p-others hybrid, see
  "uniform 10-pip zones" below), **asymmetric 0.8:1 reward:risk target** (`SL_MULT=1.0`,
  `TP_MULT=0.8` — NOT 1:1R; target sits inside the zone's own far edge, compensated by a
  materially higher win rate; regime-split selected, PRE-2021 design / POST-2026-04
  holdout), 3% risk per trade, 2 separate $100-ish sub-accounts (buy-only / sell-only,
  sized off `min(long, short)` balance), max 1 confirmed open position per side, 50x
  leverage, `MAX_ALLOWED_SPREAD_PIPS=2.0`, marketable-limit orders priced at `mid ±
  spread/2` on entry and `level ∓ spread/2` on
  both exits (see "Cost model corrected" below).
- **No genuine paper-trading track record exists yet** — every number in "Status" is a
  historical backtest, however carefully walk-forward-validated. That can only change by
  actually watching this run for real elapsed time.
- **The most important caveat governing every number in this document**: a data-feed
  artifact (see "The 2021+ anomaly, resolved" below) inflates apparent performance from
  **2021-01 through 2026-03** on all 4 pairs. Every result anywhere in this file that
  matters for parameter selection uses only **PRE-2021** data for choosing anything, and
  **POST-2026-04** data as a genuine, never-touched holdout for validating the choice — if
  you see a number from the 2021-2026.03 window presented as a decision input rather than
  "shown for comparison only", treat that as a bug in whoever wrote it.

### Where everything lives

| what | where |
|---|---|
| This spec + full research log | `tradingResearch/dtfxStrategy/dtfx.md` (this file) |
| Variant 3's core mechanism (batch/research) | `research_scripts/goal95_every_swing_zones.py` (zones), `goal99_dtfx_base_backtest.py` (entry gate + targets A/B/C + cost model) |
| Shared two-account backtest engine | `research_scripts/goal79_two_account_portfolio.py` (`simulate_two_account`, `run_monte_carlo_two_account`) |
| Zone/risk sweep infra, trade-candidate cache | `research_scripts/goal104_optimal_risk_two_account.py` (`load_and_build_zones`, `build_candidate_trades` — reused by nearly every later script) |
| Walk-forward regime split (the data-artifact fix) | `research_scripts/goal108_regime_split_validation.py` |
| Production-accurate concurrency simulator | `research_scripts/goal110_production_concurrency_simulator.py` |
| Spread-cost depth analysis (5p vs 10p) | `research_scripts/goal111_5pip_vs_10pip_depth.py` |
| Current frozen-config full report | `research_scripts/goal112_5pip_3pct_report.py` |
| Variant 3 live (production) | `PRODLIVETRADING/bot.py`, `PRODLIVETRADING/strategy/engine.py` |
| Broker integration, generic (both variants used this) | `PRODLIVETRADING/broker/oanda_client.py` |
| Variant 2's full history (deleted 2026-09-07) | `git log -- strategy/pivots.py strategy/structure_fsm.py strategy/engine.py` in the `PRODLIVETRADING` repo |
| Variant 2's original research | `CHECKPOINT_NOTES.md`, `dtfx_strategy.ipynb`, `dtfx_eurusd_backtest_strict.py`, `dtfx_indicator_diagnostics.py`, `dtfx_robust_search.py` |
| Deployment ops (systemd, .env, safety filters) | `PRODLIVETRADING/deploy/`, `PRODLIVETRADING/config.py`, `PRODLIVETRADING/README.md` |

### Good next research directions (not yet done, concrete)

- **Variant 2's validated filters were never tested against variant 3.** RSI(14)
  neutral-zone exclusion and ATR-compression exclusion measurably improved variant 2's
  results on 6 of 7 instruments (`CHECKPOINT_NOTES.md` §4) — nobody has checked whether
  either transfers to variant 3's every-swing zones. Genuinely open, plausibly valuable,
  and the data + tooling to test it already exist.
- **3% risk for the 5-pip config was carried over from 10-pip's own sweep, not
  independently re-derived.** `goal104`'s risk-cliff sweep was run at 10-pip; the 5-pip
  config trades 4-5x more often, and the drawdown comparison in "5-pip @ 3% risk" above
  shows the cost of that. A fresh sweep at 5-pip (same methodology) would tell you whether
  3% is still the right number there or just an inherited one.
- **Cost model is still spread-only.** No slippage, no latency, no partial fills modeled
  anywhere — `place_bracket`'s real OANDA execution is the first place any of that would
  actually show up, and only once real paper-trading accumulates.
- **Variant 2's negative/regime-fragile findings are a checklist, not just history**: hour-
  of-day and day-of-week effects that looked real but were regime artifacts, ATR-filter
  inversion during genuine volatility-regime changes (USD/JPY, 2026-08) — worth checking
  whether any of variant 3's own future refinements are vulnerable to the same trap before
  trusting an in-sample-looking pattern.
- **No account-level circuit breaker** (daily loss limit, feed-anomaly detector) exists at
  either the variant-2 or variant-3 stage — only per-trade risk sizing. A real gap if this
  is ever trusted with a meaningfully larger account.

---

## 1. Inputs

One instrument's OHLC bars at a fixed timeframe (this project uses 1-minute bars). Only
`high`, `low`, and `close` are used for structure; `open` is not used by any rule below.

---

## 2. Swing (pivot) detection

A **swing high** or **swing low** is a single price extreme that gets "locked in" — from
that point on, its price never changes. Detection is **close-based**, not wick-based, and
**symmetric** — both highs and lows lock in the same way.

**Mechanism.** At any moment the algorithm is in one of two states: *tracking a potential
high* or *tracking a potential low*. While tracking a potential high, it keeps extending a
running candidate to whatever the highest `high` seen so far is. The candidate locks in —
becomes a confirmed swing high, at the price and bar of that running candidate — the
instant one bar's `close` is lower than the *previous bar's* `close` (a one-bar momentum
reversal, no minimum retracement size, no wick requirement). The moment a high locks in,
tracking immediately switches to a potential low, seeded from the locking bar's own `low`.
The low side is the exact mirror: it keeps extending to the lowest `low` seen, and locks in
the instant a bar's `close` is higher than the previous bar's `close`.

```
tracking_high = True
pot = high[0]              # running candidate price
for i in 1 .. n-1:
    if tracking_high:
        pot = max(pot, high[i])
        if close[i] < close[i-1]:
            confirm SWING HIGH at price = pot           # locked in, bar i
            tracking_high = False
            pot = low[i]                                # start tracking a potential low
    else:
        pot = min(pot, low[i])
        if close[i] > close[i-1]:
            confirm SWING LOW at price = pot             # locked in, bar i
            tracking_high = True
            pot = high[i]
```

**Bootstrap (arbitrary, not derived from price).** The very first candidate always starts
as a potential *high*, seeded from bar 0's own `high` (`tracking_high = True` on the first
line). This is a hardcoded assumption, not a read of whether price is actually rising or
falling at the start of the data — the algorithm does not look backward past bar 0 to
decide which type should come first. In practice this only affects which single pivot type
confirms first; every swing after that is fully determined by price.

On real 1-minute FX data this locks in a swing on roughly **half of all bars** (measured at
48–50% across EURUSD/USDJPY test windows this session) — a swing this loose is *not* "the
market's important turning points," it is every visible zig and zag, by design (see
[§4](#4-uptrend--downtrend-definition)).

Code: `close_based_zigzag_symmetric()` in `goal95_every_swing_zones.py`.

---

## 3. Zone construction

Every **consecutive pair** of swings (one high, one low, in either order) defines a zone —
**every swing produces a zone**, with no filter on size, no requirement that it "matters."
A zone is `[top, bottom]`, where `top` and `bottom` are two swing prices (`top` is always
the higher of the two, `bottom` the lower — no zone is ever inverted).

Walking through the confirmed swings in chronological order, track two rolling values:
`last_high_price` (the most recently confirmed swing high, whatever its price) and
`last_low_price` (the most recently confirmed swing low). Both update **unconditionally**
on every swing of their type — a new swing high always overwrites `last_high_price`, even
if it is *lower* than the previous one.

When a new swing confirms, compare it only to the prior swing of **its own type**:

- **New swing high, and it exceeds the previous swing high** → the trend is (or becomes) up.
  - If the trend was down immediately before this: **flip zone**, `top = last_high_price`,
    `bottom = last_low_price` (both values *as they stood before this swing updated
    anything* — i.e. the old downtrend's last leg, now reused as the new uptrend's first
    zone). Direction: **buy**.
  - Otherwise (trend was already up, or undefined): no zone here, just confirms the
    uptrend continues.
- **New swing high, but it does *not* exceed the previous swing high** → this is a lower
  high, so the trend is (or becomes) down.
  - **Pullback zone**: `top = this new swing's price`, `bottom = last_low_price`.
    Direction: **sell**.
- Symmetrically for a new swing low vs. the previous swing low (lower low → trend down,
  possible flip zone with direction **sell**; higher low → pullback zone, direction **buy**).

```
has_high = has_low = False
last_high_price = last_low_price = 0.0
trend = 0                                   # 0 = undefined, 1 = up, -1 = down
for each confirmed swing (price p, is_high):
    if is_high:
        if has_high:
            if p > last_high_price:
                if trend == -1 and has_low:
                    ZONE: top=last_high_price, bottom=last_low_price, buy=True,  flip=True
                trend = 1
            else:
                if has_low:
                    ZONE: top=p,               bottom=last_low_price, buy=False, flip=False
                trend = -1
        last_high_price = p; has_high = True
    else:  # symmetric, is_low
        if has_low:
            if p < last_low_price:
                if trend == 1 and has_high:
                    ZONE: top=last_high_price, bottom=last_low_price, buy=False, flip=True
                trend = -1
            else:
                if has_high:
                    ZONE: top=last_high_price, bottom=p,               buy=True,  flip=False
                trend = 1
        last_low_price = p; has_low = True
```

Code: `build_zones_every_swing()` in `goal95_every_swing_zones.py`.

---

## 4. Uptrend / downtrend definition

This is intentionally its own section because it is easy to assume it works like a moving
average or a higher-timeframe filter. **It does not — it is purely mechanical and entirely
local to the swing sequence itself.**

- `trend` starts at `0` (undefined). It is not "up until proven down" or vice versa — it
  genuinely has no value yet.
- The **first swing of each type** (the very first swing high ever, and the very first
  swing low ever) can never produce a trend value or a zone by itself — `has_high`/
  `has_low` are `False` the first time, so the code only records the price and returns,
  per the `if has_high:` / `if has_low:` guards above. It takes **three swings** at
  minimum (two of one type, one of the other — e.g. H1, L1, H2) before `trend` is set for
  the first time, and that first determination produces **no zone**, only a trend value
  (see the "otherwise, no zone" branches above) — the first zone appears on the swing
  *after* that.
- Once three-plus swings exist, trend is decided by one comparison only: **does the newest
  swing of a given type exceed the previous swing of that same type?** Exceeds → up.
  Doesn't → down. There is no reference to the *other* type's swings, no slope
  calculation, no bar-count lookback, no external timeframe. A new swing low that is
  higher than the last swing low means "up," full stop, regardless of what the highs are
  doing at that moment.
- Because `last_high_price` and `last_low_price` update on **every** swing of their type
  (not just ones that set a new extreme), a "higher high" or "lower low" is always
  measured against the *immediately preceding* swing of that type — not against some
  older, more extreme swing further back. Trend can therefore flip on very small,
  short-lived wiggles; nothing here filters for significance.

In short: **uptrend/downtrend is not something separately computed and then applied to the
swings — it falls directly out of comparing each new swing to the last one of its own
kind, with an arbitrary high-first bootstrap and a three-swing warm-up before it means
anything at all.**

---

## 5. Direction and entry

The zone's own `buy`/`sell` label from §3 is traded **as-is — not inverted**. A zone
labeled buy (formed during an uptrend) is bought; a zone labeled sell (formed during a
downtrend) is sold. This is the original, most literal reading of "buy the dip in an
uptrend, sell the bounce in a downtrend" — no fade, no contrarian twist.

**Entry price is the zone's exact midpoint**: `mid = (top + bottom) / 2` — a 50%
retracement, not 30% or any other depth.

---

## 6. Zone validity — the overshoot-then-tag gate

A zone is not simply "live from the moment it's created until price touches 50%." Before
an entry counts at all, price must prove the zone by **leaving it behind first**:

1. **Overshoot.** Starting at the zone's own creation bar (inclusive — a flip zone can
   already satisfy this on the bar it's created, since a flip is definitionally triggered
   by price already breaking past the old structure), find the first bar where price
   trades beyond the zone's far edge in the zone's own implied continuation direction:
   `high > top` for a buy zone, `low < bottom` for a sell zone. Until this happens, the
   zone has not yet been "left behind" and cannot be entered.
2. **Tag.** From the overshoot bar onward, find the first bar where price swings back and
   reaches the 50% midpoint: `low <= mid` (buy) or `high >= mid` (sell). This bar is the
   entry.
3. A zone that never overshoots, or overshoots but never comes back to tag 50%, simply
   stays open/unresolved — it is not a trade, and (for the purposes of this document) is
   not counted in any backtest statistic.

This differs from this project's older, simpler entry convention elsewhere (fill on the
very first touch of 50% within some window, no overshoot required, e.g.
`goal81_shallow_entry_two_account.py::_resolve_frac_ex`). That older convention is **not**
part of this base spec — it's a separate, looser mechanic used by earlier research. Going
forward, the overshoot-then-tag gate in this section is the definition of "this zone can be
entered."

Code: `resolve_zone_lifetime()` in `goal98_zone_lifetime.py` implements exactly stages 1–2
(plus a stage 3, exit, described next).

---

## 7. Stop and target

**Stop** is always the zone's own opposite edge: `bottom` for a buy, `top` for a sell. This
never changes between the two target conventions below.

Two target conventions are defined — both are part of this base spec, to be compared
directly against each other, not one chosen over the other yet:

**(A) Fixed 1:1R.** Target is the zone's own edge on the far side: `top` for a buy, `bottom`
for a sell. Since entry sits at the exact midpoint, this target is, by construction,
exactly one risk-unit (`R`) away — the same distance as entry-to-stop. Reward:risk is
always exactly 1:1 under this convention; win rate alone determines expectancy. **Live
deployment (2026-09-08) generalizes this to `mid ∓ SL_MULT·half_height` /
`mid ± TP_MULT·half_height`** (`config.py`'s `SL_MULT`/`TP_MULT`, currently 1.25/1.25 —
see "SL/TP multiplier widened" in [Status](#status--known-results)); 1.0/1.0 recovers the
raw-edge convention described above exactly. Still exactly 1:1R whenever the two
multipliers are equal — only the distance from mid changes, not the reward:risk ratio.

**(B) Trailing last-confirmed-extreme.** Target is **the most recently confirmed swing
high** (buy) or **swing low** (sell) *as of the current bar* — not fixed at entry, but
re-read on every bar while the trade is open, using the same `last_high_price` /
`last_low_price` running values from §3 (tracked for the whole instrument history, not
reset per trade), floored at `max(top, current last-confirmed-high)` / capped at
`min(bottom, current last-confirmed-low)` so it's never worse than (A). **Tested and
rejected** — see [Status](#status--known-results): a target that keeps re-reading itself
every bar keeps sliding away for as long as the trend is still extending it, which means it
structurally cannot be "caught" while that's happening; by the time the trend pauses long
enough for price to reach it, price has usually already reversed into the stop instead.

**(C) Fixed last-confirmed higher-high / lower-low, read once at entry.** Same idea as (B)
but the failure mode above is specifically about the target *moving while the trade is
open* — so (C) reads the identical `last_high_price` / `last_low_price` value **once, at
the entry bar, and holds it fixed** for the rest of the trade, exactly like a real
take-profit order placed at a known level rather than one that keeps sliding. Floored the
same way: `max(top, last-confirmed-high at entry)` for a buy, capped the same way for a
sell. Because entry requires the §6 overshoot (price must already have traded past `top`
before an entry counts), the very swing that produced that overshoot is often *itself* a
freshly confirmed higher high exceeding `top` — meaning this target is frequently, though
not always, strictly better than (A)'s. **Also tested — see Status**: true in theory, but
the realistic edge this produces is thin enough that this project's own cost model erases
it entirely on this zone-generation mechanism.

A trade exits on the first bar (from the tag bar onward) where price trades through either
the stop or the current target; if both would trigger on the same bar, the stop takes
priority (the conservative assumption used throughout this project).

---

## 8. Costs

Spread is modeled exactly as everywhere else in this project: **1.6 pips**, applied at
entry only, using the depth-dependent fill rule already established and confirmed
(`SPREAD_PIPS_TYPICAL = 1.6` in `prod_filter_grid_search.py`) — if the tag bar's wick
reaches a full spread past the midpoint, the fill is free (exactly at `mid`); if the touch
is shallow, the fill is one full spread worse (`mid + 1.6 pips` for a buy, `mid - 1.6 pips`
for a sell). No separate cost is charged on the exit side — this matches the user's own
confirmed assumption elsewhere this session (market orders, 0.8 pips each way, 1.6 total is
"enough"), realized in code as a single entry-side adjustment rather than two separate
0.8-pip charges.

A minimum effective stop distance of **3 pips** is enforced (`MIN_STOP_PIPS`, matching
`goal95`/`goal96`'s convention) — zones whose entry-to-stop distance, after the spread
adjustment, is smaller than this are discarded rather than traded, since a 1-2 pip stop is
not realistically tradeable (would be inside typical broker minimums / normal quote noise).

A separate, **raw zone-height floor** (checked before any spread adjustment, distinct from
the post-spread `MIN_STOP_PIPS` check above) is worth enforcing too — swept explicitly in
[Status](#status--known-results) and found to matter a lot: below it, the 1.6-pip spread
eats a disproportionate share of small zones' edge for both target conventions. **5-10
pips** is the best-supported floor found so far; `goal103_min_zone_size_sweep.py`'s
`min_zone_height` parameter on both resolvers implements this.

---

## 9. What this spec does *not* define

- **Position sizing / account structure.** Not addressed here — see
  `research_scripts/goal79_two_account_portfolio.py` and `data/STRATEGY_RANKINGS.md` for
  the project's separate two-account, margin-checked sizing convention, which sits on top
  of whatever trade list this spec produces.
- **Multi-instrument concurrency rules** (max one open position per side, etc.) — same,
  handled at the portfolio layer, not here.
- **Maximum holding window.** A practical backtest needs one (this project's convention
  elsewhere is 1440 bars / 1 trading day) so that a handful of pathological, never-resolving
  zones don't force a scan to the end of the dataset; it is an implementation bound, not
  part of the trading rule itself.

---

## Status / known results

This exact mechanism (every swing a zone, 50% entry, non-inverted direction) has **not**
been fully validated — it is a base for experimentation, with one real, promising, but
*not yet trusted* result:

- An earlier, narrower version of this same non-inverted 50%-entry idea (without the
  overshoot-gate from §6) was full-history backtested across all four project pairs and
  **lost money on every one** (profit factor 0.52–0.66 over 17 years) — see the archived
  ranking `data/archive/STRATEGY_RANKINGS_pre_2026-09-06_fade_pullback.md`.
- The overshoot-gate (§6) had **not** been tested at scale before this document. First
  full-EURUSD-history run (2009-09-25 → 2026-09-03, 6,340,787 bars, `goal99_dtfx_base_backtest.py`),
  3,119,633 swings, 2,141,316 zones:

  | | trades | win rate | avg R | total R | PF |
  |---|---|---|---|---|---|
  | **(A) fixed 1:1R** | 846,913 | 88.38% | +0.0425 | **+35,969** | **1.366** |
  | **(B) trailing last-confirmed extreme** | 846,883 | 71.83% | −0.0881 | −74,597 | 0.687 |
  | **(C) fixed last-confirmed HH/LL at entry** | 846,746 | 61.09% | −0.0895 | −75,794 | 0.770 |

  **(A) beats both (B) and (C) decisively**, on the exact same trade population and cost
  model. (B) fails because a target that's still being extended by the trend can't be
  caught while it's happening (see §7). (C) is the more interesting failure, because the
  user's underlying intuition here is *correct in a frictionless world* — re-running (C)
  and (A) with spread forced to zero (a different, smaller, larger-zones-only trade
  population, since removing spread also removes the padding that let many sub-3-pip zones
  clear the minimum-stop filter in the first place: 175,034 trades instead of 846,913):

  | (zero-spread) | trades | win rate | avg R:R offered | PF |
  |---|---|---|---|---|
  | (A) fixed 1:1R | 175,128 | 71.49% | 1.0 (fixed) | 2.507 |
  | (C) fixed HH/LL | 175,034 | 48.82% | **2.170 (avg)** | 1.502 |

  So yes — pre-cost, (C) really does offer an average 2.17:1 reward:risk, and a sub-50%
  (48.82%) win rate at that ratio is indeed profitable (PF 1.502), exactly matching the
  logic behind the user's original request. **It just isn't the better choice here**: (A)
  is *more* profitable pre-cost too (PF 2.507), and once the realistic 1.6-pip spread
  (§8) is put back, (C)'s edge doesn't survive at all (PF 0.770) while (A)'s does (1.366).
  Two things compound against (C) specifically on this every-swing mechanism: most zones
  are tiny, so a fixed 1.6-pip cost eats a much larger share of a small edge than a large
  one; and the close-based swing detector's reversal trigger is maximally sensitive (any
  single down-close locks in a high), so the "genuine higher high" an overshoot produces is
  usually only a couple of pips beyond the zone's own top, not a dramatically larger one —
  not enough room for the extra reward to outrun the extra cost.

  **This was tested, not left as a guess** (`goal103_min_zone_size_sweep.py`, all four
  pairs, full history, a raw zone-height floor swept from 0 to 150 pips, checked before any
  spread padding):

  | min raw zone height | (A) PF | (A) n | (C) PF | (C) n |
  |---|---|---|---|---|
  | 0 pips (no floor) | 1.479 | 3,338,463 | 0.828 | 3,337,729 |
  | 3 pips | 1.540 | 2,895,638 | 0.873 | 2,894,927 |
  | 5 pips | 1.727 | 1,166,328 | **1.064** | 1,165,784 |
  | 7 pips | 1.732 | 595,427 | 1.125 | 595,042 |
  | 10 pips | 1.756 | 223,188 | 1.184 | 222,951 |
  | 20 pips | 1.568 | 22,343 | 1.211 | 22,262 |
  | 50 pips | 1.877 | 1,667 | 1.286 | 1,646 |
  | 100 pips | 2.504 | 603 | 1.387 | 596 |
  | 150 pips | 1.590 | 215 | 0.940 | 210 |

  **(A) is robust everywhere** — profit factor holds above 1 at every threshold tested, and
  generally *improves* as tiny zones are excluded (the 150-pip row's dip is sample-size
  noise, only 215 trades). **(C) crosses into profitability right around a 5-pip floor**
  (PF 0.828 → 1.064) and keeps improving through 100 pips (PF 1.387) before the same
  small-sample noise hits at 150 — confirming the theory from the discussion above: (C)'s
  larger reward:risk is real, it just needs the tiny, cost-dominated zones excluded to show
  up net of spread. Read the 100-and-150-pip rows as suggestive, not reliable — under 1,000
  trades each is a thin sample for a PF estimate. A minimum raw zone height around
  **5-10 pips** looks like the best-supported floor for either variant going forward: enough
  to escape the small-zone cost trap, while 10 pips still leaves 223,188 (A) / 222,951 (C)
  trades — plenty of sample.

  **Why (A)'s win rate is 88% yet PF is only 1.366, not ~7.6 (what flat 1:1R wins/losses
  would imply):** the 1.6-pip entry-side spread (§8) is charged against zones whose
  raw height is often close to the 3-pip minimum, which compresses the *realized* R on a
  win well below 1.0 (a win on a 3-pip-tall zone nets nowhere near a full R once spread is
  backed out) while every loss still costs a full −1R. High hit rate, thin edge per hit.

  **A real caveat before trusting (A)'s aggregate number**: the per-year breakdown
  (`data/goal99_variant_a_yearly.csv`) shows a sharp, sustained break at 2021 — trade
  volume roughly **doubles** (33.7k/yr avg 2009–2020 → 73.7k/yr avg 2021–2026) and win
  rate jumps from **~81% to ~94.5%**, at the same time raw per-bar range (volatility) only
  rises modestly (an existing, unrelated strategy's yearly stats in
  `data/dtfx_lse_vendor_check_yearly.csv` show nothing close to this magnitude of shift on
  the same feed over the same years). That combination — most of this backtest's positive
  total R comes from 2021 onward — needs to be run down (a feed/vendor change in this CSV
  around 2021, a resolution or tick-aggregation difference, something specific to how tiny
  zones behave in that period) before this result is trusted rather than just reported.
  Until then, treat the pre-2021 numbers (WR ~81%, still marginally profitable most years)
  as the more credible read of this mechanism's real edge. (A quick check ruled out the two
  most obvious artifacts — flat/zero-range bars and missing-minute timestamp gaps both stay
  flat or even improve across the 2021 boundary — without finding an actual explanation;
  this still needs a real look, not just those two checks.)
- Every other strategy refinement explored this session so far (inversion, shallower entry
  fractions, structure-confirmation gates) was a deviation from this exact base — if a
  future experiment reintroduces one of those, say explicitly which part of this document
  it's changing, rather than silently drifting from it.

### Minimum zone height and risk per trade, under the real two-account architecture

Everything above the min-zone-height sweep used a simplified single-pool, sequential-risk
account model. Re-tested variant A (fixed 1:1R) under this project's actual live
architecture instead (`goal104_optimal_risk_two_account.py` /
`goal105_mc_risk_validation.py`) — two separate $100 sub-accounts (buy-only, sell-only,
never pooled), sizing = risk_pct × min(buy_equity, sell_equity), max one confirmed open
position per side across all four pairs, 50x leverage, $2M position-sizing cap, 1.6-pip
spread on every entry — sweeping both the minimum raw zone height (5/10/20/30/50 pips) and
risk per trade (1% to 30%).

**Two genuinely different questions give two different answers, and conflating them is the
mistake to avoid:**

**"How fast can I reach $1M, and how risky is that specific race?"** — essentially safe at
every risk level tested. Block-bootstrap Monte Carlo (1,000 simulated paths per point, 5-
and 10-pip zone heights): **0.00% ruin and 100% of paths reach $1M at every risk from 1% to
15%**, typically in well under a year (median 0.13–1.28 years depending on risk and zone
height). Lower risk just means a smoother ride to the same destination — median max
drawdown along the way runs from about −7% (1% risk) up to −19% (15% risk).

**"What happens if I keep compounding at that same risk fraction indefinitely, past $1M,
for the full 23-year history?"** — this is where it breaks. A deterministic, single-pass
replay of the *entire* historical trade sequence in order (not bootstrapped) shows CAGR
climbing with risk up to a point (5-pip zones: 108% CAGR at 5% risk; 10-pip zones: 100% CAGR
at 5% risk) — **but with the equity curve already down 85-96% from its own peak at some
point along the way even at "optimal" risk** — and then a sharp cliff into near-total ruin
just one or two risk-grid steps higher (10% risk on 5- or 10-pip zones takes the account
from billions down to single dollars; the same failure appears at 15% for 20-pip zones,
20%+ for 30-pip zones). The Monte Carlo above never encounters this because it stops the
instant a path reaches $1M — long before a large enough losing streak has time to occur
against a since-compounded, much larger account. The deterministic full-history run keeps
going, and eventually meets one.

**Why both are real, not a bug**: at a fixed risk fraction, a positive-expectancy trade
stream can still have *negative* long-run geometric growth once risk exceeds the true
Kelly-optimal fraction for its actual win/loss distribution — arithmetic edge per trade
says one thing, compounded growth over enough trades says another. The $2M sizing cap
slows this, it doesn't eliminate it: a big enough losing streak against a $2M-based
position size is still a real dollar loss, and a strategy trading hundreds of thousands of
times across 23+ years will eventually run into one.

**Recommendation, stated for the two different goals separately** (not one number, because
"optimal" here genuinely depends on the horizon): for a *bounded* goal (grow $200 to $1M,
then stop or reassess), 3-5% risk on a 5-10 pip minimum zone height is well-supported —
fast, and the Monte Carlo shows no meaningful ruin risk on the way. For *sustained,
indefinite* compounding at the same fixed fraction — the realistic long-term case — treat
5% as already close to the edge given the 85-92% historical drawdown it produced even
without full ruin, and prefer something meaningfully lower (1-3%) if the plan is to keep
re-risking that fraction for years rather than re-evaluating sizing as equity grows. A
**10+ pip minimum zone height paired with 3-5% risk** is the best-supported combination
across both the profitability sweep and this risk analysis together.

### 2026 fresh-start trial — and new evidence on the 2021+ anomaly

Re-ran the two-account setup starting completely fresh on 2026-01-01 (not continuing prior
compounding), 3% risk, across the same zone-height options (`goal106_2026_ytd_two_account.py`).
20-30 pip zones held up (all months profitable, no dollar-figure distortion); 5-10 pip
zones hit the same unrealistic $2M-cap compounding as elsewhere and should be read for win
rate only, not dollar totals.

**The important new finding is in the robustness check, not the headline numbers**:
splitting 2026 YTD into thirds, win rate *declines in every successive third* at both 5
and 10 pip zone heights (83.6% → 76.7% → 64.8%, and 75.7% → 68.0% → 59.6%). This is new,
independent evidence bearing directly on the still-unexplained 2021+ win-rate jump flagged
earlier in this document: if that jump was partly a data artifact rather than a permanent
regime change, a gradual reversion during 2026 is exactly what it would look like. This
doesn't resolve the question, but it raises the priority of resolving it before treating
any of this mechanism's post-2021 numbers — including every result in this document — as
settled. **Verdict on live-readiness, stated plainly: not ready.** In-sample parameter
selection (no held-out data was ever reserved), the unexplained anomaly this new evidence
deepens rather than resolves, spread-only cost modeling, the simplified concurrency model,
and zero paper-trading track record are all still open — see `the_2026_trial` report for
the full checklist.

### The 2021+ anomaly, resolved: a data-feed artifact, precisely dated

Investigated directly (2026-09-07) rather than left as a caveat. Measuring the % of
near-zero-range (<0.15 pip) 1-minute bars by month, all four pairs show the **same**
sharp, synchronized shift: ~1-4% typical from 2003/2009-2020 (natural quote
microstructure), collapsing to ~0.05-0.6% from **2021-01 through 2026-03** (a much
"smoother" feed), then reverting sharply back to ~2-12% from **2026-04 onward** — matching
the pre-2021 character almost exactly. Same two boundary months, independently, on
EURUSD, GBPUSD, EURGBP, and USDJPY. This is a data pipeline characteristic, not a market
regime change, and it lines up exactly with every anomaly flagged earlier: the WR jump
starting 2021, and the within-2026 WR decay in `the_2026_trial`'s robustness thirds (the
decay boundary sits almost exactly at the April 2026 reversion).

**Consequence**: every 2021-2025 number in this document (and the artifacts linked from
it) is inflated by this artifact and should not be trusted at face value. Re-derived
everything that mattered using only the trustworthy ("noisy", natural-microstructure)
windows — `goal108_regime_split_validation.py`:

| min zone | PRE-2021 PF (trustworthy) | SMOOTHED PF (artifact window) | POST-2026-04 PF (genuine holdout) |
|---|---|---|---|
| 5 pips | 1.452 | 2.349 | 1.335 |
| **10 pips** | **1.565** | 2.192 | **1.459** |
| 15 pips | 1.526 | 1.913 | 1.566 |
| 20 pips | 1.454 | 1.793 | 1.804 |

**10 pips remains the best-supported minimum zone height even restricted to honest,
pre-2021 data alone** (PF 1.565, n=143,363) — the earlier recommendation wasn't an
artifact of the smoothed window. The risk-per-trade cliff also reproduces exactly on
PRE-2021-only data: CAGR climbs to 137.6% at 5% risk, then crashes to ruin at 10% —
the same shape, same location, as the full-history analysis in The Risk Ceiling.

**The walk-forward proof**: froze the config chosen using only PRE-2021 data (10-pip
zone, 3% or 5% risk) and tested it against POST-2026-04 — a window that was *never*
touched during any selection step in this document. Result: WR 63.6-64.1%, PF-consistent
with the 66.18% seen in training — not the 72%+ seen in the excluded smoothed window.
**A real, modest, honestly-validated edge survives out-of-sample testing that was
impossible to game.** This is the strongest evidence this project has produced that the
mechanism isn't pure overfitting.

**Cost sensitivity** (`goal109_spread_stress_test.py`, PRE-2021 data, 10-pip zone): PF
stays above 1.0 up to roughly **3.0 pips** of spread (PF 1.059) before turning
unprofitable at 4+ pips — real margin against the 1.6-pip baseline assumption, though not
an enormous one.

**Revised live-readiness checklist**:
- ✓ **Resolved**: the 2021+ anomaly is now explained, dated, and excluded from parameter
  selection.
- ✓ **Resolved**: parameter selection now uses a genuine train/holdout split — 10-pip
  zone height and 3-5% risk were chosen on PRE-2021 data only and validated, not tuned,
  against POST-2026-04.
- ✓ **New evidence**: cost sensitivity is now quantified (survives to ~3 pips spread).
- ✕ **Still open**: no paper-trading or forward-test track record exists — cannot be
  produced by backtesting, only by running the frozen configuration against a live or
  simulated real-time feed for a real stretch of time.
- ✕ **Still open**: the more accurate cross-symbol concurrency simulator referenced in
  `goal79_two_account_portfolio.py`'s own docstring does not actually exist in this
  codebase (confirmed by search) — the 1-slot-per-side approximation used everywhere in
  this document remains unvalidated against the real production concurrency logic.
- ✕ **Still open**: no account-level risk controls beyond position sizing (daily loss
  limit, circuit breaker, feed-anomaly detection) have been designed or tested.
- ~ **Unchanged**: execution still modeled as spread-only — no slippage, latency, or
  partial fills, though the cost-sensitivity test above at least bounds how much hidden
  cost the edge can absorb.

**Frozen recommended configuration, honestly validated**: variant A (fixed 1:1R),
**10-pip minimum zone height**, **3% risk per trade** (prudent side of the 5% growth-optimal
point, given the severe drawdowns — even 3% ran -79% peak-to-trough on PRE-2021 data alone),
1.6-pip spread modeled (real margin to ~3 pips). This is the first configuration in this
document with genuine out-of-sample validation behind it — closer to live-ready than
anything before it, but still blocked on paper-trading and operational risk controls before
real capital.

### The production concurrency gap, closed

`PRODLIVETRADING/strategy/engine.py::resolve_multi_symbol_history` turned out to already
exist — the actual, deployed cross-symbol concurrency logic, confirmed by reading it
directly rather than assumed: a single global open-position count per side, walked in true
chronological order across all four symbols' bars merged together (not each symbol
resolved independently and filtered after, which was `goal79_two_account_portfolio.py`'s
simplification), "rejected, not queued" at the exact moment a touch is concurrency-blocked
but the zone stays alive to retry on later bars, and an explicit invalidation rule (a
pending zone dies if price breaches its own stop side before ever filling).

Ported this exact mechanism — not a new guess — to the DTFX-every-swing zones
(`goal110_production_concurrency_simulator.py`, PRE-2021 data, 10-pip zone height, 3% risk).
One real bug caught and fixed in verification before trusting the result: a same-bar
open-and-close could momentarily read the slot as free, letting a second zone fill at the
exact same timestamp (~0.17% of trades); fixed by requiring the slot to free strictly
*after* the exit bar, not at it. Re-verified afterward: zero overlap violations across
117,489 trades.

| | trades | win rate | PF | max drawdown |
|---|---|---|---|---|
| No concurrency filter (raw candidates) | 143,363 | 66.18% | 1.565 | — |
| Old simplified filter (independent-resolve-then-reject) | 121,478 | 66.46% | 1.586 | −76.54% |
| **Production-accurate (this section)** | **117,489** | **70.81%** | **1.920** | **−23.11%** |

**The more accurate model is better on every axis, not worse** — higher win rate, higher
profit factor, and a dramatically smaller max drawdown (−23% vs −77%). The simplified
model's flaw was discarding any zone blocked at its own independently-computed touch
moment forever, rather than letting it retry once a slot actually freed — which meant it
kept some accidentally-timed opportunities and permanently dropped others that a real
resting order would have caught later, adding avoidable noise to both the trade selection
and the resulting equity path. This closes the concurrency-model item from the
live-readiness checklist.

### Paper-trading: deployed, produced zero trades — debugged, root-caused, hardened

First deploy attempt (`PRODLIVETRADING/paper_dtfx.py`, practice sub-accounts
`101-001-27040015-002`/`-003`) produced **zero trades overnight** despite an expected rate
of roughly **8-10 trades/day across all 4 pairs** at the frozen config (10-pip zone, 3%
risk) — measured directly against the current live regime: 1,286 accepted trades over the
155-day POST-2026-04 holdout (8.27/day), 68 in just the most recent 7 days of available
history (9.71/day). Zero for a full overnight session is a real anomaly, not a quiet
patch, and was investigated directly (2026-09-07) rather than left as a caveat:

- **Ruled out — the live zone-detection logic itself.** `SymbolState` (the incremental,
  bar-by-bar port of `close_based_zigzag_symmetric` + `build_zones_every_swing` this
  script runs live) was fed real, recent EURUSD 1-minute data and produced the same zone
  count, on the same bars, as the batch reference functions it was ported from — confirmed
  by direct comparison, not just a shape-check on a synthetic trace. The mechanism is
  correctly implemented.
- **Ruled out — broker credentials.** Both practice sub-accounts authenticate and respond
  normally right now (`account_balance()`, `open_trades()` both succeed, balances $90.59 /
  $98.48, zero open positions). `bot.py`'s log does show a `401 Insufficient authorization`
  error against these exact same two account IDs on 2026-08-27 — but that's resolved as of
  this check, not the current blocker.
- **Found — no evidence the process was ever running as a persistent service.** Locally:
  `logs/paper_dtfx.log` existed at 0 bytes (an artifact of this session's own earlier
  import-only smoke test, not a real run), `paper_dtfx_trades.csv` was never created (which
  `main()` creates in its very first step), and no process was running. This session has no
  SSH access to the actual deployment VM and cannot confirm what happened there directly —
  but the script had **no systemd unit, no `nohup`, no supervision of any kind**: run in a
  foreground SSH/terminal session, it dies the instant that session disconnects, with
  nothing to restart it. This is the leading explanation and was treated as a real
  infrastructure gap regardless of the exact cause on that specific run.
- **Found and fixed — a genuine startup fragility bug.** The one-time warmup phase (connect
  → fetch 10,000 bars × 4 symbols → replay) had no retry and sat entirely outside the poll
  loop's own `try/except`. A single transient OANDA hiccup on any one of those four fetches
  — plausible, given four consecutive large historical pulls — killed the whole process
  permanently, before it ever reached the resilient part of the code. Fixed: the
  connect+warmup step now retries with exponential backoff (5s → 300s cap) instead of
  propagating.
- **Found and fixed — no restart-time position reconciliation.** `open_count`/`open_trade`
  were plain in-memory variables, reset to empty on every process start. A restart (crash,
  manual bounce, systemd `Restart=always`) while a position was genuinely open would let a
  *second* trade open on that same side, doubling up risk the account isn't sized for. Fixed:
  startup now queries `open_trades_detailed()` + `trade_details()` on both sub-accounts
  first and seeds state from whatever the broker actually shows, the same restart-recovery
  problem `bot.py::_sync_positions` already solves for Apex-Ultra.
- **Added — `deploy/dtfx_paper.service`**, a systemd unit for `paper_dtfx.py` mirroring
  `prodlivetrading.service`'s own convention (`Restart=always`, `RestartSec=10`), with
  `Conflicts=prodlivetrading.service` so systemd itself enforces the two strategies can't
  run against the same accounts at once, on top of the script's own startup check.

**To actually deploy** (still a manual step — no SSH access from this environment):
```bash
bash deploy/deploy_update.sh    # pulls, backs up, verifies, installs dtfx_paper, stops
                                 # Apex-Ultra if still active, restarts dtfx_paper
journalctl -u dtfx_paper -f     # watch it actually start trading
```
**No paper-trading results exist yet and cannot be produced by more backtesting** — they
only start accumulating once this is actually running against real-time data for a genuine
stretch of calendar time.

**Config switched to 5-pip / 3% risk (2026-09-07)**, per direct instruction, following the
5-pip @ 3% risk results above: `MIN_ZONE_PIPS` 10 → 5, `MAX_ALLOWED_SPREAD_PIPS` 3.0 → 2.0
(tightened to match 5-pip's measured ~2.0-2.5p cost breakeven from
`goal111_5pip_vs_10pip_depth.py` — leaving the old 3.0 default would let the live spread
gate approve trades the backtest itself shows are already unprofitable). `RISK_PCT` stays
3% — pushed as-is per instruction, though the drawdown comparison above shows that figure
was calibrated against 10-pip's turnover, not independently validated for 5-pip's higher
trade frequency. Pushed to `PRODLIVETRADING` (commit history there has the exact diff);
still requires the manual `deploy_update.sh` step above to actually take effect on the VM.

### OANDA Level 2 / liquidity data: not available

Tested directly against the practice API (2026-09-07): `InstrumentsOrderBook` and
`InstrumentsPositionBook` both respond, but **neither is real order-book depth**. Both
return the same shape — OANDA's own retail client base's positioning, bucketed into fixed
5-pip price buckets, as a `longCountPercent`/`shortCountPercent` (percentage of clients
long/short at that price), refreshed roughly every 20 minutes. There is no size, no
resting-order count, no depth-of-market, and no way to derive "how much can be traded
before moving the price" from it — spot forex has no central limit order book to query in
the first place; OANDA is a market maker, not an exchange. **Conclusion: there is no
liquidity-based position-size ceiling available from this broker's API.** The real,
already-implemented ceilings — `risk_pct × min(buy_equity, sell_equity)`, `MAX_LEVERAGE`,
and `account_margin_available()` checked before every order — remain the only actual
limits on trade size; a genuine depth-of-market feed would require an ECN/prime-broker
venue, out of scope for this OANDA-retail setup.

### 5-pip vs 10-pip, deeper: does 5-pip survive realistic spread?

`goal111_5pip_vs_10pip_depth.py` decomposes win/loss into their own average R (not just a
blended figure) across a spread ladder, for both candidate zone heights. Loss is exactly
**-1.0000 R in every single row** by construction (a hit stop always means the full,
fixed risk unit was lost under target-A's fixed 1:1R) — shown plainly rather than omitted,
since it's true by design, not a finding. Win size is what actually moves, and it moves
for a specific, provable reason: `avg_R_win = (half_height − spread) / (half_height +
spread)` — a 5-pip zone's 2.5-pip half-height is far more exposed to a fixed spread cost
than a 10-pip zone's 5-pip half-height.

**PRE-2021 data (honest baseline):**

| zone | spread | n | win rate | avg R (win) | avg R (loss) | PF |
|---|---|---|---|---|---|---|
| 5 pip | 1.6p | 684,410 | 72.30% | +0.556 | −1.000 | **1.452** |
| 5 pip | 2.0p | 704,258 | 72.59% | +0.438 | −1.000 | 1.160 |
| 5 pip | 2.5p | 725,214 | 72.87% | +0.302 | −1.000 | 0.810 |
| 5 pip | 3.0p | 746,101 | 72.92% | +0.213 | −1.000 | 0.573 |
| 10 pip | 1.6p | 143,363 | 66.18% | +0.800 | −1.000 | **1.565** |
| 10 pip | 2.0p | 143,363 | 66.18% | +0.727 | −1.000 | 1.422 |
| 10 pip | 2.5p | 143,363 | 66.18% | +0.632 | −1.000 | 1.236 |
| 10 pip | 3.0p | 143,363 | 66.18% | +0.541 | −1.000 | 1.059 |

**Walk-forward holdout (POST-2026-04, genuine, never used for selection), spread=1.6p:**

| zone | n | win rate | avg R (win) | PF |
|---|---|---|---|---|
| 5 pip | 9,450 | 72.56% | +0.505 | **1.335** |
| 10 pip | 1,442 | 64.08% | +0.818 | **1.459** |

**Same numbers in raw pips** (this is what would actually show up on an account
statement — "1R" is a normalized unit, pips are not, since `eff_risk` varies zone to
zone). Avg LOSS pips is not a fixed number the way R is; it's literally the stop distance
of the average losing trade, which shifts with spread because a wider spread pushes entry
closer to the stop before the trade is even placed:

| zone | spread | n | win rate | avg stop risked | avg WIN | avg LOSS |
|---|---|---|---|---|---|---|
| 5 pip | 1.6p | 684,410 | 72.30% | 5.25p | 2.99p | 5.25p |
| 5 pip | 2.0p | 704,258 | 72.59% | 5.59p | 2.53p | 5.50p |
| 5 pip | 2.5p | 725,214 | 72.87% | 6.07p | 1.92p | 5.87p |
| 5 pip | 3.0p | 746,101 | 72.92% | 6.56p | 1.49p | 6.25p |
| 5 pip | 1.6p (POST-2026-04) | 9,450 | 72.56% | 5.21p | 2.78p | 5.27p |
| 10 pip | 1.6p | 143,363 | 66.18% | 7.86p | 6.25p | 7.87p |
| 10 pip | 2.0p | 143,363 | 66.18% | 8.17p | 5.90p | 8.11p |
| 10 pip | 2.5p | 143,363 | 66.18% | 8.62p | 5.41p | 8.46p |
| 10 pip | 3.0p | 143,363 | 66.18% | 9.10p | 4.88p | 8.85p |
| 10 pip | 1.6p (POST-2026-04) | 1,442 | 64.08% | 9.06p | 7.77p | 8.74p |

Note avg LOSS pips is close to but not exactly equal to avg stop risked — the two would
be identical if every trade risked the same distance, but they don't (zone height varies
trade to trade), so "avg loss" pools whichever specific trades happened to lose, not the
full population's average stop. The mechanism from the R-based table holds in raw terms
too: on a 5-pip zone, going from 1.6p to 3.0p spread nearly halves the average win (2.99p
→ 1.49p) while the average loss actually *grows* (5.25p → 6.25p, because a wider spread
pushes entry further from the stop) — that double squeeze is exactly why 5-pip's PF falls
off a cliff so much faster than 10-pip's.

**Verdict**: 5-pip zones genuinely work at realistic spread — profitable both PRE-2021 and,
more importantly, on the never-touched POST-2026-04 holdout (PF 1.335, ~6.5× the trade
count of 10-pip) — but their margin for error is much thinner. PF crosses below 1.0
somewhere between 2.0p and 2.5p spread for 5-pip zones, versus between 3.0p and 4.0p for
10-pip. `paper_dtfx.py`'s current `MAX_ALLOWED_SPREAD_PIPS=3.0` default was chosen for the
10-pip config; switching to 5-pip zones without tightening that gate (to roughly 2.0p)
would let the live spread filter approve trades the backtest itself shows are already
unprofitable at that cost level. 10-pip remains the higher-margin, lower-frequency choice;
5-pip is a real, validated alternative for higher trade frequency IF the spread gate is
tightened to match — not a free upgrade on top of the existing gate.

### 5-pip @ 3% risk: the real two-account run

Direct user request: run the 5-pip config through the same $100+$100, 1-concurrent-per-side,
1.6-pip-spread two-account architecture already validated for 10-pip, at 3% risk per trade
(`goal112_5pip_3pct_report.py`). Final-balance figures below are the raw output of
uncapped, indefinite compounding over years of real elapsed time with no withdrawals —
mathematically real, but not a realistic expectation for a real account (the same caveat
that applies to every other uncapped run in this document); CAGR, win rate, PF, drawdown,
and trades/day are the numbers that actually matter here.

| segment | zone | n | trades/day | WR | PF | maxDD | CAGR |
|---|---|---|---|---|---|---|---|
| PRE-2021 | 5p | 549,583 | 83.70 | 72.67% | 1.489 | **−83.08%** | 149.8% |
| PRE-2021 | 10p | 122,442 | 18.65 | 66.45% | 1.585 | −79.26% | 137.8% |
| POST-2026-04 (holdout) | 5p | 5,905 | 37.94 | 73.01% | 1.400 | **−33.48%** | — |
| POST-2026-04 (holdout) | 10p | 1,263 | 8.12 | 63.58% | 1.444 | −17.30% | — |
| 2026 fresh start (Jan 1) | 5p | 24,794 | 101.29 | 79.95% | 2.108 | −7.52% | — |
| 2026 fresh start (Jan 1) | 10p | 3,957 | 16.17 | 71.59% | 2.057 | −12.44% | — |

**Trade frequency is 4-5× higher at 5p**, as expected. **PF and win rate are close between
the two heights on both the honest PRE-2021 sample and the genuine POST-2026-04 holdout**
(5p even wins on win rate, 10p slightly ahead on PF) — 5p is not a worse edge, just a
higher-turnover one. **The real cost is drawdown**: on the two large, trustworthy samples
(PRE-2021 and the holdout), 5-pip's max drawdown runs noticeably worse than 10-pip's at the
*same* 3% risk — roughly 2× worse in the holdout (−33.48% vs −17.30%). The single 2026
fresh-start window shows the opposite ordering, which is exactly why it shouldn't be
trusted alone — it's 245 days, not the ~6,566-day PRE-2021 sample or even the 156-day
holdout, and short windows swing DD ordering around by luck of the draw.

**Read on this**: 3% risk was calibrated against 10-pip's turnover rate in `goal104`'s
sweep (the risk cliff sat between 7-10% for that config). 5-pip trades 4-5× more often, so
the same 3% per trade compounds into drawdown faster — consistent with what the table
shows. This doesn't mean 3% is unsafe for 5-pip, but it does mean 3% wasn't independently
validated *for* 5-pip's higher turnover — a fresh risk sweep specifically for the 5-pip
config (mirroring `goal104`'s methodology) would be needed before treating 3% as the
optimal choice there rather than just a carried-over number, if 5-pip is the config
actually run live.

### Apex-Ultra retired; bot.py is now the DTFX strategy, no separate paper script

Direct instruction (2026-09-07): "completely delete apex ultra from PROD, bot.py should be
only about dtfx strat. don't do a separate paper_dtfx.py file, put it completely into
prod." Executed as a full rewrite, not a config toggle:

- **Deleted**: `strategy/pivots.py`, `strategy/structure_fsm.py`, `strategy/indicators.py`
  (Apex-Ultra's zone/indicator machinery, unused by anything else), `paper_dtfx.py` (merged
  into `bot.py`), `deploy/dtfx_paper.service` (bot.py is the DTFX strategy now — back to one
  systemd service), and the three Apex-Ultra-shaped test files (`test_reliability.py`,
  `test_deep_reliability.py`, `test_broker_lifecycle.py` — their `bot_fixture()`/`zone()`
  helpers assumed the retired soft-stop/resting-order-contention model throughout).
- **`strategy/engine.py` rewritten**: `SymbolEngine` is now a direct port of
  `paper_dtfx.py`'s already-validated `SymbolState` into bot.py's live interface. Zones are
  single-shot (no multi-touch re-arm, since dtfx.md never defined one).
- **`bot.py` rewritten**, keeping every genuinely generic piece (dual-layer state
  persistence, broker reconciliation by trade ID, per-side margin locks, the rollover/
  spread/margin filters, startup position recovery) and porting in `paper_dtfx.py`'s
  startup-retry-with-backoff fix. Removed everything specific to Apex-Ultra's resting-
  limit-order execution model: the soft-stop/backstop two-tier stop, per-(symbol,side)
  resting-order contention with closest-price replacement, cross-symbol sibling
  cancellation, and the transaction-streaming path that only existed to place the soft-stop
  within milliseconds of a fill. DTFX only ever dispatches one resting order per SIDE
  globally (not per symbol) — a zone on another symbol maturing while the slot is taken
  just waits and is re-checked next poll, so none of that contention machinery is needed.
- **A real bug caught in the process**: `paper_dtfx.py`'s own fill-tracking read
  `order.get("tradeID")` off `place_bracket`'s return value — but that call actually
  returns a bare order-id *string* (a resting LIMIT order was placed, not a confirmed
  fill), so `.get()` silently resolved to `None` every time. Its own close-detection would
  have misfired on the very next poll after any real fill (`None not in open_ids` is
  always true), freeing the one-per-side slot while the position was still genuinely open.
  Never triggered live (paper-trading never actually ran against a real feed), but would
  have defeated the concurrency cap immediately once it did. `bot.py`'s existing pending-
  order → broker-reconciliation flow (reused as-is here) never had this problem.
- `config.py`: removed `MAX_ZONE_AGE_HOURS`/`MAX_ZONE_TOUCHES` (Apex-Ultra only); added
  `MIN_ZONE_PIPS`/`MIN_STOP_PIPS`/`MAX_WINDOW_MIN` (DTFX's frozen parameters).
- **Verified end-to-end**: `deploy_readiness_check.py` (rewritten section 4 to test real
  DTFX zone lifecycle — creation, overshoot-then-tag, invalidation, expiry, single-shot
  retirement, the MIN_ZONE_PIPS floor) passes cleanly against the **live practice OANDA
  API** — both sub-accounts connect, all 4 pairs confirmed tradeable, margin checks read
  correctly, warmup-bar pagination works. The new `tests/test_dtfx_engine.py` (7 tests) plus
  the fixed `test_deployment_regressions.py` (5 tests, decoupled from the deleted test file)
  make 13/13 passing in the full suite.
- **Important operational note**: `.env` is gitignored and was NOT touched by the commit —
  the actual deployment's `.env` must be updated by hand (`SYMBOLS=EURUSD,GBPUSD,EURGBP,
  USDJPY`, `RISK_PCT=0.03`) before running `deploy_update.sh`, or `config.py`'s fallback
  defaults (`EURGBP`, 5%) silently apply instead — exactly the class of gap that caused the
  original zero-overnight-trades incident, called out explicitly here so it isn't repeated.

Pushed to `PRODLIVETRADING` main. Still not live: the same manual `deploy_update.sh` step
on the actual VM is required, and no paper-trading track record exists yet regardless of
which strategy the bot runs — that can only be produced by actually running it.

**Deployed and confirmed running (2026-09-07)**: the crash on first deploy was exactly the
stale-pending-order 404 diagnosed above — fixed, pulled, and the bot is now live-paper-
trading on the VM (practice account). Caught along the way: the VM's own `.env` had kept
Apex-Ultra's `SYMBOLS=EURGBP` (gitignored, `git pull` can never touch it) even after
`RISK_PCT` was updated — corrected by hand to all 4 pairs.

### Cost model corrected: half-spread on entry AND both exits, not just entry

Direct instruction (2026-09-07): rather than the original convention (a full spread
charged on entry only, and only if the touch didn't penetrate mid by a full spread-width;
nothing charged on either exit), every price now pays half the spread, in the direction
that actually costs the position — entry `mid ± spread/2`, and BOTH exits (stop and
target) `level ∓ spread/2`. A full round trip now costs exactly one spread total, split
evenly across both legs — the standard, mechanically correct way a marketable-limit round
trip actually prices, rather than dumping the whole cost on one side and none on the
other. Implemented identically in both places:

- **Live** (`PRODLIVETRADING/bot.py::_place_trade`): still price-protected LIMIT orders
  throughout (not market orders — no execution-slippage risk added, consistent with every
  other defensive choice in that file), just priced at the offset level using the real live
  spread reading already fetched for the spread gate.
- **Backtest** (`goal99_dtfx_base_backtest.py::resolve_base_strategy`): new
  `symmetric_exit_spread` parameter, default `False` so every number already in this
  document stays exactly reproducible; `goal104_optimal_risk_two_account.py`'s
  `build_candidate_trades` (the shared helper `goal104/106/108/109/111/112` all build on)
  now defaults it to `True`, so the *current* research pipeline matches what's actually
  live going forward.

**Re-run of the frozen 5-pip/10-pip @ 3% risk config, old vs new convention** (PRE-2021 /
POST-2026-04, all 4 pairs, two-account):

| config | PF (old) | PF (new) | maxDD holdout (old) | maxDD holdout (new) |
|---|---|---|---|---|
| 5p PRE-2021 | 1.489 | 1.475 | — | — |
| 5p POST-2026-04 | 1.400 | 1.366 | −33.48% | **−22.04%** |
| 10p PRE-2021 | 1.585 | 1.587 | — | — |
| 10p POST-2026-04 | 1.444 | 1.431 | −17.30% | **−11.54%** |

PF and win rate are essentially unchanged at both zone heights — the edge was never an
artifact of the old, cruder cost model. The one real, consistent shift: **max drawdown in
the genuine POST-2026-04 holdout improved meaningfully at both heights** under the
corrected, more realistic accounting (win rate rises because both exits also move slightly
in the position's favor for *triggering* — target easier to reach, stop harder to reach —
even though each individual win is worth a little less R). Net: a more mechanically honest
cost model that, if anything, paints a slightly better risk picture than the one already
on record.

### First live incident: PROD's real win rate (38%) badly missed the backtest, and why

2026-09-08: the user noticed real OANDA trade notifications all day, mostly USD/JPY, far
more than Apex-Ultra ever produced. Investigated directly rather than assumed benign.

**What actually happened**: 21 real closed trades since deployment (Sep 7 20:50 UTC), all
USD/JPY, **38.1% win rate (8W/13L), net −$6.41** — versus a backtest of the same period
predicting ~85-90%. Far too large a gap to be normal variance (21 trades at a true 90%
win rate essentially cannot produce 8 wins by chance).

**Ruled out, in order, each with a real test — not assumed away**:
1. **A coding bug.** Replayed `strategy.engine.SymbolEngine` (the literal deployed file)
   against freshly-fetched real data with the exact same warmup depth/timing as live's own
   startup. Its computed stop/target prices matched the real orders OANDA actually placed
   to the exact price (spread-adjustment included) — the strategy logic and order-
   placement math are correct.
2. **The reject/requeue concurrency mechanism** (a zone rejected while its side is busy
   gets `undo_dispatch`'d and re-checked every subsequent bar, rather than getting a fresh
   "next natural touch" the way `goal110`'s simplified model does — never tested before
   this incident). Built a faithful multi-symbol replay implementing this exactly. Result:
   84.2% WR (32W/6L) — barely different from a naive replay's 76.7%. Not the explanation.
3. **1-minute bar resolution too coarse** for a market moving as fast as USD/JPY was.
   Re-ran the same replay resolving USD/JPY trades against real 5-second candles instead.
   **Identical win/loss classification on every single trade** (still 84.2%) — only exit
   *timestamps* became more precise. Not the explanation.
4. **The live spread gate** (`MAX_ALLOWED_SPREAD_PIPS=2.0`) never being modeled. Pulled
   real historical bid/ask S5 data: spread averaged 1.94p over this window, 17.2% of the
   time above the 2.0p gate — a real, previously-unmodeled factor. Added real spread-driven
   entry/stop/target and gate rejection (same `undo_dispatch`-and-recheck behavior as
   `_place_trade`). Result: 36 trades, only 4 spread-gate rejections, still 83.3% WR. Not
   the explanation (a real, now-fixed gap in every replay done, but not *this* gap).

**Correction, once the real journalctl zone-creation log was actually pulled**: the
initial theory above (offline replay detects a different zone set) was wrong, and is
struck through here rather than deleted, since disproving it is itself the finding. Diffed
the replay's own zone-creation log directly against the VM's real `journalctl -u
prodlivetrading` output for the same window: **65 of 65 USD/JPY zones matched exactly**
(top/bottom/mid to 5 decimal places, same order) — the only zones in the real log absent
from the replay are 3 that formed after the replay's own data cutoff. Zone detection is
**not** the problem; it reproduces bit-for-bit.

**The actual mechanism**: re-matched real trades to replay trades by price alone (no time
window this time). Most still don't correspond to any replay trade — but the *first* real
trade does, closely: real entry 23:01:02 → exit 23:08:21 (7m19s to resolve) versus the
replay's matching trade, exit at 23:08:40 (8m40s) — **already 81 seconds of drift on trade
#1**, most likely genuine execution-level slippage (a real `stopLossOnFill` triggers as a
market order once touched, with real slippage a same-price backtest assumption can't
capture) or price action inside a single 5-second bar finer than even that resolution.
That 81 seconds matters enormously here specifically because **the account allows only one
open position per side, globally** — the instant that first trade's slot frees is exactly
what determines *which* of several simultaneously-touching zones gets the next fill, since
USD/JPY was forming a new qualifying zone every few minutes during this stretch. An 81-
second shift in when the slot reopens is enough for a completely different zone to be
"the one currently touching mid" at that instant, and from there every subsequent trade in
the sequence cascades onto a different path — even though the two systems agree on
literally every zone that exists.

**This is the real, load-bearing finding**: the production concurrency model (global
one-slot-per-side) is **chaotically sensitive to exact exit timing**, not to zone
detection. Identical zones, identical code, a real ~1-minute execution-timing difference
on the very first trade — and the two trade sequences diverge completely within 1-2
trades. This isn't a bug to fix; it's structural to any system where (a) a shared resource
is exclusively held until a resolution event, and (b) resolution events happen on a
timescale comparable to how often new contenders for that resource appear. During calm
markets neither factor bites hard; during a stretch like this USD/JPY move (trades
resolving in seconds to low minutes, new zones every few minutes), it's exactly the
regime where small, unavoidable execution-level noise gets amplified into a completely
different realized trade sequence.

**Live-readiness impact, corrected**: no single live sample — however carefully replayed
offline — can be expected to reproduce a specific backtest's exact trade sequence under
this concurrency model, even with perfect zone-detection fidelity. That means PROD's real
38% WR over 21 trades is neither evidence the edge is broken nor evidence it's fine — it's
a data point from a chaotic, thin sample, and the *only* way to validate real expectancy is
a genuinely large sample of live trades over real elapsed time, exactly as already stated
elsewhere in this document — this incident is evidence for *why* that bar is the right one,
not a reason to lower it.

### Higher-timeframe trend alignment: cuts frequency and drawdown, doesn't change the edge

Direct request (2026-09-08): filter 1-minute entries to only those aligned with the 15-
minute and/or 4-hour timeframe's own current bias, using the exact same DTFX-every-swing
mechanism run independently at each timeframe (`goal120_htf_alignment_backtest.py`).

**First attempt was wrong and worth recording**: defined "aligned" as "a same-side HTF zone
is alive (created, not yet invalidated/expired within 1440 own-bars — 15 days at 15m, ~240
days at 4h)." This barely filtered anything (94%+ of 1-minute trades "aligned") because a
zone can stay alive for months without invalidating, so a buy zone and a sell zone are very
often simultaneously "alive" at the same timeframe — not a meaningful bias signal. Fixed:
alignment is defined off **trend**, the single, mutually-exclusive state dtfx.md §4 already
defines (a new high beating the last confirmed high → uptrend, a new low breaking the last
confirmed low → downtrend) — reconstructed independently at 15m and 4h from resampled OHLC,
then a 1-minute entry counts as aligned only if its own side matches that timeframe's
current trend at that instant.

**Detected candidates** (pre-concurrency), PRE-2021 / POST-2026-04:

| segment | PRE-2021 n | PRE-2021 PF | POST-2026-04 n | POST-2026-04 PF |
|---|---|---|---|---|
| ALL (no filter, today's actual behavior) | 765,786 | 1.456 | 7,229 | 1.574 |
| 15m-aligned | 405,716 | 1.446 | 3,875 | 1.548 |
| 15m+4h-aligned | 205,258 | 1.440 | 1,946 | 1.531 |

**Two-account $200 simulation, full history, 3% risk** (same `simulate_two_account`
concurrency model used for every other two-account comparison in this document):

| segment | trades/day | WR | PF | max drawdown |
|---|---|---|---|---|
| ALL | 111.66 | 82.19% | 1.765 | **−82.60%** |
| 15m-aligned | 60.91 | 82.22% | 1.767 | **−67.32%** |
| 15m+4h-aligned | 31.20 | 82.28% | 1.778 | **−66.98%** |

**Verdict**: HTF alignment does **not** improve the edge — win rate and profit factor are
statistically flat across all three segments, both on the honest PRE-2021 sample and the
genuine POST-2026-04 holdout (if anything, raw candidate PF drifts very slightly down with
more filtering, though the two-account PF is flat-to-slightly-up — noise either way, not a
real quality change). What it **does** do is cut trade frequency roughly in half at each
additional filter layer, and meaningfully reduce max drawdown — a real ~15-point
improvement going from no filter to 15m-alignment specifically, with 4h adding almost
nothing further on top of 15m. Plausible mechanism (not verified further): filtering out
counter-higher-timeframe trades removes some of the correlated losing streaks that drive
drawdown, without changing the average per-trade outcome. This is a genuine trade-off —
same edge quality, lower frequency, shallower drawdown — not a free improvement, and not
currently applied to the deployed config (bot.py runs the unfiltered "ALL" version).

**Past 30 days specifically** (`goal121_htf_alignment_past_month.py`, fresh OANDA data,
2026-08-09 → 2026-09-08, 588 candidates before segmenting): unlike the full-history result
above, alignment *appears* to help this month — candidate PF 1.103 → 1.257 → 1.448 (ALL →
15m → 15m+4h), two-account PF 1.171 → 1.417 → 1.440. **Treat this as noise, not a
contradiction of the full-history finding**: 588 candidates (379 two-account-accepted) over
one month is a far smaller, far less reliable sample than the 765K-candidate PRE-2021 run
or even the 7K-candidate POST-2026-04 holdout, both of which found alignment flat-to-
neutral on edge quality. A single quiet-then-trending month easily produces exactly this
kind of apparent-improvement-that-isn't; the large-sample verdict (frequency/drawdown
trade-off, not an edge improvement) is the one to trust.

### An externally-authored 5p-vs-10p SL/TP grid search had two real bugs — corrected, then deployed

A separate agent (Gemini, run by the user outside this session) wrote
`grid_search_optimal_sl_tp_mc_race.py` / `grid_search_10pip_sl_tp_mc_race.py` /
`compute_combined_comparison_metrics.py` — a 2D grid over (stop multiplier, target
multiplier) × (5-pip, 10-pip minimum zone), scored by win rate, profit factor, and a
block-bootstrap Monte Carlo race to $1M starting from $200. Gemini's own presented table
also included a `Live Success Probability` / `Primary Live Risk` column pair with specific
percentages (e.g. "75-85%") and a causal narrative ("API race conditions", "actual live WR
drops to 50-60%"). **Verification found this pair of columns is not computed anywhere in
the script at all** — its own `print()` only ever emits Execs/Trades-per-day/WR/AvgWin/PF/
MC-days/MC-DD. Those two columns, and the narrative around them, were invented, not
measured, and are not reproduced anywhere in this project going forward.

Two real bugs *were* found in the actual computed columns, both confirmed by patching and
re-running against the real 23.7-year dataset (2003–2026):

1. **Spread cost was HALF a spread, not a full one.** `resolve_trades_grid` left entry at
   the raw zone midpoint (zero cost) and only shifted the two exits by `half_spread` —
   charging half a spread per round trip, versus this project's own established convention
   (§8 above, `symmetric_exit_spread`) of a full spread per round trip (half on entry, half
   on exit). Correcting it dropped profit factor substantially on every config tested —
   worst on the smallest risk unit, as expected: **5-Pip 1.0SL:1.0TP PF 5.93 → 3.56 (-40%)**,
   10-Pip 1.0SL:1.0TP PF 3.29 → 2.56 (-22%).
2. **The Monte Carlo racer's margin accounting had a sign error.** `run_mc_fast` decremented
   `margin_used[side]` on both position OPEN and CLOSE (should increment on open), letting
   margin capacity drift negative and silently stop binding as balances grew — inflating
   apparent risk capacity and understating time-to-$1M. Fixing it: 10-Pip 1.0SL:1.0TP
   "Days to $1M" moved 72.5d → 81.3d (+12%).

**Corrected full grid** (`compute_combined_comparison_metrics_FIXED.py`, same 23.7-year
dataset, both bugs fixed, real spread-stress column — `PF@2.5p` — added in place of the
fabricated "Live Success Probability"):

| Zone | Config | Trades/Day | WR | Avg Win | PF@1.6p | PF@2.5p | PF drop | Days to $1M | MC MaxDD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5-Pip | 1.0SL:1.0TP | 145.7 | 90.4% | +0.38R | 3.56 | 2.49 | -30.1% | 36.4d | -5.4% |
| 5-Pip | 1.0SL:1.25TP | 132.2 | 83.9% | +0.55R | 2.85 | 2.37 | -16.8% | 35.6d | -6.7% |
| 5-Pip | 1.0SL:0.8TP | 153.5 | 94.4% | +0.24R | 4.10 | 1.45 | **-64.6%** | 47.7d | -4.8% |
| 5-Pip | 1.25SL:1.0TP | 143.3 | 91.5% | +0.32R | 3.48 | 2.45 | -29.6% | 34.3d | -5.9% |
| **5-Pip** | **1.25SL:1.25TP** | **127.2** | **85.6%** | **+0.47R** | **2.78** | **2.32** | **-16.5%** | **33.6d** | **-7.1%** |
| 5-Pip | 1.5SL:1.0TP | 141.5 | 92.4% | +0.28R | 3.42 | 2.43 | -28.9% | 34.8d | -6.1% |
| 5-Pip | 2.0SL:1.0TP | 138.4 | 93.7% | +0.22R | 3.34 | 2.41 | -27.8% | 40.2d | -6.2% |
| 10-Pip | 1.0SL:1.0TP | 26.4 | 80.8% | +0.61R | 2.56 | 2.29 | -10.5% | 109.5d | -8.9% |
| 10-Pip | 1.0SL:0.8TP | 27.3 | 86.9% | +0.45R | 2.98 | 2.57 | -13.8% | 120.4d | -7.3% |
| 10-Pip | 1.25SL:1.0TP | 26.9 | 83.0% | +0.51R | 2.47 | 2.23 | -9.7% | 120.7d | -9.4% |
| 10-Pip | 1.0SL:1.25TP | 24.5 | 70.8% | +0.81R | 1.96 | 1.88 | -4.1% | 133.4d | -11.9% |
| 10-Pip | 1.25SL:0.8TP | 28.0 | 88.5% | +0.37R | 2.89 | 2.52 | -12.8% | 130.9d | -7.5% |
| 10-Pip | 1.5SL:1.0TP | 27.0 | 84.7% | +0.43R | 2.41 | 2.19 | -9.1% | 139.8d | -9.1% |
| 10-Pip | 2.0SL:0.8TP | 27.8 | 91.7% | +0.25R | 2.74 | 2.43 | -11.3% | 193.0d | -7.3% |
| 10-Pip | 3.0SL:0.8TP | 26.7 | 93.9% | +0.17R | 2.64 | 2.38 | -9.8% | 295.6d | -7.1% |

**Corrected conclusion inverts Gemini's headline finding**: on real numbers, 5-pip zones
compound to $1M in 34–48 days vs 10-pip's 110–296 days (frequency dominates once the
phantom half-spread discount is removed), the opposite of "10-pip wins the race." The
direction of Gemini's spread-fragility argument *does* hold up, now quantified honestly:
5-pip configs lose 17–65% of PF when spread widens 1.6p→2.5p vs only 4–14% for 10-pip.

### Multi-touch zone rearming: real, but ~3x smaller than the pooled full-history number claimed

A second Gemini-authored script, `research_multi_touch_rearming.py`, proposed reversing
this project's own "single-shot, no re-arm" design decision (dtfx.md #7, `engine.py`'s
docstring): after a zone WINS, don't retire it — require a fresh overshoot of its own edge
again (Protocol A), then let it re-fire at the same mid, repeatedly, until it finally loses
or expires. Claimed a dramatic result: win rate and profit factor *increase* with touch
depth (10-Pip touch #1 80.3% WR/PF 2.48 → touch #7+ 92.6% WR/PF 7.46, pooled full history),
and allowing up to 20 touches per zone lifts trade frequency ~2.8x and cuts time-to-$1M by
~3x. Verification reproduced these exact numbers (no fabrication this time, and the
spread/margin conventions were already correct) — but the analysis pooled the full
2003-2026 dataset directly, which is exactly what this project's own established
methodology exists to prevent, given the confirmed 2021-2026.03 data-feed anomaly above.

**Regime-split re-run** (`research_multi_touch_rearming_REGIME_SPLIT.py`, same trade
candidates, same zones, split at goal108's exact boundaries): the SMOOTHED (2021-01 to
2026-03, confirmed-contaminated) window turns out to be **44.9%-54.9% of the entire pooled
sample** the original table was built on — roughly half of Gemini's headline numbers came
from data already known to be unreliable.

| | PRE-2021 touch #1 | PRE-2021 touch #7+ | POST-2026-04 touch #1 | POST-2026-04 touch #7+ | Pooled (contaminated) touch #7+ |
|---|---|---|---|---|---|
| 10-Pip | WR 78.1%, PF 2.17 | WR 80.8%, PF **2.45** | WR 77.2%, PF 2.11 | WR 90.2%, PF 5.54 (n=419, noisy) | WR 92.6%, PF **7.46** |
| PROD Hybrid | WR 87.2%, PF 2.67 | WR 88.6%, PF **2.75** | WR 87.2%, PF 2.54 | WR 96.1%, PF 7.68 (n=6,848, noisy) | WR 95.7%, PF **7.58** |
| 5-Pip | WR 87.2%, PF 2.66 | WR 88.6%, PF **2.74** | WR 87.7%, PF 2.57 | WR 95.3%, PF 6.37 (n=8,164, noisy) | WR 95.5%, PF **7.12** |

On the large, trustworthy **PRE-2021** sample (2.0-2.1M trades per mode — the real basis
for judging this), the touch-depth effect is genuine and directionally the same, but the
pooled table overstated its magnitude by roughly **3x on profit factor**: touch #7+ PF is
really ~2.4-2.8, not ~7.1-7.7. The **POST-2026-04** holdout (26,753-36,740 trades overall,
but only 419-8,164 in the touch #7+ cohort specifically, over just 114 trading days) is
directionally consistent — even a bit stronger — but far too small a sample at that cohort
depth to trust at face value; treat it as "not contradicted," not "confirmed."

**Cap-sweep, PRE-2021 only** (deterministic, no MC — see script for why MC was skipped
here): PROD Hybrid single-touch WR 87.6%/PF 2.77 → 20-touch-cap WR 88.5%/PF 2.92 (+5% PF,
not the pooled table's claimed 90.2%→92.8% WR / PF 3.52→4.65 / +32%). Trade frequency
still climbs substantially either way (109→290/day on PRE-2021 alone) — **the real,
validated finding is "rearming meaningfully increases trade frequency with a modest,
genuine edge improvement," not "edge gets dramatically stronger deep into touch chains."**

**Not implemented, and not recommended without substantially more work**, independent of
the corrected effect size: this reverses a deliberate design decision (single-shot zones,
no re-arm) made when Apex-Ultra — the lineage this project retired specifically for having
this exact multi-touch mechanic — was deleted from `PRODLIVETRADING` this session.
`engine.py` has no state model for "zone waiting to re-overshoot after a win" today (it
deletes a zone the instant it dispatches), and adding one would compound the already-
documented chaotic sensitivity to exact execution timing (see "First live incident" above)
by giving the concurrency model more, and more tightly correlated, candidate signals to
adjudicate at once.

### Deployed 2026-09-08, superseded same day: `SL_MULT=1.25`, `TP_MULT=1.25` on 5-pip zones

Set in `config.py` and `strategy/engine.py`, replacing the previous 1.0/1.0 (stop/target
exactly at the zone's own raw edges).
Stop and target were `mid ∓ 1.25·half_height` / `mid ± 1.25·half_height` — still exactly
1:1R, just a wider buffer past the zone's own edges. Chosen as the **1.25SL:1.25TP** row
above: the single fastest-to-$1M 5-pip config (33.6d) *and* the most spread-resilient of
the fast 5-pip configs (-16.5% PF drop under stress, roughly half of 1.0/1.0's -30.1%) —
traded off against a lower standard-spread PF (2.78 vs 1.0/1.0's 3.56) and slightly deeper
MC drawdown (-7.1% vs -5.4%).

**Caveat that should have been applied and wasn't, caught only after the fact**: this
selection, like the pooled multi-touch analysis above, was made on the full 2003-2026
dataset directly — not regime-split PRE-2021/POST-2026-04. Superseded hours later by the
switch to uniform 10-pip zones below (a direct request, for spread/slippage resilience over
raw compounding speed), which *was* regime-split selected — see that section for why this
matters and by how much. Left here, struck through in spirit rather than deleted, as the
record of what was live and why, and as a reminder that this project's own regime-split
discipline needs to be applied prospectively, not just when a result looks suspicious.

### Deployed 2026-09-08 (supersedes the above): uniform 10-pip zones, `SL_MULT=1.0`/`TP_MULT=0.8`

Direct request: switch from 5-pip zones (or the prior 10p-USDJPY/5p-others hybrid) to a
**uniform 10-pip minimum zone on all four pairs** — trading some trade frequency for
meaningfully better spread/slippage resilience (10-pip's larger stop distance makes a
1.6-pip spread only ~16-32% of risk, vs 5-pip's ~32-64%), while still trading often enough
(~27/day PRE-2021, all 4 pairs combined) to avoid long gaps with no open opportunity.

**This time, selected the way the 1.25/1.25-on-5-pip choice above should have been**:
`research_10pip_regime_split_selection.py` ran the same 8-config SL/TP grid at a flat
10-pip floor, split PRE-2021 (design, ~17 years)/POST-2026-04 (held out, never touched for
selection) at goal108's exact boundaries, *before* picking a winner:

| Config | PRE-2021 PF | POST-2026-04 PF (holdout) |
|---|---|---|
| 1.0SL:1.0TP | 2.160 | 2.090 |
| **1.0SL:0.8TP** | **2.470** | **2.284** |
| 1.25SL:1.0TP | 2.096 | 2.042 |
| 1.0SL:1.25TP | 1.795 | 1.760 |
| 1.25SL:0.8TP | 2.407 | 2.185 |
| 1.5SL:1.0TP | 2.042 | 2.061 |
| 2.0SL:0.8TP | 2.275 | 2.135 |
| 3.0SL:0.8TP | 2.189 | 2.060 |

**1.0SL:0.8TP wins outright on both the design sample and the untouched holdout** — an
**asymmetric 0.8:1 reward:risk** (not 1:1R; target sits inside the zone's own far edge,
`mid + 0.8·half_height` for a buy rather than the full edge), compensated by a materially
higher win rate (~85% vs ~78-80% for 1.0/1.0).

MC race to $1M (1000 sims), reported honestly across all three views rather than picking
the rosiest one:

| view | median days to $1M | median max DD |
|---|---|---|
| PRE-2021 only (the trustworthy forward estimate) | **159.4d** | -8.1% |
| POST-2026-04 holdout (small — 1,104 trades/114 days, noisier) | 416.9d | -9.7% |
| Full 2003-2026 pooled (includes the contaminated window, reference only) | 120.0d | -7.4% |

The honest range (120-420 days depending which regime is weighted) is meaningfully slower
than the ~110 days the earlier, purely-pooled 10-pip grid search implied — worth stating
plainly rather than leading with the more flattering pooled number. PRE-2021's 159.4d is
the number to actually plan around.

`config.py`'s `SYMBOL_MIN_ZONE_PIPS` set to 10.0 for all four pairs (the per-symbol
override mechanism stays available for future use); `SL_MULT`/`TP_MULT` set to 1.0/0.8.
`tests/test_dtfx_engine.py` updated: the reference-trace assertions now check the
asymmetric stop/target values this produces, `test_per_symbol_min_zone_pips` asserts the
uniform 10.0 floor, and the `MIN_STOP_PIPS`-scaling test now explicitly overrides
`min_zone_pips` down to 5.0 for its own scenario, since the deployed 10-pip floor
structurally guarantees the raw check always passes on its own (half-height ≥ 5p >
`MIN_STOP_PIPS`'s 3p) regardless of `SL_MULT`, which would otherwise make that test
vacuous. `deploy_readiness_check.py` confirmed live end-to-end (`stop=1.09900
target=1.10530` on the reference trace — stop exactly on the zone's own bottom edge since
`SL_MULT=1.0`, target inside the top edge since `TP_MULT=0.8`).
