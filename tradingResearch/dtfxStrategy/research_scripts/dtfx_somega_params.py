"""SOMEGA -- Session-filtered Omega. NOT deployed anywhere; this is the canonical
definition of the strategy variant for future backtests/scripts to import, so the exact
rule doesn't drift or get reconstructed from memory.

Base zone-construction/simulation parameters are IDENTICAL to the live Omega config in
PRODLIVETRADING/strategy/engine.py -- SOMEGA changes nothing about zone detection, stop/
target placement, or the RSI/ATR filter. The only addition is a per-instrument UTC-hour
entry filter, discovered by testing win rate against hour-of-day (dtfx_omega_winloss_
factors.py), then backtested per instrument (dtfx_omega_stop_cost_ratio.py /
dtfx_omega_generalize_other_tickers.py).

Status as of 2026-08-25: research-only. Explicitly NOT wired into the live/paper bot.

Per-instrument hour rule (UTC), and why:
  - EURGBP: hour >= 18       (chi2 p=3.2e-10 on the full unfiltered hour breakdown)
  - GBPUSD: hour in 20,21,22 (chi2 p=1.4e-3)
  - USDJPY: hour in 19,20,21,22 (chi2 p=2.5e-3)
  - EURUSD: NO FILTER -- hour-of-day showed no significant effect (p=0.22). SOMEGA on
    EUR/USD is just Omega, unchanged.
  - XAUUSD: NO FILTER DEFINED -- hour-of-day IS significant (p=5.4e-3) but in the OPPOSITE
    direction (busy London/NY hours score better, not late hours). Applying the EURGBP/
    GBPUSD/USDJPY rule here would hurt, not help. A gold-specific busy-hours rule was not
    backtested as an exclusion filter -- flagged as a follow-up, not part of SOMEGA today.
  - BTCUSD: NO FILTER -- no significant hour-of-day effect (p=0.17), consistent with crypto
    having no session structure to produce one.

Backtested results (asymmetric cost model -- spread charged on stop-outs only -- flat
spread per instrument, illustrative/not measured for anything but EUR/USD's 1.6-pip live
account reading): see trading-data/SOMEGA_definition.md for the full table. Headline
numbers, baseline Omega -> SOMEGA:
  EURGBP:  Sharpe 1.57 -> 5.76,  PF 1.15 -> 1.97,  maxdd 33.0% -> 8.4%,  win rate 58.9% -> 70.9%
  GBPUSD:  Sharpe 2.35 -> 5.93,  PF 1.22 -> 2.05,  maxdd 35.3% -> 8.3%,  win rate 58.7% -> 70.2%
  USDJPY:  Sharpe 1.68 -> 5.25,  PF 1.14 -> 1.86,  maxdd 29.2% -> 7.1%,  win rate 57.6% -> 68.7%

Trade-off: this cuts trade frequency to roughly 5-13% of baseline Omega on the filtered
instruments. It buys a much steadier, lower-drawdown ride at the cost of most of the
opportunity set -- a real choice, not a strict improvement, and worth stating plainly
rather than only citing the flattering numbers above.

Caveats carried over from the underlying research (repeat here so this file is
self-contained, not just a pointer):
  - No real historical spread data exists for any of these instruments; costs are modeled,
    not measured (except the one live EUR/USD spread snapshot this project took: 1.6 pips).
  - The specific hour cutoffs (18, 20-22, 19-22) were chosen by inspecting the same data
    they're then backtested on -- the DIRECTION of the effect is well-supported (three
    independent pairs, low p-values on the un-cherry-picked full 24-hour breakdown), but
    the exact boundary hasn't been validated out-of-sample on data these thresholds weren't
    chosen from.
"""

BASE_OMEGA_PARAMS = dict(
    min_retracement=2,
    n_multiplier=3.343496,
    min_risk_pct=0.000170,
    min_fill_age_hours=3.366682,
    use_rsi_atr_filter=True,
)

# instrument -> set of UTC hours (0-23) an entry must fall in to be taken. Absence of a key
# means no filter -- trade every hour, i.e. plain Omega.
SESSION_HOUR_FILTER = {
    "EURGBP": set(range(18, 24)),      # hour >= 18
    "GBPUSD": {20, 21, 22},
    "USDJPY": {19, 20, 21, 22},
}


def passes_somega_filter(instrument, entry_hour_utc):
    """True if a zone's entry hour is allowed under SOMEGA for this instrument. Instruments
    with no defined filter (EURUSD, XAUUSD, BTCUSD, or anything else not in the dict above)
    always pass -- SOMEGA reduces to plain Omega for them."""
    allowed_hours = SESSION_HOUR_FILTER.get(instrument)
    if allowed_hours is None:
        return True
    return entry_hour_utc in allowed_hours
