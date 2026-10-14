# Continuation checkpoint

This goal turn made progress: found raw data corruption, invalidated old performance, rebuilt all symbols from 2003 with 2004 scoring, replayed on one-minute OHLC, tested a preregistered 36-configuration grid including completed-hour alignment, quantified uncertainty, and implemented quote-side checks. Twelve regression tests pass. The goal is not complete.

Frozen candidate: >=40p flip zones, hourly EMA20 direction and slope aligned, single touch, one portfolio position. Selection uses training only. See evaluation.json and validation_uncertainty.json. Base-cost historical periods are positive, but 2016–2020 cost stress fails and all later confidence intervals cross zero.

Processes started in this turn are terminal. Public quote probe attempted the same 32 hash-ordered hours twice: 21 complete, 11 transient failures. Cached compressed raw files are in trading-data/beid_quote_cost_probe. The entire frozen entry-hour schedule has 385 unique hours. Available S5 quote cache has zero eligible candidate trades.

Next meaningful work: validate price/time alignment between master bars and downloaded public quotes; expand quote coverage with bounded, resumable acquisition. Cost-only entry snapshots cannot establish exact fills or portfolio returns. A full execution study must include original candidate orders and actual expiry/occupancy, not just sample already-selected profitable trades. Do not tune on reused later results or infer profitability from 0.8p median sampled spread.

Concurrent workspace additions appeared during this turn: beid_real_engine.py, discover_champion_beid_strategy.py, run_beid_ultimate_sweep.py, test_advanced_beid_features.py. These were not created or edited here. Inspect their current results if relevant, preserve others' changes, and audit assumptions before relying on them.
