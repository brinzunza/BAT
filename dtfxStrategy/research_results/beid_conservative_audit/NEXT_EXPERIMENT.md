# Fixed next experiment — 2026-09-09

Previous turn produced evidence and code, not a verified profitable strategy.

Raw-data audit found two invalid/suspect EURUSD bars in 2001. Other instruments passed the same broad checks. Do not silently remove those bars and preserve their downstream state. Restart every instrument at 2003-01-01, retain 2003 as warmup, and score from 2004. This is a revised research sample, not an untouched holdout. Keep full-history outputs as invalidated diagnostics.

Rebuild the existing 3-minute signals, replay their frozen order set on one-minute OHLC, and compare the same 18 zone/type/hour combinations with and without one additional preregistered condition: trade direction must agree with the latest fully closed hourly candle relative to its 20-observation EMA, and the EMA slope. This is a proxy for trend, not the BEID structural trend. No countertrend option. Total new grid: 36, selected exclusively on pre-2016 mean R at 2.4 pip costs, minimum 500 trades. Validation: 2016–2020, 2021–March 2026, and reused April-2026 onward. Never select from validation results.

Preserve one-position portfolio and one-touch rule. Report 1.6/2.4/3.2-pip scenarios, calendar-block confidence bounds on validation expectancy, and drawdown limitations. Reject if validation is negative. The existing five-second bid/ask caches span roughly two weeks and may provide a further execution check, but cannot establish statistically durable profitability.
