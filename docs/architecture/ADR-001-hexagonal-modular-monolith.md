# ADR-001: Adopt a hexagonal modular monolith

**Status:** Accepted (2026-07-12)

## Context

BAT combined strategy execution, portfolio accounting, provider SDK calls and
CLI orchestration. `BacktestEngine` and `LiveTradingEngine` exposed different
execution paths, making it difficult to test trading rules independently of
Alpaca, Polygon, or interactive prompts.

## Decision

We introduce `src/core/` as the supported application core:

* `domain`: value objects and deterministic portfolio accounting.
* `application`: use cases, initially `BacktestService`.
* `ports`: structural contracts for strategies, market data, and execution.
* `adapters`: anti-corruption wrappers over legacy providers and brokers.
* `bootstrap`: the composition root.

Existing `engines`, `data_providers`, strategies, and CLI remain compatibility
adapters during the incremental migration. The backtest engine delegates to
the application service, preserving its public dataframe-oriented API.

## Consequences and trade-offs

* Trading rules can be verified without API credentials or network access.
* Providers and brokers can be replaced behind explicit ports.
* Pandas remains at the strategy boundary for compatibility; replacing it with
  a `Bar` stream is deferred because it would require rewriting all strategies.
* We deliberately do not use microservices or a message broker. BAT currently
  has one deployable and needs deterministic, auditable execution more than
  distributed scaling. An internal event/outbox abstraction may be added when
  durable live-order auditing is required.
* `LiveTradingEngine` is not rewritten in this change: it remains a legacy
  adapter because modifying live order handling without staged broker testing
  would be unsafe. New live orchestration must consume `ExecutionGateway`.

## Validation

`tests/test_backtest_service.py` covers long-only accounting, long/short
reversal, spread costs, and market-data contract validation.
