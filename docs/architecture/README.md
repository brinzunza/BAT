# BAT Architecture

BAT is a hexagonal modular monolith. The `bat.domain` package owns business
rules and has no broker, HTTP, UI, or plotting dependency. `bat.application`
contains use cases. `bat.ports` declares contracts and `bat.adapters` converts
legacy providers and brokers to those contracts. The CLI and legacy `engines`
are inbound adapters retained for backwards compatibility.

Dependency direction is always: adapters -> ports/application -> domain.

## Module ownership

| Module | Owns | Must not depend on |
| --- | --- | --- |
| `bat.domain` | Portfolio state, orders, fills and accounting | pandas, UI, SDKs, HTTP |
| `bat.application` | Backtest workflow and validation | concrete providers/brokers |
| `bat.ports` | Stable strategy, market-data, and execution contracts | SDKs |
| `bat.adapters` | Legacy/API translation | domain implementation details |
| `engines`, `ui`, `data_providers` | Compatibility and delivery mechanisms | ownership of new business rules |

The legacy live engine is intentionally an adapter at the boundary during this
migration. Do not add domain behavior there; new live workflows should call
application services through `MarketDataProvider` and `ExecutionGateway`.
