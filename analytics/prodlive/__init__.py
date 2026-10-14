"""Analytics pipeline over the PRODLIVETRADING bot's SQLite store (trades, account_snapshots, system_events, open positions).
sync.py pulls a consistent copy of the bot's trades.db (local path or over SSH) -> queries.py computes analytics -> __main__ is the CLI."""
