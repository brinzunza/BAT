"""The shared event-driven settlement core: apply P&L only at a trade's real exit time, free
margin only then, and check each pyramid addition's own incremental margin at its own real
timestamp. This is the fix for a documented "portfolio-accounting lookahead" bug that once
credited a trade's complete eventual P&L at entry instead of exit, letting overlapping
positions be sized off profits that did not exist yet.

Extracted out of `research_scripts/beid_goal_hunt/margin_aware_mc.py`'s `settle_through`
closure, which was copy-pasted near-verbatim into `strategy_ranking_2026.py`'s
`replay_real_path` -- the duplication itself already caused a real drift bug: the copy in
`strategy_ranking_2026.py` silently stopped honoring an `account`-column sizing-key override
that the original respects. One shared implementation removes that whole class of drift.

Deliberately a lower-level primitive than "the Monte Carlo loop" or "the real-path replay loop"
-- it owns only the event queue and per-key equity/margin bookkeeping. What "done" means (a
target-years-crossing for MC, a running equity curve for a real chronological replay) is left
to the caller, who calls `settle_through()` and inspects `.equity`/`.alive` afterward.
"""
from __future__ import annotations

import heapq
from typing import Any


class EventSettler:
    def __init__(self, keys, start_equity: dict, *, leg_floor: float | None = None):
        self.equity: dict = dict(start_equity)
        self.used_margin: dict = {k: 0.0 for k in keys}
        self.alive: dict = {k: True for k in keys}
        self.leg_floor = leg_floor
        self.pending: list = []  # heap: (event_ns, priority, sequence, event_type, parent_id, payload)
        self.parents: dict = {}
        self.sequence = 0
        self.skipped = 0

    def available_margin(self, key) -> float:
        return self.equity[key] - self.used_margin[key]

    def settle_through(self, timestamp_ns: int, on_settle=None):
        """Pops and applies every pending event with event_ns <= timestamp_ns, in
        (time, priority, sequence) order -- priority 0 (pyramid adds) before 1 (exits) on an
        exact tie, sequence breaking any remaining tie deterministically.

        `on_settle(key, pnl, exit_payload)`, when given, is called immediately after EACH
        individual exit event is applied (not for adds, which never realize P&L on their own).
        If it returns a value that is not None, processing stops immediately -- even if more
        events remain at or before `timestamp_ns` -- and that value is returned from
        `settle_through()`. This preserves exact per-event early-stop semantics (e.g. "return
        the instant total equity crosses a target, using THAT event's own payload") rather than
        batching every event up to timestamp_ns and only checking afterward, which would
        silently change which event's payload gets reported as the crossing point whenever
        more than one exit lands in the same `settle_through` call.

        Returns None if the queue up to `timestamp_ns` is exhausted without `on_settle` ever
        returning non-None (or if `on_settle` is None)."""
        while self.pending and self.pending[0][0] <= timestamp_ns:
            event_ns, _, _, event_type, parent_id, payload = heapq.heappop(self.pending)
            parent = self.parents.get(parent_id)
            if parent is None:
                continue
            key = parent["key"]
            if event_type == "add":
                add_price, add_r = payload
                add_margin = parent["base_margin"] * add_price / parent["base_entry"]
                if add_margin <= self.equity[key] - self.used_margin[key]:
                    self.used_margin[key] += add_margin
                    parent["reserved"] += add_margin
                    parent["accepted_r"] += add_r
                else:
                    self.skipped += 1
                continue

            self.used_margin[key] = max(0.0, self.used_margin[key] - parent["reserved"])
            pnl = parent["risk_dollars"] * parent["accepted_r"]
            self.equity[key] += pnl
            del self.parents[parent_id]
            if self.leg_floor is not None and self.equity[key] <= self.leg_floor:
                self.alive[key] = False
            if on_settle is not None:
                result = on_settle(key, pnl, payload)
                if result is not None:
                    return result
        return None

    def open_position(self, key, *, entry_ns: int, exit_ns: int, margin: float,
                       risk_dollars: float, entry_price: float, unit_rs: list[float],
                       unit_prices: list[float], unit_offsets_ns: list[int],
                       exit_payload: Any = None) -> None:
        """Registers a new position, including any pyramid adds, into the event queue. The
        caller must already have checked available margin for the base/first unit and is
        expected to call this immediately after that check (this method itself reserves the
        base unit's margin unconditionally -- it does not re-check availability, matching the
        existing call pattern in margin_aware_mc.py/strategy_ranking_2026.py where the check
        happens once, right before sizing)."""
        self.sequence += 1
        parent_id = self.sequence
        self.used_margin[key] += margin
        self.parents[parent_id] = dict(
            key=key, base_margin=margin, base_entry=float(entry_price), reserved=margin,
            risk_dollars=risk_dollars, accepted_r=float(unit_rs[0]),
        )
        for add_offset, add_price, add_r in zip(unit_offsets_ns[1:], unit_prices[1:], unit_rs[1:]):
            self.sequence += 1
            heapq.heappush(self.pending, (entry_ns + int(add_offset), 0, self.sequence,
                                           "add", parent_id, (float(add_price), float(add_r))))
        self.sequence += 1
        heapq.heappush(self.pending, (exit_ns, 1, self.sequence, "exit", parent_id, exit_payload))

    def total_equity(self) -> float:
        return sum(self.equity.values())

    def any_alive(self) -> bool:
        return any(self.alive.values())
