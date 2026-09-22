// Market-structure state machine and zone construction. Exact port of
// PRODLIVETRADING/strategy/structure_fsm.py::run_structure_fsm (identical algorithm to
// tradingResearch/dtfxStrategy/research_scripts/dtfx_eurusd_backtest.py::run_structure_fsm).
//
// Uptrend: confirmed_high updates immediately on any new higher high. confirmed_low only
// gets "confirmed" once a high beats confirmed_high. Trend flips to DOWN the moment a low
// prints below confirmed_low. Downtrend is symmetric. Each newly-confirmed extreme creates a
// zone: [previous confirmed extreme on the far side, the extreme just being validated],
// entry at 50%, target ("tp") at the trigger price that created the zone.
//
// NOTE: the Python original also returns an `events` list (annotated zigzag with
// bootstrap/pending/confirmed/flip tags). Nothing downstream of run_structure_fsm in the
// actual research pipeline (_prep_zones -> simulate_generic) consumes `events` -- only
// `zones` -- so this port only tracks what `zones` construction actually needs (the two
// pending pivot values) and does not reconstruct the tagged event log.
#pragma once
#include <cstdint>
#include <vector>
#include "pivots.hpp"

struct Zone {
    double top, bottom, mid, tp;
    int64_t created_at_ns;
    bool is_buy;  // true = "buy" (long) zone, false = "sell" (short) zone
};

std::vector<Zone> run_structure_fsm(const std::vector<Pivot>& zigzag);
