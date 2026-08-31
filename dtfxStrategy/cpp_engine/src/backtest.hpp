// Exact port of research_scripts/prod_filter_grid_search.py::_simulate_generic (numba).
// General-purpose multi-touch zone simulator: asymmetric stop/target, min fill age,
// optional bounce-gated re-arm (require_bounce=False = PROD's unconditional re-arm on any
// win, up to max_touches), zone resolution independent of zone invalidation. Every knob is
// a runtime argument -- nothing about a specific strategy is baked in here.
#pragma once
#include <cstdint>
#include <vector>

struct SimResult {
    std::vector<int64_t> touch_num;
    std::vector<int64_t> entry_idx;  // true fill bar, not the resolution bar
    std::vector<uint8_t> is_win;
    std::vector<double> mid;
    std::vector<double> stop_risk;
};

// zone_* arrays have length n_zones; times_ns/lows/highs have length n_bars.
SimResult simulate_generic(const double* zone_top, const double* zone_bottom, const double* zone_mid,
                            const uint8_t* zone_is_buy, const int64_t* zone_start_idx,
                            const int64_t* zone_created_ns, int64_t n_zones,
                            const int64_t* times_ns, const double* lows, const double* highs, int64_t n_bars,
                            int64_t min_fill_age_ns, double stop_mult, double target_mult,
                            bool require_bounce, double probe_depth, double exc_mult,
                            int64_t max_touches, int64_t max_window);
