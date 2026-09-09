#include "zone_detect.hpp"
#include "pivots.hpp"
#include "structure_fsm.hpp"
#include <algorithm>

ZoneArrays detect_zones(const double* open, const double* high, const double* low,
                         const double* close, const int64_t* times_ns, int64_t n_bars,
                         int min_retracement) {
    std::vector<Pivot> zigzag = find_pivots_strict(open, high, low, close, times_ns, n_bars, min_retracement);
    std::vector<Zone> zones = run_structure_fsm(zigzag);

    ZoneArrays out;
    const size_t n = zones.size();
    out.top.reserve(n);
    out.bottom.reserve(n);
    out.mid.reserve(n);
    out.is_buy.reserve(n);
    out.created_at_ns.reserve(n);
    out.start_idx.reserve(n);

    const int64_t* times_end = times_ns + n_bars;
    for (const Zone& z : zones) {
        out.top.push_back(z.top);
        out.bottom.push_back(z.bottom);
        out.mid.push_back(z.mid);
        out.is_buy.push_back(z.is_buy ? 1 : 0);
        out.created_at_ns.push_back(z.created_at_ns);
        const int64_t idx = std::upper_bound(times_ns, times_end, z.confirmed_ns) - times_ns;
        out.start_idx.push_back(idx);
    }
    return out;
}
