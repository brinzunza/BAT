// Glue: raw OHLC bars -> packed zone arrays, mirroring the exact output contract of
// dtfx_robust_search.py::_prep_zones (top, bottom, mid, is_buy, created_at_ns, start_idx).
//
// start_idx = np.searchsorted(times, created_at, side="right") in the Python original --
// equivalent to std::upper_bound on the int64 nanosecond timestamps here (times_ns is
// guaranteed strictly ascending, unique -- see load_instrument()'s sort+dedup).
#pragma once
#include <cstdint>
#include <vector>

struct ZoneArrays {
    std::vector<double> top, bottom, mid;
    std::vector<uint8_t> is_buy;
    std::vector<int64_t> created_at_ns;
    std::vector<int64_t> start_idx;
};

ZoneArrays detect_zones(const double* open, const double* high, const double* low,
                         const double* close, const int64_t* times_ns, int64_t n_bars,
                         int min_retracement);
