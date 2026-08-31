#include "structure_fsm.hpp"

std::vector<Zone> run_structure_fsm(const std::vector<Pivot>& zigzag) {
    std::vector<Zone> zones;

    // Bootstrap: scan forward until we've seen at least one L and one H.
    size_t i = 0;
    bool have_low = false, have_high = false;
    double seen_low_v = 0.0, seen_high_v = 0.0;
    while (i < zigzag.size() && (!have_low || !have_high)) {
        const Pivot& p = zigzag[i];
        if (p.type == 'L' && !have_low) { have_low = true; seen_low_v = p.price; }
        if (p.type == 'H' && !have_high) { have_high = true; seen_high_v = p.price; }
        ++i;
    }
    if (!(have_low && have_high)) return zones;

    double confirmed_low = seen_low_v;
    double confirmed_high = seen_high_v;
    bool trend_up = true;

    bool have_pending_low = false, have_pending_high = false;
    double pending_low_v = 0.0, pending_high_v = 0.0;

    for (; i < zigzag.size(); ++i) {
        const Pivot& p = zigzag[i];
        const double v = p.price;
        const int64_t t = p.time_ns;

        if (p.type == 'L') {
            if (trend_up) {
                if (v < confirmed_low) {
                    confirmed_low = v;  // confirmed_high unchanged
                    trend_up = false;
                    have_pending_high = false;
                } else {
                    pending_low_v = v;
                    have_pending_low = true;
                }
            } else {  // trend down: lower low re-confirms
                if (v < confirmed_low) {
                    const double old_low = confirmed_low;
                    const double zone_top = have_pending_high ? pending_high_v : confirmed_high;
                    confirmed_low = v;
                    if (have_pending_high) {
                        confirmed_high = pending_high_v;
                        have_pending_high = false;
                    }
                    if (zone_top > old_low) {
                        zones.push_back(Zone{zone_top, old_low, (zone_top + old_low) / 2.0,
                                              v, t, false});
                    }
                } else {
                    pending_low_v = v;
                    have_pending_low = true;
                }
            }
        } else {  // 'H'
            if (!trend_up) {
                if (v > confirmed_high) {
                    confirmed_high = v;  // confirmed_low unchanged
                    trend_up = true;
                    have_pending_low = false;
                } else {
                    pending_high_v = v;
                    have_pending_high = true;
                }
            } else {  // trend up: higher high re-confirms
                if (v > confirmed_high) {
                    const double old_high = confirmed_high;
                    const double zone_bottom = have_pending_low ? pending_low_v : confirmed_low;
                    confirmed_high = v;
                    if (have_pending_low) {
                        confirmed_low = pending_low_v;
                        have_pending_low = false;
                    }
                    if (old_high > zone_bottom) {
                        zones.push_back(Zone{old_high, zone_bottom, (old_high + zone_bottom) / 2.0,
                                              v, t, true});
                    }
                } else {
                    pending_high_v = v;
                    have_pending_high = true;
                }
            }
        }
    }
    return zones;
}
