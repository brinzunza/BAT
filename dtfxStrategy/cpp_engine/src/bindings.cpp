// pybind11 module `dtfx_engine`: exposes detect_zones, simulate_generic, run_mc_capped as
// near-drop-in replacements for the Python/numba originals. Every strategy/account parameter
// stays a Python-level function argument -- nothing is precomputed or baked in here.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstring>
#include <vector>
#include "zone_detect.hpp"
#include "backtest.hpp"
#include "monte_carlo.hpp"
#include "features.hpp"

namespace py = pybind11;

namespace {

py::array_t<double> to_f64(const std::vector<double>& v) {
    py::array_t<double> out(v.size());
    if (!v.empty()) std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(double));
    return out;
}

py::array_t<int64_t> to_i64(const std::vector<int64_t>& v) {
    py::array_t<int64_t> out(v.size());
    if (!v.empty()) std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(int64_t));
    return out;
}

py::array_t<bool> to_bool(const std::vector<uint8_t>& v) {
    py::array_t<bool> out(v.size());
    bool* p = out.mutable_data();
    for (size_t i = 0; i < v.size(); ++i) p[i] = v[i] != 0;
    return out;
}

using f64arr = py::array_t<double, py::array::c_style | py::array::forcecast>;
using i64arr = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;
using boolarr = py::array_t<bool, py::array::c_style | py::array::forcecast>;

py::tuple py_detect_zones(f64arr open, f64arr high, f64arr low, f64arr close, i64arr times_ns,
                           int min_retracement) {
    const int64_t n_bars = open.shape(0);
    if (high.shape(0) != n_bars || low.shape(0) != n_bars || close.shape(0) != n_bars ||
        times_ns.shape(0) != n_bars) {
        throw std::invalid_argument("open/high/low/close/times_ns must all have the same length");
    }
    ZoneArrays z = detect_zones(open.data(), high.data(), low.data(), close.data(),
                                 times_ns.data(), n_bars, min_retracement);
    return py::make_tuple(to_f64(z.top), to_f64(z.bottom), to_f64(z.mid), to_bool(z.is_buy),
                           to_i64(z.created_at_ns), to_i64(z.start_idx));
}

py::tuple py_simulate_generic(f64arr zone_top, f64arr zone_bottom, f64arr zone_mid,
                               boolarr zone_is_buy, i64arr zone_start_idx, i64arr zone_created_ns,
                               i64arr times_ns, f64arr lows, f64arr highs, int64_t min_fill_age_ns,
                               double stop_mult, double target_mult, bool require_bounce,
                               double probe_depth, double exc_mult, int64_t max_touches,
                               int64_t max_window) {
    const int64_t n_zones = zone_top.shape(0);
    const int64_t n_bars = lows.shape(0);

    std::vector<uint8_t> is_buy_u8(n_zones);
    {
        const bool* p = zone_is_buy.data();
        for (int64_t i = 0; i < n_zones; ++i) is_buy_u8[i] = p[i] ? 1 : 0;
    }

    SimResult r = simulate_generic(zone_top.data(), zone_bottom.data(), zone_mid.data(),
                                    is_buy_u8.data(), zone_start_idx.data(), zone_created_ns.data(),
                                    n_zones, times_ns.data(), lows.data(), highs.data(), n_bars,
                                    min_fill_age_ns, stop_mult, target_mult, require_bounce,
                                    probe_depth, exc_mult, max_touches, max_window);
    return py::make_tuple(to_i64(r.touch_num), to_i64(r.entry_idx), to_bool(r.is_win),
                           to_f64(r.mid), to_f64(r.stop_risk));
}

py::tuple py_simulate_generic_ex(f64arr zone_top, f64arr zone_bottom, f64arr zone_mid,
                                  boolarr zone_is_buy, i64arr zone_start_idx, i64arr zone_created_ns,
                                  i64arr times_ns, f64arr lows, f64arr highs, int64_t min_fill_age_ns,
                                  double stop_mult, double target_mult, bool require_bounce,
                                  double probe_depth, double exc_mult, int64_t max_touches,
                                  int64_t max_window) {
    const int64_t n_zones = zone_top.shape(0);
    const int64_t n_bars = lows.shape(0);

    std::vector<uint8_t> is_buy_u8(n_zones);
    {
        const bool* p = zone_is_buy.data();
        for (int64_t i = 0; i < n_zones; ++i) is_buy_u8[i] = p[i] ? 1 : 0;
    }

    SimResult r = simulate_generic(zone_top.data(), zone_bottom.data(), zone_mid.data(),
                                    is_buy_u8.data(), zone_start_idx.data(), zone_created_ns.data(),
                                    n_zones, times_ns.data(), lows.data(), highs.data(), n_bars,
                                    min_fill_age_ns, stop_mult, target_mult, require_bounce,
                                    probe_depth, exc_mult, max_touches, max_window);
    return py::make_tuple(to_i64(r.touch_num), to_i64(r.entry_idx), to_i64(r.exit_idx),
                           to_bool(r.is_win), to_f64(r.mid), to_f64(r.stop_risk));
}

py::tuple py_run_mc_capped(f64arr net_r, int64_t n_sims, int64_t block, int64_t max_trades,
                            uint32_t seed, double start, double risk_frac, double cap,
                            double target1, double target2, double ruin_floor, int n_threads) {
    const int64_t n = net_r.shape(0);
    MCResult r = run_mc_capped(net_r.data(), n, n_sims, block, max_trades, seed, start, risk_frac,
                                cap, target1, target2, ruin_floor, n_threads);
    return py::make_tuple(to_i64(r.trades_to_1m), to_i64(r.trades_to_10m), to_f64(r.max_dd),
                           to_bool(r.ruined));
}

py::tuple py_compute_regime_features(f64arr high, f64arr low, f64arr close, int64_t window,
                                      int64_t stride, int64_t vol_lookback_points,
                                      int n_threads) {
    const int64_t n_bars = close.shape(0);
    RegimeFeatures r = compute_regime_features(high.data(), low.data(), close.data(), n_bars,
                                                window, stride, vol_lookback_points, n_threads);
    return py::make_tuple(to_f64(r.efficiency_ratio), to_f64(r.hurst_est),
                           to_f64(r.vol_percentile), to_f64(r.realized_vol));
}

}  // namespace

PYBIND11_MODULE(dtfx_engine, m) {
    m.doc() = "C++ backend for DTFX zone detection, backtest simulation, and Monte Carlo. "
              "Drop-in replacement for the numba/pure-Python originals in research_scripts/ "
              "-- every strategy/account parameter is a runtime argument.";

    m.def("detect_zones", &py_detect_zones,
          py::arg("open"), py::arg("high"), py::arg("low"), py::arg("close"),
          py::arg("times_ns"), py::arg("min_retracement"),
          "Pivot detection + structure FSM -> packed zone arrays "
          "(top, bottom, mid, is_buy, created_at_ns, start_idx). "
          "Mirrors dtfx_robust_search.py::_prep_zones' output exactly.");

    m.def("simulate_generic", &py_simulate_generic,
          py::arg("zone_top"), py::arg("zone_bottom"), py::arg("zone_mid"), py::arg("zone_is_buy"),
          py::arg("zone_start_idx"), py::arg("zone_created_ns"), py::arg("times_ns"),
          py::arg("lows"), py::arg("highs"), py::arg("min_fill_age_ns"), py::arg("stop_mult"),
          py::arg("target_mult"), py::arg("require_bounce"), py::arg("probe_depth"),
          py::arg("exc_mult"), py::arg("max_touches"), py::arg("max_window"),
          "General-purpose multi-touch trade simulator -> "
          "(touch_num, entry_idx, is_win, mid, stop_risk). "
          "Exact port of prod_filter_grid_search.py::_simulate_generic.");

    m.def("simulate_generic_ex", &py_simulate_generic_ex,
          py::arg("zone_top"), py::arg("zone_bottom"), py::arg("zone_mid"), py::arg("zone_is_buy"),
          py::arg("zone_start_idx"), py::arg("zone_created_ns"), py::arg("times_ns"),
          py::arg("lows"), py::arg("highs"), py::arg("min_fill_age_ns"), py::arg("stop_mult"),
          py::arg("target_mult"), py::arg("require_bounce"), py::arg("probe_depth"),
          py::arg("exc_mult"), py::arg("max_touches"), py::arg("max_window"),
          "Same as simulate_generic, plus exit_idx (the resolution bar -- when the trade "
          "actually closes, needed for position-duration/concurrency/margin modeling) -> "
          "(touch_num, entry_idx, exit_idx, is_win, mid, stop_risk).");

    m.def("run_mc_capped", &py_run_mc_capped,
          py::arg("net_r"), py::arg("n_sims"), py::arg("block"), py::arg("max_trades"),
          py::arg("seed"), py::arg("start") = 200.0, py::arg("risk_frac") = 0.05,
          py::arg("cap") = 2000000.0, py::arg("target1") = 1e6, py::arg("target2") = 1e7,
          py::arg("ruin_floor") = 1.0, py::arg("n_threads") = 0,
          "Position-sizing-capped block-bootstrap Monte Carlo, multi-threaded -> "
          "(trades_to_1m, trades_to_10m, max_drawdown, ruined). "
          "Account params default to today's assumptions but are all runtime overridable.");

    m.def("compute_regime_features", &py_compute_regime_features,
          py::arg("high"), py::arg("low"), py::arg("close"), py::arg("window") = 500,
          py::arg("stride") = 50, py::arg("vol_lookback_points") = 200, py::arg("n_threads") = 0,
          "Strided-grid, forward-filled regime features over the full bar history -> "
          "(efficiency_ratio, hurst_est, vol_percentile, realized_vol), each length n_bars. "
          "efficiency_ratio: Kaufman ER (0=choppy, 1=trending). hurst_est: fast two-scale R/S "
          "Hurst estimate (<0.5 mean-reverting, >0.5 trending). vol_percentile: current "
          "window's realized vol ranked against the trailing vol_lookback_points grid points "
          "(0=quietest recently seen, 1=most volatile). Recomputed only every `stride` bars "
          "from the trailing `window` bars, held constant between grid points -- a regime "
          "read has no business refreshing every single bar.");
}
