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
#include "two_account.hpp"
#include "portfolio_mc.hpp"

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
using i32arr = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;
using boolarr = py::array_t<bool, py::array::c_style | py::array::forcecast>;
using u8arr = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;

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

py::tuple py_position_size_and_margin_debug(int32_t instr_code, double equity,
                                             double entry_price, double stop_risk,
                                             double risk_pct, double max_leverage, double cap,
                                             double gbp_usd_approx) {
    double units, risk_dollars, margin;
    position_size_and_margin_debug(instr_code, equity, entry_price, stop_risk, risk_pct,
                                    max_leverage, cap, gbp_usd_approx, units, risk_dollars, margin);
    return py::make_tuple(units, risk_dollars, margin);
}

py::tuple py_simulate_two_account_var_risk(i64arr entry_ns, i64arr exit_ns, i32arr instrument_code,
                                            f64arr entry_price, f64arr stop_risk, f64arr net_r,
                                            u8arr is_buy, f64arr risk_pct, double start_per_side,
                                            double max_leverage, double cap, double gbp_usd_approx,
                                            bool track_detail) {
    const int64_t n = entry_ns.shape(0);
    TwoAccountBacktestResult r = simulate_two_account_var_risk(
        entry_ns.data(), exit_ns.data(), instrument_code.data(), entry_price.data(),
        stop_risk.data(), net_r.data(), is_buy.data(), risk_pct.data(), n, start_per_side,
        max_leverage, cap, gbp_usd_approx, track_detail);
    return py::make_tuple(to_bool(r.accepted), r.equity_buy, r.equity_sell, r.max_dd_pct,
                           to_i64(r.trace_close_ns), to_f64(r.trace_equity_buy),
                           to_f64(r.trace_equity_sell));
}

py::tuple py_run_two_account_mc_var_risk(i64arr entry_ns, i64arr exit_ns, i32arr instrument_code,
                                          f64arr entry_price, f64arr stop_risk, f64arr net_r,
                                          u8arr is_buy, f64arr risk_pct, int64_t n_sims,
                                          int64_t block, int64_t max_trades, uint32_t seed,
                                          double start_per_side, double max_leverage, double cap,
                                          double target1, double target2, double year_ns,
                                          double gbp_usd_approx, double ruin_floor,
                                          int n_threads, int64_t boundary_gap_ns) {
    const int64_t n = entry_ns.shape(0);
    TwoAccountMCResult r = run_two_account_mc_var_risk(
        entry_ns.data(), exit_ns.data(), instrument_code.data(), entry_price.data(),
        stop_risk.data(), net_r.data(), is_buy.data(), risk_pct.data(), n, n_sims, block,
        max_trades, seed, start_per_side, max_leverage, cap, target1, target2, year_ns,
        gbp_usd_approx, ruin_floor, n_threads, boundary_gap_ns);
    return py::make_tuple(to_f64(r.t1_years), to_f64(r.t10_years), to_f64(r.max_dd_pct),
                           to_bool(r.ruined), to_i64(r.processed), to_i64(r.accepted_n));
}

py::tuple py_simulate_two_account_trace(i64arr entry_ns, i64arr exit_ns, i32arr instrument_code,
                                         f64arr entry_price, f64arr stop_risk, f64arr net_r,
                                         u8arr is_buy, double risk_pct, double start_per_side,
                                         double max_leverage, double cap, double gbp_usd_approx) {
    const int64_t n = entry_ns.shape(0);
    TwoAccountTraceResult r = simulate_two_account_trace(
        entry_ns.data(), exit_ns.data(), instrument_code.data(), entry_price.data(),
        stop_risk.data(), net_r.data(), is_buy.data(), n, risk_pct, start_per_side,
        max_leverage, cap, gbp_usd_approx);
    return py::make_tuple(to_bool(r.accepted), to_f64(r.equity_buy_after), to_f64(r.equity_sell_after));
}

py::tuple py_simulate_two_account(i64arr entry_ns, i64arr exit_ns, i32arr instrument_code,
                                   f64arr entry_price, f64arr stop_risk, f64arr net_r,
                                   u8arr is_buy, double risk_pct, double start_per_side,
                                   double max_leverage, double cap, double gbp_usd_approx,
                                   bool track_detail) {
    const int64_t n = entry_ns.shape(0);
    TwoAccountBacktestResult r = simulate_two_account(
        entry_ns.data(), exit_ns.data(), instrument_code.data(), entry_price.data(),
        stop_risk.data(), net_r.data(), is_buy.data(), n, risk_pct, start_per_side,
        max_leverage, cap, gbp_usd_approx, track_detail);
    return py::make_tuple(to_bool(r.accepted), r.equity_buy, r.equity_sell, r.max_dd_pct,
                           to_i64(r.trace_close_ns), to_f64(r.trace_equity_buy),
                           to_f64(r.trace_equity_sell));
}

py::tuple py_run_two_account_mc(i64arr entry_ns, i64arr exit_ns, i32arr instrument_code,
                                 f64arr entry_price, f64arr stop_risk, f64arr net_r,
                                 u8arr is_buy, int64_t n_sims, int64_t block, int64_t max_trades,
                                 uint32_t seed, double risk_pct, double start_per_side,
                                 double max_leverage, double cap, double target1, double target2,
                                 double year_ns, double gbp_usd_approx, double ruin_floor,
                                 int n_threads, int64_t boundary_gap_ns) {
    const int64_t n = entry_ns.shape(0);
    TwoAccountMCResult r = run_two_account_mc(
        entry_ns.data(), exit_ns.data(), instrument_code.data(), entry_price.data(),
        stop_risk.data(), net_r.data(), is_buy.data(), n, n_sims, block, max_trades, seed,
        risk_pct, start_per_side, max_leverage, cap, target1, target2, year_ns, gbp_usd_approx,
        ruin_floor, n_threads, boundary_gap_ns);
    return py::make_tuple(to_f64(r.t1_years), to_f64(r.t10_years), to_f64(r.max_dd_pct),
                           to_bool(r.ruined), to_i64(r.processed), to_i64(r.accepted_n));
}

py::tuple py_run_portfolio_mc(i64arr entry_ns, i64arr exit_ns, i32arr account_code,
                               f64arr risk_pct, f64arr leverage_risk_cap, i32arr unit_count,
                               f64arr unit_r, i64arr unit_offset_ns, f64arr unit_price_ratio,
                               int32_t n_accounts, int64_t n_sims, int64_t block,
                               int64_t max_trades, uint32_t seed, double start_per_account,
                               double cap, double target, double leg_floor, double total_floor,
                               double year_ns, int64_t boundary_gap_ns, int n_threads) {
    const int64_t n = entry_ns.shape(0);
    if (unit_r.ndim() != 2 || unit_offset_ns.ndim() != 2 || unit_price_ratio.ndim() != 2)
        throw std::invalid_argument("unit arrays must be 2D");
    const int32_t max_units = static_cast<int32_t>(unit_r.shape(1));
    if (unit_offset_ns.shape(0) != n || unit_price_ratio.shape(0) != n ||
        unit_offset_ns.shape(1) != max_units || unit_price_ratio.shape(1) != max_units)
        throw std::invalid_argument("unit arrays must share shape [n_trades, max_units]");
    PortfolioMCResult r = run_portfolio_mc(
        entry_ns.data(), exit_ns.data(), account_code.data(), risk_pct.data(),
        leverage_risk_cap.data(), unit_count.data(), unit_r.data(), unit_offset_ns.data(),
        unit_price_ratio.data(), n, max_units, n_accounts, n_sims, block, max_trades, seed,
        start_per_account, cap, target, leg_floor, total_floor, year_ns, boundary_gap_ns,
        n_threads);
    return py::make_tuple(to_f64(r.target_years), to_f64(r.max_dd_pct), to_bool(r.ruined),
                           to_i64(r.processed), to_i64(r.accepted), to_i64(r.skipped));
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

    m.def("position_size_and_margin_debug", &py_position_size_and_margin_debug,
          "Debug-only: exposes the internal position_size_and_margin helper for unit "
          "testing against the Python original -> (units, risk_dollars, margin).");

    m.def("simulate_two_account_var_risk", &py_simulate_two_account_var_risk,
          py::arg("entry_ns"), py::arg("exit_ns"), py::arg("instrument_code"),
          py::arg("entry_price"), py::arg("stop_risk"), py::arg("net_r"), py::arg("is_buy"),
          py::arg("risk_pct"), py::arg("start_per_side") = 100.0, py::arg("max_leverage") = 50.0,
          py::arg("cap") = 500000.0, py::arg("gbp_usd_approx") = 1.36,
          py::arg("track_detail") = true,
          "Same as simulate_two_account, but risk_pct is a PER-TRADE array (one risk fraction "
          "per row) instead of one scalar -- for confidence-weighted position sizing.");

    m.def("run_two_account_mc_var_risk", &py_run_two_account_mc_var_risk,
          py::arg("entry_ns"), py::arg("exit_ns"), py::arg("instrument_code"),
          py::arg("entry_price"), py::arg("stop_risk"), py::arg("net_r"), py::arg("is_buy"),
          py::arg("risk_pct"), py::arg("n_sims"), py::arg("block"), py::arg("max_trades"),
          py::arg("seed"), py::arg("start_per_side") = 100.0, py::arg("max_leverage") = 50.0,
          py::arg("cap") = 500000.0, py::arg("target1") = 1e6, py::arg("target2") = 1e7,
          py::arg("year_ns") = 365.25 * 86400 * 1e9, py::arg("gbp_usd_approx") = 1.36,
          py::arg("ruin_floor") = 50.0, py::arg("n_threads") = 0,
          py::arg("boundary_gap_ns") = 0,
          "Same as run_two_account_mc, but risk_pct is a PER-TRADE array -- a trade's own "
          "risk_pct travels with it through block-bootstrap resampling, so confidence-weighted "
          "sizing survives the Monte Carlo intact.");

    m.def("simulate_two_account_trace", &py_simulate_two_account_trace,
          "Debug-only: like simulate_two_account but returns per-row equity snapshots for "
          "diffing against a Python step-by-step trace.");

    m.def("simulate_two_account", &py_simulate_two_account,
          py::arg("entry_ns"), py::arg("exit_ns"), py::arg("instrument_code"),
          py::arg("entry_price"), py::arg("stop_risk"), py::arg("net_r"), py::arg("is_buy"),
          py::arg("risk_pct"), py::arg("start_per_side") = 100.0, py::arg("max_leverage") = 50.0,
          py::arg("cap") = 500000.0, py::arg("gbp_usd_approx") = 1.36,
          py::arg("track_detail") = true,
          "Deterministic single pass over real trades (sorted by entry_ns, kind='stable' -- "
          "1-min-bar trade sets routinely have tied entry_ns values, and tie order affects "
          "the result) through the TWO separate buy-only/sell-only $-per-side sub-account "
          "model (max 1 open position per side, real per-instrument margin sizing) -> "
          "(accepted, equity_buy, equity_sell, max_dd_pct, trace_close_ns, trace_equity_buy, "
          "trace_equity_sell). instrument_code: 0=EURUSD/GBPUSD, 1=USDJPY, 2=EURGBP. Exact "
          "port of goal30_apex_king_deployment.py::simulate_two_account.");

    m.def("run_two_account_mc", &py_run_two_account_mc,
          py::arg("entry_ns"), py::arg("exit_ns"), py::arg("instrument_code"),
          py::arg("entry_price"), py::arg("stop_risk"), py::arg("net_r"), py::arg("is_buy"),
          py::arg("n_sims"), py::arg("block"), py::arg("max_trades"), py::arg("seed"),
          py::arg("risk_pct"), py::arg("start_per_side") = 100.0, py::arg("max_leverage") = 50.0,
          py::arg("cap") = 500000.0, py::arg("target1") = 1e6, py::arg("target2") = 1e7,
          py::arg("year_ns") = 365.25 * 86400 * 1e9, py::arg("gbp_usd_approx") = 1.36,
          py::arg("ruin_floor") = 50.0, py::arg("n_threads") = 0,
          py::arg("boundary_gap_ns") = 0,
          "Block-bootstrap Monte Carlo over the two-sub-account model, multi-threaded -> "
          "(t1_years, t10_years, max_dd_pct, ruined, processed, accepted_n). Exact port of "
          "goal30_apex_king_deployment.py's account rules, with corrected calendar block gaps "
          "and target checks at exit settlement -- this is the engine "
          "actually used for BEID strategy validation in beid_goal_hunt/.");

    m.def("run_portfolio_mc", &py_run_portfolio_mc,
          py::arg("entry_ns"), py::arg("exit_ns"), py::arg("account_code"),
          py::arg("risk_pct"), py::arg("leverage_risk_cap"), py::arg("unit_count"),
          py::arg("unit_r"), py::arg("unit_offset_ns"), py::arg("unit_price_ratio"),
          py::arg("n_accounts"), py::arg("n_sims"), py::arg("block"),
          py::arg("max_trades"), py::arg("seed"), py::arg("start_per_account"),
          py::arg("cap") = 500000.0, py::arg("target") = 1e6,
          py::arg("leg_floor") = 2.0, py::arg("total_floor") = 10.0,
          py::arg("year_ns") = 365.25 * 86400 * 1e9,
          py::arg("boundary_gap_ns") = 1, py::arg("n_threads") = 0,
          "Calendar-aware, exit-settled, margin-aware multi-account pyramid Monte Carlo.");

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
