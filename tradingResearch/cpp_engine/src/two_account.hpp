// C++ implementation of the TWO separate $-per-side sub-account model (buy-only,
// sell-only, never pooled), max 1 open position per side, real per-instrument margin sizing
// via position_size_and_margin, block-bootstrap Monte Carlo over real trade blocks (preserving
// each block's internal entry/exit timing and an explicit empirical boundary gap). This is the engine actually
// used for BEID strategy validation in beid_goal_hunt/ -- distinct from (and not covered by)
// monte_carlo.cpp's single-account run_mc_capped, which ports a different, single-account
// module (goal10_cana_nt_mc_capped.py).
//
// Account/strategy parameters (risk_pct, start_per_side, max_leverage, cap, targets,
// ruin_floor) are all runtime arguments -- nothing about any specific strategy is baked in.
//
// instrument_code: 0 = default margin formula (EURUSD/GBPUSD -- risk_per_unit_usd=stop_risk),
//                  1 = USDJPY (risk_per_unit_usd=stop_risk/entry_price, notional=units),
//                  2 = EURGBP (risk_per_unit_usd=stop_risk*gbp_usd_approx,
//                              notional=units*entry_price*gbp_usd_approx)
// exactly mirroring goal20_speed_king_margin_report.py::position_size_and_margin.
#pragma once
#include <cstdint>
#include <vector>

// Exposed for isolated unit testing against the Python original -- not used internally via
// this declaration (two_account.cpp keeps its own static inline copy for the hot path).
void position_size_and_margin_debug(int32_t instr_code, double equity, double entry_price,
                                     double stop_risk, double risk_pct, double max_leverage,
                                     double cap, double gbp_usd_approx,
                                     double& out_units, double& out_risk_dollars,
                                     double& out_margin);

struct TwoAccountBacktestResult {
    std::vector<uint8_t> accepted;
    double equity_buy;
    double equity_sell;
    double max_dd_pct;  // negative percent, e.g. -12.3 means -12.3%
    // trace: one entry per position CLOSE event (heap-pop settlement), in chronological
    // order -- mirrors Python's close_times/close_buy/close_sell when track_detail=True.
    std::vector<int64_t> trace_close_ns;
    std::vector<double> trace_equity_buy;
    std::vector<double> trace_equity_sell;
};

// Deterministic single pass over real trades in chronological order (entry_ns must be sorted
// ascending) -- the "real history" backtest, exact port of simulate_two_account.
// track_detail: when true, also records the close-event trace (matches Python's default).
TwoAccountBacktestResult simulate_two_account(
    const int64_t* entry_ns, const int64_t* exit_ns, const int32_t* instrument_code,
    const double* entry_price, const double* stop_risk, const double* net_r,
    const uint8_t* is_buy, int64_t n,
    double risk_pct, double start_per_side, double max_leverage, double cap,
    double gbp_usd_approx, bool track_detail);

// Debug-only: like simulate_two_account, but also records equity[both sides] immediately
// after each row's accept/reject decision (before any later heap settlement) -- for isolating
// exactly where a divergence from the Python original first appears.
struct TwoAccountTraceResult {
    std::vector<uint8_t> accepted;
    std::vector<double> equity_buy_after;
    std::vector<double> equity_sell_after;
};

TwoAccountTraceResult simulate_two_account_trace(
    const int64_t* entry_ns, const int64_t* exit_ns, const int32_t* instrument_code,
    const double* entry_price, const double* stop_risk, const double* net_r,
    const uint8_t* is_buy, int64_t n,
    double risk_pct, double start_per_side, double max_leverage, double cap,
    double gbp_usd_approx);

struct TwoAccountMCResult {
    std::vector<double> t1_years;   // -1.0 if target1 never reached within max_trades
    std::vector<double> t10_years;  // -1.0 if target2 never reached within max_trades
    std::vector<double> max_dd_pct;
    std::vector<uint8_t> ruined;
    std::vector<int64_t> processed;
    std::vector<int64_t> accepted_n;
};

// Calendar-aware block-bootstrap Monte Carlo. entry_ns must be
// sorted ascending (same trades array as fed to simulate_two_account). Multi-threaded over
// independent simulation paths.
TwoAccountMCResult run_two_account_mc(
    const int64_t* entry_ns, const int64_t* exit_ns, const int32_t* instrument_code,
    const double* entry_price, const double* stop_risk, const double* net_r,
    const uint8_t* is_buy, int64_t n_trades,
    int64_t n_sims, int64_t block, int64_t max_trades, uint32_t seed,
    double risk_pct, double start_per_side, double max_leverage, double cap,
    double target1, double target2, double year_ns, double gbp_usd_approx, double ruin_floor,
    int n_threads, int64_t boundary_gap_ns = 0);

// Variable-risk variants: same semantics, but risk_pct is a PER-TRADE array (risk_pct[i] used
// when sizing trade i) instead of one scalar for every trade -- for confidence-weighted
// position sizing (e.g. risk more on a stronger signal). In the MC, a trade's risk_pct travels
// with it through block-bootstrap resampling exactly like net_r/stop_risk/entry_price already
// do, so a resampled trade is still sized the way its own signal strength calls for.
TwoAccountBacktestResult simulate_two_account_var_risk(
    const int64_t* entry_ns, const int64_t* exit_ns, const int32_t* instrument_code,
    const double* entry_price, const double* stop_risk, const double* net_r,
    const uint8_t* is_buy, const double* risk_pct, int64_t n,
    double start_per_side, double max_leverage, double cap,
    double gbp_usd_approx, bool track_detail);

TwoAccountMCResult run_two_account_mc_var_risk(
    const int64_t* entry_ns, const int64_t* exit_ns, const int32_t* instrument_code,
    const double* entry_price, const double* stop_risk, const double* net_r,
    const uint8_t* is_buy, const double* risk_pct, int64_t n_trades,
    int64_t n_sims, int64_t block, int64_t max_trades, uint32_t seed,
    double start_per_side, double max_leverage, double cap,
    double target1, double target2, double year_ns, double gbp_usd_approx, double ruin_floor,
    int n_threads, int64_t boundary_gap_ns = 0);
