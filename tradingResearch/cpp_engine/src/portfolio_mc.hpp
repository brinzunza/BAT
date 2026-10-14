#pragma once
#include <cstdint>
#include <vector>

struct PortfolioMCResult {
    std::vector<double> target_years;
    std::vector<double> max_dd_pct;
    std::vector<uint8_t> ruined;
    std::vector<int64_t> processed;
    std::vector<int64_t> accepted;
    std::vector<int64_t> skipped;
};

// Multi-account event-driven block bootstrap.  Each row is one parent trade.  Unit data is
// row-major [n_trades, max_units]; unit zero enters with the parent, later units are evaluated
// at their actual offsets and consume incremental margin.  P&L is realized only at parent exit.
PortfolioMCResult run_portfolio_mc(
    const int64_t* entry_ns, const int64_t* exit_ns, const int32_t* account_code,
    const double* risk_pct, const double* leverage_risk_cap, const int32_t* unit_count,
    const double* unit_r, const int64_t* unit_offset_ns, const double* unit_price_ratio,
    int64_t n_trades, int32_t max_units, int32_t n_accounts,
    int64_t n_sims, int64_t block, int64_t max_trades, uint32_t seed,
    double start_per_account, double cap, double target, double leg_floor,
    double total_floor, double year_ns, int64_t boundary_gap_ns, int n_threads);

