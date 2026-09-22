// Strict swing-pivot detection. Exact port of PRODLIVETRADING/strategy/pivots.py
// (identical algorithm to tradingResearch/dtfxStrategy/research_scripts/dtfx_eurusd_backtest_strict.py::find_pivots_strict).
//
// A candidate high/low only ever extends on a bar whose CLOSE (never a wick) beats the
// candidate's own high/low, and only gets validated as an actual swing point after
// min_retracement CONSECUTIVE opposite-color candles follow without extending it.
#pragma once
#include <cstdint>
#include <vector>

struct Pivot {
    int64_t time_ns;  // times_ns[candidate_i] at the moment the pivot was validated
    double price;
    char type;         // 'H' or 'L'
};

// open/high/low/close/times_ns must all have length n, times_ns strictly ascending.
std::vector<Pivot> find_pivots_strict(const double* open, const double* high, const double* low,
                                       const double* close, const int64_t* times_ns,
                                       int64_t n, int min_retracement);
