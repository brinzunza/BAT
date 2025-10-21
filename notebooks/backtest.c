/*
 * Mean Reversion Backtest in C
 *
 * COMPILATION:
 *   gcc -o backtest backtest.c -lm
 *
 * USAGE:
 *   ./backtest <csv_file> [sma_period] [std_multiplier]
 *
 * ARGUMENTS:
 *   csv_file        - Path to CSV file with OHLCV data (required)
 *   sma_period      - Period for Simple Moving Average (default: 20)
 *   std_multiplier  - Standard deviation multiplier for bands (default: 2.0)
 *
 * EXAMPLES:
 *   # Basic usage with default parameters (20-period SMA, 2.0 std)
 *   ./backtest polygon_data.csv
 *
 *   # Custom parameters: 30-period SMA with 2.5 standard deviations
 *   ./backtest polygon_data.csv 30 2.5
 *
 * CSV FORMAT:
 *   The CSV file must have a header row and the following columns:
 *   timestamp,open,high,low,close,volume
 *
 *   Example:
 *   timestamp,open,high,low,close,volume
 *   2024-01-01T09:30:00Z,150.25,151.00,150.00,150.75,1000000
 *   2024-01-01T09:31:00Z,150.75,151.50,150.50,151.25,1200000
 *
 * STRATEGY:
 *   Mean Reversion using Bollinger Bands:
 *   - Buy when price crosses below lower band (mean - std_multiplier * std)
 *   - Exit long when price returns to mean
 *   - Short when price crosses above upper band (mean + std_multiplier * std)
 *   - Exit short when price returns to mean
 *
 * OUTPUT:
 *   - Trade log showing all entries and exits
 *   - Performance metrics including P&L, win rate, profit factor, etc.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_LINE 1024
#define MAX_BARS 100000

typedef struct {
    char timestamp[64];
    double open;
    double high;
    double low;
    double close;
    double volume;
} Bar;

typedef struct {
    int total_trades;
    int winning_trades;
    int losing_trades;
    double total_pnl;
    double max_drawdown;
    double win_rate;
    double avg_win;
    double avg_loss;
    double profit_factor;
    double sharpe_ratio;
} BacktestResults;

typedef struct {
    int position;  // 1 for long, -1 for short, 0 for no position
    double entry_price;
    double current_pnl;
    double total_pnl;
    double peak_equity;
    double max_drawdown;
    int total_trades;
    int winning_trades;
    int losing_trades;
    double total_wins;
    double total_losses;
} TradingState;

// Calculate Simple Moving Average
double calculate_sma(Bar *bars, int current_idx, int period) {
    if (current_idx < period - 1) {
        return 0.0;
    }

    double sum = 0.0;
    for (int i = 0; i < period; i++) {
        sum += bars[current_idx - i].close;
    }
    return sum / period;
}

// Calculate Standard Deviation
double calculate_std(Bar *bars, int current_idx, int period, double mean) {
    if (current_idx < period - 1) {
        return 0.0;
    }

    double sum_sq_diff = 0.0;
    for (int i = 0; i < period; i++) {
        double diff = bars[current_idx - i].close - mean;
        sum_sq_diff += diff * diff;
    }
    return sqrt(sum_sq_diff / period);
}

// Mean Reversion Strategy
// Buy when price is below lower band (mean - 2*std)
// Sell when price is above upper band (mean + 2*std)
void execute_strategy(Bar *bars, int bar_count, TradingState *state, int sma_period, double std_multiplier) {
    for (int i = sma_period; i < bar_count; i++) {
        double sma = calculate_sma(bars, i, sma_period);
        double std = calculate_std(bars, i, sma_period, sma);

        if (sma == 0.0 || std == 0.0) continue;

        double upper_band = sma + (std_multiplier * std);
        double lower_band = sma - (std_multiplier * std);
        double current_price = bars[i].close;

        // Entry signals
        if (state->position == 0) {
            // Buy signal: price crosses below lower band
            if (current_price < lower_band) {
                state->position = 1;
                state->entry_price = current_price;
                printf("BUY at %s: Price=%.2f, SMA=%.2f, Lower Band=%.2f\n",
                       bars[i].timestamp, current_price, sma, lower_band);
            }
            // Short signal: price crosses above upper band
            else if (current_price > upper_band) {
                state->position = -1;
                state->entry_price = current_price;
                printf("SHORT at %s: Price=%.2f, SMA=%.2f, Upper Band=%.2f\n",
                       bars[i].timestamp, current_price, sma, upper_band);
            }
        }
        // Exit signals
        else if (state->position == 1) {
            // Exit long when price returns to mean or crosses above upper band
            if (current_price >= sma) {
                double pnl = current_price - state->entry_price;
                state->total_pnl += pnl;
                state->total_trades++;

                if (pnl > 0) {
                    state->winning_trades++;
                    state->total_wins += pnl;
                } else {
                    state->losing_trades++;
                    state->total_losses += fabs(pnl);
                }

                printf("SELL at %s: Price=%.2f, Entry=%.2f, PnL=%.2f\n",
                       bars[i].timestamp, current_price, state->entry_price, pnl);

                state->position = 0;
            }
        }
        else if (state->position == -1) {
            // Exit short when price returns to mean or crosses below lower band
            if (current_price <= sma) {
                double pnl = state->entry_price - current_price;
                state->total_pnl += pnl;
                state->total_trades++;

                if (pnl > 0) {
                    state->winning_trades++;
                    state->total_wins += pnl;
                } else {
                    state->losing_trades++;
                    state->total_losses += fabs(pnl);
                }

                printf("COVER at %s: Price=%.2f, Entry=%.2f, PnL=%.2f\n",
                       bars[i].timestamp, current_price, state->entry_price, pnl);

                state->position = 0;
            }
        }

        // Track drawdown
        if (state->total_pnl > state->peak_equity) {
            state->peak_equity = state->total_pnl;
        }
        double current_drawdown = state->peak_equity - state->total_pnl;
        if (current_drawdown > state->max_drawdown) {
            state->max_drawdown = current_drawdown;
        }
    }
}

// Load CSV data from Polygon API format
int load_csv_data(const char *filename, Bar *bars) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error: Could not open file %s\n", filename);
        return -1;
    }

    char line[MAX_LINE];
    int count = 0;
    int header_skipped = 0;

    while (fgets(line, sizeof(line), file) && count < MAX_BARS) {
        // Skip header line
        if (!header_skipped) {
            header_skipped = 1;
            continue;
        }

        // Parse CSV line: timestamp,open,high,low,close,volume
        char *token;
        char *line_copy = strdup(line);
        int field = 0;

        token = strtok(line_copy, ",");
        while (token != NULL && field < 6) {
            switch(field) {
                case 0: // timestamp
                    strncpy(bars[count].timestamp, token, sizeof(bars[count].timestamp) - 1);
                    break;
                case 1: // open
                    bars[count].open = atof(token);
                    break;
                case 2: // high
                    bars[count].high = atof(token);
                    break;
                case 3: // low
                    bars[count].low = atof(token);
                    break;
                case 4: // close
                    bars[count].close = atof(token);
                    break;
                case 5: // volume
                    bars[count].volume = atof(token);
                    break;
            }
            field++;
            token = strtok(NULL, ",");
        }

        free(line_copy);

        if (field >= 5) {  // At least timestamp, open, high, low, close
            count++;
        }
    }

    fclose(file);
    printf("Loaded %d bars from %s\n", count, filename);
    return count;
}

// Calculate and print results
void print_results(TradingState *state) {
    printf("\n");
    printf("========================================\n");
    printf("       BACKTEST RESULTS ANALYSIS        \n");
    printf("========================================\n\n");

    printf("Trading Statistics:\n");
    printf("  Total Trades:      %d\n", state->total_trades);
    printf("  Winning Trades:    %d\n", state->winning_trades);
    printf("  Losing Trades:     %d\n", state->losing_trades);
    printf("\n");

    printf("Performance Metrics:\n");
    printf("  Total P&L:         $%.2f\n", state->total_pnl);
    printf("  Max Drawdown:      $%.2f\n", state->max_drawdown);

    if (state->total_trades > 0) {
        double win_rate = (double)state->winning_trades / state->total_trades * 100;
        printf("  Win Rate:          %.2f%%\n", win_rate);

        if (state->winning_trades > 0) {
            double avg_win = state->total_wins / state->winning_trades;
            printf("  Average Win:       $%.2f\n", avg_win);
        }

        if (state->losing_trades > 0) {
            double avg_loss = state->total_losses / state->losing_trades;
            printf("  Average Loss:      $%.2f\n", avg_loss);
        }

        if (state->winning_trades > 0 && state->losing_trades > 0) {
            double profit_factor = state->total_wins / state->total_losses;
            printf("  Profit Factor:     %.2f\n", profit_factor);
        }
    }

    printf("\n");
    printf("========================================\n");
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <csv_file> [sma_period] [std_multiplier]\n", argv[0]);
        printf("Example: %s data.csv 20 2.0\n", argv[0]);
        return 1;
    }

    const char *filename = argv[1];
    int sma_period = (argc > 2) ? atoi(argv[2]) : 20;
    double std_multiplier = (argc > 3) ? atof(argv[3]) : 2.0;

    printf("========================================\n");
    printf("   MEAN REVERSION BACKTEST\n");
    printf("========================================\n");
    printf("Data File:         %s\n", filename);
    printf("SMA Period:        %d\n", sma_period);
    printf("Std Multiplier:    %.1f\n", std_multiplier);
    printf("========================================\n\n");

    // Allocate memory for bars
    Bar *bars = (Bar *)malloc(MAX_BARS * sizeof(Bar));
    if (!bars) {
        printf("Error: Could not allocate memory for bars\n");
        return 1;
    }

    // Load data
    int bar_count = load_csv_data(filename, bars);
    if (bar_count <= 0) {
        free(bars);
        return 1;
    }

    // Initialize trading state
    TradingState state = {0};
    state.position = 0;
    state.entry_price = 0.0;
    state.current_pnl = 0.0;
    state.total_pnl = 0.0;
    state.peak_equity = 0.0;
    state.max_drawdown = 0.0;
    state.total_trades = 0;
    state.winning_trades = 0;
    state.losing_trades = 0;
    state.total_wins = 0.0;
    state.total_losses = 0.0;

    // Run backtest
    printf("Running backtest...\n\n");
    execute_strategy(bars, bar_count, &state, sma_period, std_multiplier);

    // Print results
    print_results(&state);

    // Cleanup
    free(bars);

    return 0;
}
