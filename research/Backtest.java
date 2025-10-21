import java.io.*;
import java.util.*;
import java.text.DecimalFormat;

/**
 * Mean Reversion Backtest in Java
 *
 * COMPILATION:
 *   javac Backtest.java
 *
 * USAGE:
 *   java Backtest <csv_file> [sma_period] [std_multiplier]
 *
 * ARGUMENTS:
 *   csv_file        - Path to CSV file with OHLCV data (required)
 *   sma_period      - Period for Simple Moving Average (default: 20)
 *   std_multiplier  - Standard deviation multiplier for bands (default: 2.0)
 *
 * EXAMPLES:
 *   # Basic usage with default parameters (20-period SMA, 2.0 std)
 *   java Backtest polygon_data.csv
 *
 *   # Custom parameters: 30-period SMA with 2.5 standard deviations
 *   java Backtest polygon_data.csv 30 2.5
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
public class Backtest {

    private static final DecimalFormat DF2 = new DecimalFormat("#0.00");
    private static final DecimalFormat DF1 = new DecimalFormat("#0.0");

    /**
     * Represents a single OHLCV bar
     */
    static class Bar {
        String timestamp;
        double open;
        double high;
        double low;
        double close;
        double volume;

        Bar(String timestamp, double open, double high, double low, double close, double volume) {
            this.timestamp = timestamp;
            this.open = open;
            this.high = high;
            this.low = low;
            this.close = close;
            this.volume = volume;
        }
    }

    /**
     * Tracks the current trading state
     */
    static class TradingState {
        int position;           // 1 for long, -1 for short, 0 for no position
        double entryPrice;
        double totalPnl;
        double peakEquity;
        double maxDrawdown;
        int totalTrades;
        int winningTrades;
        int losingTrades;
        double totalWins;
        double totalLosses;

        TradingState() {
            this.position = 0;
            this.entryPrice = 0.0;
            this.totalPnl = 0.0;
            this.peakEquity = 0.0;
            this.maxDrawdown = 0.0;
            this.totalTrades = 0;
            this.winningTrades = 0;
            this.losingTrades = 0;
            this.totalWins = 0.0;
            this.totalLosses = 0.0;
        }
    }

    /**
     * Load CSV data from file
     */
    static List<Bar> loadCsvData(String filename) throws IOException {
        List<Bar> bars = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            boolean headerSkipped = false;

            while ((line = br.readLine()) != null) {
                // Skip header line
                if (!headerSkipped) {
                    headerSkipped = true;
                    continue;
                }

                // Parse CSV line: timestamp,open,high,low,close,volume
                String[] fields = line.split(",");

                if (fields.length >= 5) {
                    try {
                        String timestamp = fields[0].trim();
                        double open = Double.parseDouble(fields[1].trim());
                        double high = Double.parseDouble(fields[2].trim());
                        double low = Double.parseDouble(fields[3].trim());
                        double close = Double.parseDouble(fields[4].trim());
                        double volume = fields.length > 5 ? Double.parseDouble(fields[5].trim()) : 0.0;

                        bars.add(new Bar(timestamp, open, high, low, close, volume));
                    } catch (NumberFormatException e) {
                        // Skip malformed lines
                        System.err.println("Warning: Skipping malformed line: " + line);
                    }
                }
            }
        }

        System.out.println("Loaded " + bars.size() + " bars from " + filename);
        return bars;
    }

    /**
     * Calculate Simple Moving Average
     */
    static double calculateSMA(List<Bar> bars, int currentIdx, int period) {
        if (currentIdx < period - 1) {
            return 0.0;
        }

        double sum = 0.0;
        for (int i = 0; i < period; i++) {
            sum += bars.get(currentIdx - i).close;
        }

        return sum / period;
    }

    /**
     * Calculate Standard Deviation
     */
    static double calculateStd(List<Bar> bars, int currentIdx, int period, double mean) {
        if (currentIdx < period - 1) {
            return 0.0;
        }

        double sumSqDiff = 0.0;
        for (int i = 0; i < period; i++) {
            double diff = bars.get(currentIdx - i).close - mean;
            sumSqDiff += diff * diff;
        }

        return Math.sqrt(sumSqDiff / period);
    }

    /**
     * Execute mean reversion strategy
     */
    static void executeStrategy(List<Bar> bars, TradingState state, int smaPeriod, double stdMultiplier) {
        for (int i = smaPeriod; i < bars.size(); i++) {
            double sma = calculateSMA(bars, i, smaPeriod);
            double std = calculateStd(bars, i, smaPeriod, sma);

            if (sma == 0.0 || std == 0.0) continue;

            double upperBand = sma + (stdMultiplier * std);
            double lowerBand = sma - (stdMultiplier * std);
            double currentPrice = bars.get(i).close;
            String timestamp = bars.get(i).timestamp;

            // Entry signals
            if (state.position == 0) {
                // Buy signal: price crosses below lower band
                if (currentPrice < lowerBand) {
                    state.position = 1;
                    state.entryPrice = currentPrice;
                    System.out.println("BUY at " + timestamp + ": Price=" + DF2.format(currentPrice) +
                                     ", SMA=" + DF2.format(sma) + ", Lower Band=" + DF2.format(lowerBand));
                }
                // Short signal: price crosses above upper band
                else if (currentPrice > upperBand) {
                    state.position = -1;
                    state.entryPrice = currentPrice;
                    System.out.println("SHORT at " + timestamp + ": Price=" + DF2.format(currentPrice) +
                                     ", SMA=" + DF2.format(sma) + ", Upper Band=" + DF2.format(upperBand));
                }
            }
            // Exit signals
            else if (state.position == 1) {
                // Exit long when price returns to mean
                if (currentPrice >= sma) {
                    double pnl = currentPrice - state.entryPrice;
                    state.totalPnl += pnl;
                    state.totalTrades++;

                    if (pnl > 0) {
                        state.winningTrades++;
                        state.totalWins += pnl;
                    } else {
                        state.losingTrades++;
                        state.totalLosses += Math.abs(pnl);
                    }

                    System.out.println("SELL at " + timestamp + ": Price=" + DF2.format(currentPrice) +
                                     ", Entry=" + DF2.format(state.entryPrice) + ", PnL=" + DF2.format(pnl));

                    state.position = 0;
                }
            }
            else if (state.position == -1) {
                // Exit short when price returns to mean
                if (currentPrice <= sma) {
                    double pnl = state.entryPrice - currentPrice;
                    state.totalPnl += pnl;
                    state.totalTrades++;

                    if (pnl > 0) {
                        state.winningTrades++;
                        state.totalWins += pnl;
                    } else {
                        state.losingTrades++;
                        state.totalLosses += Math.abs(pnl);
                    }

                    System.out.println("COVER at " + timestamp + ": Price=" + DF2.format(currentPrice) +
                                     ", Entry=" + DF2.format(state.entryPrice) + ", PnL=" + DF2.format(pnl));

                    state.position = 0;
                }
            }

            // Track drawdown
            if (state.totalPnl > state.peakEquity) {
                state.peakEquity = state.totalPnl;
            }
            double currentDrawdown = state.peakEquity - state.totalPnl;
            if (currentDrawdown > state.maxDrawdown) {
                state.maxDrawdown = currentDrawdown;
            }
        }
    }

    /**
     * Print backtest results
     */
    static void printResults(TradingState state) {
        System.out.println();
        System.out.println("========================================");
        System.out.println("       BACKTEST RESULTS ANALYSIS        ");
        System.out.println("========================================");
        System.out.println();

        System.out.println("Trading Statistics:");
        System.out.println("  Total Trades:      " + state.totalTrades);
        System.out.println("  Winning Trades:    " + state.winningTrades);
        System.out.println("  Losing Trades:     " + state.losingTrades);
        System.out.println();

        System.out.println("Performance Metrics:");
        System.out.println("  Total P&L:         $" + DF2.format(state.totalPnl));
        System.out.println("  Max Drawdown:      $" + DF2.format(state.maxDrawdown));

        if (state.totalTrades > 0) {
            double winRate = (double) state.winningTrades / state.totalTrades * 100;
            System.out.println("  Win Rate:          " + DF2.format(winRate) + "%");

            if (state.winningTrades > 0) {
                double avgWin = state.totalWins / state.winningTrades;
                System.out.println("  Average Win:       $" + DF2.format(avgWin));
            } else {
                System.out.println("  Average Win:       $0.00");
            }

            if (state.losingTrades > 0) {
                double avgLoss = state.totalLosses / state.losingTrades;
                System.out.println("  Average Loss:      $" + DF2.format(avgLoss));
            } else {
                System.out.println("  Average Loss:      $0.00");
            }

            if (state.winningTrades > 0 && state.losingTrades > 0) {
                double profitFactor = state.totalWins / state.totalLosses;
                System.out.println("  Profit Factor:     " + DF2.format(profitFactor));
            } else if (state.winningTrades > 0 && state.losingTrades == 0) {
                // Perfect strategy - all wins, no losses
                System.out.println("  Profit Factor:     ∞ (no losses)");
            } else {
                System.out.println("  Profit Factor:     0.00");
            }
        }

        System.out.println();
        System.out.println("========================================");
    }

    /**
     * Main entry point
     */
    public static void main(String[] args) {
        if (args.length < 1) {
            System.out.println("Usage: java Backtest <csv_file> [sma_period] [std_multiplier]");
            System.out.println("Example: java Backtest data.csv 20 2.0");
            System.exit(1);
        }

        String filename = args[0];
        int smaPeriod = args.length > 1 ? Integer.parseInt(args[1]) : 20;
        double stdMultiplier = args.length > 2 ? Double.parseDouble(args[2]) : 2.0;

        System.out.println("========================================");
        System.out.println("   MEAN REVERSION BACKTEST");
        System.out.println("========================================");
        System.out.println("Data File:         " + filename);
        System.out.println("SMA Period:        " + smaPeriod);
        System.out.println("Std Multiplier:    " + DF1.format(stdMultiplier));
        System.out.println("========================================");
        System.out.println();

        try {
            // Load data
            List<Bar> bars = loadCsvData(filename);

            if (bars.isEmpty()) {
                System.err.println("Error: No data loaded from file");
                System.exit(1);
            }

            // Initialize trading state
            TradingState state = new TradingState();

            // Run backtest
            System.out.println("Running backtest...");
            System.out.println();
            executeStrategy(bars, state, smaPeriod, stdMultiplier);

            // Print results
            printResults(state);

        } catch (FileNotFoundException e) {
            System.err.println("Error: File not found: " + filename);
            System.exit(1);
        } catch (IOException e) {
            System.err.println("Error reading file: " + e.getMessage());
            System.exit(1);
        } catch (NumberFormatException e) {
            System.err.println("Error: Invalid parameter format");
            System.err.println("Usage: java Backtest <csv_file> [sma_period] [std_multiplier]");
            System.exit(1);
        }
    }
}
