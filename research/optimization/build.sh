#!/bin/bash
# Build script for Java backtesting implementation

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================"
echo "  Backtesting Implementation Build Script"
echo "================================================"
echo ""

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if javac is available
if ! command -v javac &> /dev/null; then
    echo -e "${RED}✗ javac not found${NC}"
    echo ""
    echo "Java JDK is required to compile the backtest."
    echo ""
    echo "Install Java JDK 8 or higher:"
    echo "  macOS:    brew install openjdk"
    echo "  Ubuntu:   sudo apt-get install openjdk-11-jdk"
    echo "  Windows:  Download from https://adoptium.net/"
    echo ""
    exit 1
fi

# Compile Java backtest
echo "Building Java backtest..."
if javac Backtest.java; then
    echo -e "${GREEN}✓ Compilation successful${NC}"
    echo ""
    echo "Generated files:"
    echo "  - Backtest.class"
    echo "  - Backtest\$Bar.class"
    echo "  - Backtest\$TradingState.class"
    echo ""
    echo "================================================"
    echo "  Build completed successfully!"
    echo "================================================"
    echo ""
    echo "Next steps:"
    echo "  1. Fetch data:    python3 fetch_polygon_data.py X:BTCUSD minute 7 btc_data.csv"
    echo "  2. Run backtest:  java Backtest btc_data.csv"
    echo "  3. Optimize:      python3 find_best.py btc_data.csv"
    echo ""
    exit 0
else
    echo -e "${RED}✗ Compilation failed${NC}"
    echo ""
    echo "Please check Backtest.java for syntax errors."
    exit 1
fi
