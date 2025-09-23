#!/usr/bin/env python3
"""
BAT - Backtesting & Automated Trading System
Main application entry point with enhanced CLI supporting stocks and crypto
"""

import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ui.cli_interface import main

if __name__ == "__main__":
    main()