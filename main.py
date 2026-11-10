#!/usr/bin/env python3
"""
BAT - Bruninvestor Algorithmic Trading
Main application entry point.

    python3 main.py             full-screen terminal UI (default)
    python3 main.py --classic   the original prompt-based CLI

The terminal UI needs an interactive terminal and the ``textual`` package; when
either is missing this falls back to the classic CLI automatically.
"""

import os
import sys

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def run_classic() -> None:
    from src.ui.cli_interface import main
    main()


def run_tui() -> None:
    from src.tui.app import run
    run()


def main() -> None:
    args = set(sys.argv[1:])
    if "--classic" in args or "-c" in args:
        return run_classic()
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        print("No interactive terminal detected - starting the classic CLI.")
        return run_classic()
    try:
        import textual  # noqa: F401
    except ImportError:
        print("The terminal UI needs 'textual' (pip install textual) - starting the classic CLI instead.")
        return run_classic()
    run_tui()


if __name__ == "__main__":
    main()
