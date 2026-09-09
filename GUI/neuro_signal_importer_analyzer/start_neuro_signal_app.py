#!/usr/bin/env python3
"""Backward-compatible wrapper. Prefer: python3 run_neuro_signal_app.py"""
from run_neuro_signal_app import main

if __name__ == "__main__":
    raise SystemExit(main())
