#!/usr/bin/env python3
"""CLI entry point for the CBPA case study experiment."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cbpa.runner.experiment import main

if __name__ == "__main__":
    main()
