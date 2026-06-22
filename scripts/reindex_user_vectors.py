#!/usr/bin/env python3
"""Backward-compatible entry — forwards to reindex_all_vectors.py."""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "scripts" / "reindex_all_vectors.py"

spec = importlib.util.spec_from_file_location("reindex_all_vectors", TARGET)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load {TARGET}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

if __name__ == "__main__":
    raise SystemExit(module.main())
