"""Backward-compatible launcher for the modular MNIST FL entry point."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.argv = [str(ROOT / "experiments" / "main_fl.py"), *sys.argv[1:]]
runpy.run_path(sys.argv[0], run_name="__main__")
