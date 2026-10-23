"""Command-line entry for the existing MNIST FL run and multi-seed study."""
from __future__ import annotations

import argparse
import runpy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("main", "multi-seed"), default="main")
    args, remainder = parser.parse_known_args()
    target = ROOT / ("experiments" / "main_fl.py" if args.mode == "main"
                     else "experiments" / "multi_seed.py")
    old_argv = sys.argv
    try:
        sys.argv = [str(target), *remainder]
        runpy.run_path(str(target), run_name="__main__")
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
