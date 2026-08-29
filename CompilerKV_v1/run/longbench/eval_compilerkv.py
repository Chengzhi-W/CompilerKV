"""Backward-compatible entry point for evaluating CompilerKV LongBench output."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("eval.py")), run_name="__main__")
