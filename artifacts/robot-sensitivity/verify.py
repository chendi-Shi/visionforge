"""Verify only: no fitting, encoder, Ollama, network, or image assets."""
import argparse
from pathlib import Path
import runpy
import sys
root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--main-bundle", type=Path, default=root.parent / "robot-benchmark")
args = parser.parse_args()
sys.path.insert(0, str(root / "verifier"))
sys.dont_write_bytecode = True
sys.argv = [str(root / "verify.py"), "--bundle", str(root), "--main-bundle", str(args.main_bundle), "verify"]
runpy.run_module("robotbench.sensitivity", run_name="__main__")
