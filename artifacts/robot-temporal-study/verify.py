"""Review/trust this verifier before executing a downloaded copy."""
from pathlib import Path
import runpy
import sys
root=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(root/'verifier'))
sys.argv=[str(root/'verify.py'),'verify','--out',str(root),'--minicpm',str(root.parent/'robot-temporal-minicpm'),'--qwen',str(root.parent/'robot-temporal')]
runpy.run_module('robotbench.temporal_comparison',run_name='__main__')
