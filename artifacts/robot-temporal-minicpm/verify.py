"""Read-only stdlib verifier. Review/trust this tool before executing it."""
from pathlib import Path
import runpy
import sys
root=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(root/'verifier'))
sys.argv=[str(root/'verify.py'),'verify','--bundle',str(root)]
runpy.run_module('robotbench.temporal_report',run_name='__main__')
