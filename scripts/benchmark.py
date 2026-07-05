'''
Compatibility wrapper for the manual profiler.

The previous benchmark used thop, which did not account for tltorch
FactorizedConv ranks correctly in this project.
'''

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("profile.py")), run_name="__main__")
