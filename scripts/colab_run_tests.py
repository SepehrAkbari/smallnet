"""Run smallnet test suite on the Colab VM."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT = Path("/content/smallnet")
os.chdir(PROJECT)
print(f"cwd={Path.cwd()}")
subprocess.check_call([sys.executable, "-m", "unittest", "discover", "-s", "tests"])
