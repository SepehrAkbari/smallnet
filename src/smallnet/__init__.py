'''
Reusable research framework for the SPL-ready smallnet experiments.
'''

from src.smallnet.config import load_config, require_keys
from src.smallnet.diagnostics import rank_energy_diagnostic
from src.smallnet.modules import get_module, set_module
from src.smallnet.results import save_manifest

__all__ = [
    "get_module",
    "load_config",
    "rank_energy_diagnostic",
    "require_keys",
    "save_manifest",
    "set_module",
]
