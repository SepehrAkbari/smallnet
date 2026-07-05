"""Run CamVid/VGG profiling for the CP rank-256 checkpoint on Colab."""

import sys

sys.path.insert(0, "/content/smallnet/scripts")

from colab_run_camvid_profile_template import run_label

run_label("cp_rank_256")
