"""Run CamVid/VGG profiling for the dense checkpoint on Colab."""

import sys

sys.path.insert(0, "/content/smallnet/scripts")

from colab_run_camvid_profile_template import run_label

run_label("dense")
