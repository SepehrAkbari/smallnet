"""Run CamVid/VGG evaluation for the CP rank-128 checkpoint on Colab."""

import sys

sys.path.insert(0, "/content/smallnet/scripts")

from colab_run_camvid_eval_template import run_label

run_label("cp_rank_128")
