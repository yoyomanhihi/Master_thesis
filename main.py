import unet_running
import sys

name = sys.argv[1]

# example call of federated equal-chances.
# Of course, a dataset organized as described in
# the report and in unet_utils is needed to work
unet_running.build_and_save_fedeq('datasets/dataset_fedAvg_example', None, 3, name)