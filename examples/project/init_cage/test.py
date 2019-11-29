from deepcage.compute import (
    plot_3d_trajectories
)
from deepcage.auxiliary.detect import detect_triangulation_result
import os


root = os.path.realpath('D:/Can_cage/DeepCage_DLC_files')
config_path = os.path.join(root, 'DeepCageKeyTest-Can-2019-11-15/config.yaml')

#plot_3d_trajectories(config_path, cm_is_real_idx=True, remap=False, cols=2)

dfs = detect_triangulation_result(config_path, change_basis=False)
