from deepcage.compute.test import (
    visualize_workflow,
    visualize_triangulation,
    visualize_basis_vectors,
    dlc3d_create_labeled_video
)

import os

root = os.path.realpath('H:/Can_cage/DeepCage_DLC_files')
config_path = os.path.join(root, 'DeepCage_MROS_V1-Can-2019-10-30/config.yaml')

# visualize_workflow(config_path)
# visualize_triangulation(config_path)
visualize_basis_vectors(config_path)

# dlc3d_create_labeled_video(config_path)
