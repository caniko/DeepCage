# from deepcage.auxiliary.detect import detect_triangulation_result
# from deepcage.project import (
#     initialise_projects,
#     calibrate_dlc_cameras
# )
from deepcage.compute import (
    visualize_workflow,
    visualize_triangulation,
    visualize_basis_vectors,
    visualize_basis_vectors_single,
    plot_3d_trajectories,

    # create_stereo_cam_origmap,
    # compute_basis_vectors,
    # dlc3d_create_labeled_video,
    # map_experiment
)
import matplotlib.pyplot as plt
import os


VIDEO_ROOT = os.path.realpath('H:/Can_cage/BonsaiRecordings/MROS_V1')

project_name = 'DeepCage_MROS_V1'
experimenter = 'Can'
root = os.path.realpath('H:/Can_cage/DeepCage_DLC_files')
config_path = os.path.join(root, 'DeepCageKeyTest-Can-2019-11-15/config.yaml')
# config_path = initialise_projects(project_name, experimenter, root, dlc_config, calib_root, vid_format='avi')


# calibrate_dlc_cameras(config_path, cbrow=9, cbcol=6, calibrate=False, alpha=0.9, skip=None)
# calibrate_dlc_cameras(config_path, cbrow=9, cbcol=6, calibrate=True, alpha=0.9, skip=None)

# triangulate_bonvideos(config_path, VIDEO_ROOT, gputouse=0, vformat='avi')
# detect_triangulation_result(config_path)

# create_stereo_cam_origmap(config_path)
# visualize_triangulation(config_path)

normalize = True
# map_experiment(config_path, paralell=False)
# (fig_all, ax_all), (fig_sep, ax_sep) = plot_3d_trajectories(config_path, normalize=normalize, cm_is_real_idx=True, remap=True, cols=2)
visualize_workflow(config_path, normalize=normalize)
# visualize_basis_vectors(config_path, normalize=normalize)
# fig, ax = visualize_basis_vectors_single(config_path, normalize=normalize)
# plt.show()


# dlc3d_create_labeled_video(config_path)
