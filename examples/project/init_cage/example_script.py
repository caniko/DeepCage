# from deepcage.auxiliary.detect import detect_triangulation_result
# from deepcage.project import (
#     create_dlc_dc_projects,
#     calibrate_dlc_cameras
# )
from deepcage.compute import (
    visualize_workflow,
    visualize_triangulation,
    visualize_basis_vectors,
    visualize_basis_vectors_single,
    plot_3d_trajectories,

    create_stereo_cam_origmap,
    compute_basis_vectors,
    dlc3d_create_labeled_video,
    map_experiment
)
import matplotlib.pyplot as plt

from pathlib import Path
import os


VIDEO_ROOT = os.path.realpath('H:/Can_cage/BonsaiRecordings/MROS_V1')
ROOT = Path('H:/Can_cage/DeepCageProjects/Projects')

project = ROOT / 'PenTrack'
dc_config = str(project / 'PenTrack-Can-2019-12-04' / 'normal' / 'config.yaml')

project_name = 'PenTrack'
experimenter = 'Can'

undistort = False
normalize = False
# dc_config = create_dlc_dc_projects(project_name, experimenter, root, dlc_config, calib_root, vid_format='avi')

# calibrate_dlc_cameras(dc_config, cbrow=9, cbcol=6, calibrate=False, alpha=0.9, skip=None)
# calibrate_dlc_cameras(dc_config, cbrow=9, cbcol=6, calibrate=True, alpha=0.9, skip=None)

# triangulate_bonvideos(dc_config, VIDEO_ROOT, gputouse=0, vformat='avi')
# detect_triangulation_result(dc_config)

# create_stereo_cam_origmap(dc_config)
# visualize_triangulation(dc_config)

# df = map_experiment(dc_config, paralell=False)
(fig_all, ax_all), (fig_sep, ax_sep) = plot_3d_trajectories(
    dc_config, undistort=undistort, cm_is_real_idx=True, remap=True, cols=2, normalize=normalize, use_saved_origmap=False
)
# ax = visualize_workflow(dc_config, undistort=undistort, normalize=normalize)
# visualize_basis_vectors(dc_config, undistort=undistort, normalize=normalize)
# fig, ax = visualize_basis_vectors_single(dc_config, undistort=undistort, normalize=normalize)
# plt.show()

# dlc3d_create_labeled_video(dc_config)
