from deepcage.project import (
    initialise_projects,
    calibrate_dlc_cameras,
    triangulate_bonvideos,
)
from deepcage.auxiliary.detect import detect_triangulation_result
from deepcage.compute import (
    visualize_workflow,
    visualize_triangulation,
    visualize_basis_vectors,
    create_stereo_cam_origmap,
    compute_basis_vectors,
    dlc3d_create_labeled_video,
    map_experiment
)
import os


VIDEO_ROOT = os.path.realpath('H:/Can_cage/BonsaiRecordings/MROS_V1')

project_name = 'DeepCage_MROS_V1'
experimenter = 'Can'
root = os.path.realpath('H:/Can_cage/DeepCage_DLC_files')
dlc_config = os.path.join(root, 'DeepLabCut/DeepCage-Can-2019-10-26/config.yaml')
calib_root = os.path.realpath('C:/Users/Can/Projects/CINPLA/Thesis_ReconstructCage/calibration/camera/BonRecordings')

config_path = os.path.join(root, 'DeepCage_MROS_V1-Can-2019-10-30/config.yaml')
# config_path = initialise_projects(project_name, experimenter, root, dlc_config, calib_root, vid_format='avi')


# calibrate_dlc_cameras(config_path, cbrow=9, cbcol=6, calibrate=False, alpha=0.9, skip=None)
# calibrate_dlc_cameras(config_path, cbrow=9, cbcol=6, calibrate=True, alpha=0.9, skip=None)

# triangulate_bonvideos(config_path, VIDEO_ROOT, gputouse=0, vformat='avi')

# visualize_workflow(config_path)
# visualize_triangulation(config_path)
# visualize_basis_vectors(config_path)

# detect_triangulation_result(config_path)

# create_stereo_cam_origmap(config_path)
dlc3d_create_labeled_video(config_path)
# map_experiment(config_path, paralell=False)
