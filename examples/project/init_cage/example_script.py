from deepcage.project import (
    initialise_prepare_projects,
    calibrate_dlc_cameras,
    triangulate_bonvideos
)
import os


VIDEO_ROOT = os.path.realpath('H:/Can_cage/BonsaiRecordings/MROS_V1')

project_name = 'DeepCage_MROS_V1'
experimenter = 'Can'
root = os.path.realpath('H:/Can_cage/DeepLabCut')
dlc_config = os.path.realpath('H:/Can_cage/DeepCage_DLC_files/DCage_1/DeepCage-Can-2019-10-26/config.yaml')
calib_root = os.path.realpath('C:/Users/Can/Projects/CINPLA/Thesis_ReconstructCage/calibration/camera/BonRecordings')

# config_path = initialise_prepare_projects(project_name, experimenter, root, dlc_config, calib_root, vid_format='avi')
config_path = os.path.realpath('H:/Can_cage/DeepLabCut/DeepCage_MROS_V1-Can-2019-10-30/config.yaml')


# calibrate_dlc_cameras(config_path, cbrow=9, cbcol=6, calibrate=False, alpha=0.9, skip=None)
# calibrate_dlc_cameras(config_path, cbrow=9, cbcol=6, calibrate=True, alpha=0.9, skip=None)

triangulate_bonvideos(config_path, VIDEO_ROOT, gputouse=0, vformat='avi')
