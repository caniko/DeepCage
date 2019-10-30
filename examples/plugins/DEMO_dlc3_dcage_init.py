import os

from deepcage.plugins import dlc3d_deepcage_calibration


project_name = 'DeepCage_MROS_V1'
experimenter = 'Can'
root = os.path.realpath('H:/Can_cage/DeepLabCut')
dlc_config = os.path.realpath('H:/Can_cage/DeepCage_DLC_files/DCage_1/DeepCage-Can-2019-10-26/config.yaml')
calib_root = os.path.realpath('C:/Users/Can/Projects/CINPLA/Thesis_ReconstructCage/calibration/camera/BonRecordings')


dlc3d_deepcage_calibration(project_name, experimenter, root, dlc_config, calib_root, vid_format='avi')
