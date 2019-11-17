from deepcage.project.create import create_project_old_cage
from deepcage.auxiliary.logistics import dlc3d_video_migrate

from glob import glob
import os


new_project_name = 'DeepCageKeyTest'

root = os.path.realpath('H:/Can_cage/DeepCage_DLC_files')
old_cage_config = os.path.join(root, 'DeepCage_MROS_V1-Can-2019-10-30/config.yaml')
dlc_working_dir = os.path.join(root, 'DeepLabCut_2D_projects')
new_root = root

config_path = os.path.join(root, 'DeepCageKeyTest-Can-2019-11-15/config.yaml')
video_root = os.path.realpath('H:/Can_cage/test')

# create_project_old_cage(
#     new_project_name, old_cage_config, new_root,
#     video_root=video_root, bonvideos=None, video_extension='avi',
#     dlc_project_config='', dlc_working_dir=dlc_working_dir,
#     new_experimenter=None,
# )

dlc3d_video_migrate(config_path, video_root, many=True, test=False)
