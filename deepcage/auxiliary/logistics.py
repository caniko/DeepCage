from shutil import copyfile
from pathlib import Path
from glob import glob
import os

from deepcage.project.get import get_dlc3d_configs
from deepcage.project.edit import read_config


def dlc3d_video_migrate(config_path, root, many=False, test=False):
    '''
    Migrate DeepCage videos to DeepCage related DeepLabCut 3D projects

    Parameters
    ----------
    config_path : string
        String containing the full path of the project config.yaml file.
    root : string
        Full path for the root directory of function operation
    many : bool; default False
        Bool indicating if root stores many projects. False means the root is a project root
    test : bool; default False
        Bool indicating if the respective function call is for testing. If True, the function will not copy video files
    '''

    dlc3d_cfgs = get_dlc3d_configs(config_path)
    pairs = tuple(dlc3d_cfgs.keys())

    if many is False:
        result = copy_videos_from_root(root, dlc3d_cfgs, test)
    else:
        project_dirs = glob(os.path.join(root, '*/'))
        result = {}
        for project_dir in project_dirs:
            project_name = Path(project_dir).stem
            result[project_name] = copy_videos_from_root(project_dir, dlc3d_cfgs, test)

    print('Overview of videos found for each pair:\n%s' % result)
    return result


def copy_videos_from_root(project_root, dlc3d_cfgs, test):
    ''' Helper function for dlc3d_video_migrate '''

    project_name = Path(project_root).stem
    pair_dirs = glob(os.path.join(project_root, '*/'))    # List of subdirs

    result = {}
    for pair_dir in pair_dirs:
        path = Path(pair_dir)
        idx, cam1, cam2 = path.stem.split('_')
        pair = (cam1, cam2)
        if pair not in dlc3d_cfgs:
            # Skip folders that do not have
            continue

        videos = glob(str(path / '*.avi'))
        if test is False:
            dlc3d_path = Path(dlc3d_cfgs[pair]).parent
            dlc3d_vpath = dlc3d_path / 'videos' / project_name
            if not os.path.exists(dlc3d_vpath):
                os.makedirs(dlc3d_vpath)

            for v_file in videos:
                v_name = os.path.basename(v_file)
                copyfile(v_file, dlc3d_vpath / v_name)
        result[pair] = len(videos)

    return result
