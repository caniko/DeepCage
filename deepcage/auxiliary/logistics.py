from shutil import copyfile
from pathlib import Path
from glob import glob
import os

from deepcage.project.edit import read_config, get_dlc3d_configs

def dlc3d_video_migrate(config_path, video_root, test=False):
    '''
    Migrate DeepCage videos to DeepCage related DeepLabCut 3D projects
    
    '''

    dlc3d_cfgs = get_dlc3d_configs(config_path)
    pairs = tuple(dlc3d_cfgs.keys())

    pair_dirs = glob(os.path.join(video_root, '*/'))    # List of subdirs
    paired_videos = {}
    result = {}
    for p_dir in pair_dirs:
        path = Path(p_dir)
        idx, cam1, cam2 = path.stem.split('_')
        pair = (cam1, cam2)

        dlc3d_path = Path(dlc3d_cfgs[pair]).parent
        dlc3d_vpath = dlc3d_path / 'videos'
        if not os.path.exists(dlc3d_vpath):
            os.mkdir(dlc3d_vpath)

        videos = glob(str(path / '*.avi'))
        if test is False:
            for v_file in videos:
                v_name = os.path.basename(v_file)
                copyfile(v_file, dlc3d_vpath / v_name)
        result[pair] = len(videos)

    if test is True:
        for pair, v_num in result.items():
            if len(v_num) != 2:
                msg = 'There should be a pair of videos, 2 videos, but it is not.\n%s' % (result,)
                raise ValueError(msg)

    print('Number of videos found for pairs', result)
    return result
