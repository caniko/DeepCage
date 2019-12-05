from psutil import cpu_count
import concurrent.futures

from shutil import copyfile
from pathlib import Path
from glob import glob
import os

from deepcage.auxiliary.constants import PAIR_IDXS


def analyse_dlc_videos(config_path, extract_outliers=False, videotype='.avi'):
    from .edit import read_config

    cfg = read_config(config_path)
    dlc_config = cfg['dlc_project_config']
    dlc_project_dir = Path(dlc_config).parent

    # Get all videos
    videos = glob(dlc_project_dir / 'videos' / ('*.'+videotype))

    deeplabcut.analyze_videos(dlc_config, videos, videotype='.avi')
    print('Done analysing')


def triangulate_dlc3d_videos(config_path, gputouse=0, vformat='avi'):
    from .get import get_dlc3d_configs

    dlc3d_cfgs = get_dlc3d_configs(config_path)
    

def triangulate_dc_videos(config_path, video_root, gputouse=0, bonvideos=False, vformat='avi'):
    '''
    Parameters
    ----------
    vid_format : string; default 'avi'
    '''

    from deeplabcut.pose_estimation_3d import triangulate

    from .get import get_dlc3d_configs
    from .edit import read_config

    cfg = read_config(config_path)
    results_path = Path(cfg['results_path'])

    pairs = []
    videos, destfolders = {}, {}
    experiment_dirs = glob(os.path.join(video_root, '*/'))
    for expdir in experiment_dirs:
        if bonvideos is True:
            bon, animal, trial, date = os.path.basename(expdir[:-1]).split('_')
            assert bon == 'BonRecordings'
        else:
            exp_name = str(Path(expdir).stem)

        pair_dirs = glob(os.path.join(expdir, '*/'))
        for pair_dir in pair_dirs:
            pair_id, cam1, cam2 = os.path.basename(pair_dir[:-1]).split('_')

            pair = (cam1, cam2)
            pairs.append(pair)
            if pair not in videos:
                videos[pair] = []
            if pair not in destfolders:
                destfolders[pair] = []

            pair_videos = glob(os.path.join(pair_dir, '*.'+vformat))
            fail_msg = 'Remove additional videos:\n%d > 2:\n%s' % (len(pair_videos), pair_videos)
            assert len(pair_videos) == 2, fail_msg
            videos[pair].append(pair_videos)

            pair_id1, cam_pair_id1, cam1, session_trial1 = os.path.splitext(os.path.basename(pair_videos[0]))[0].split('_')
            pair_id2, cam_pair_id2, cam2, session_trial2 = os.path.splitext(os.path.basename(pair_videos[1]))[0].split('_')
            fail_msg = '%s, %s != %s, %s' % (pair_id1, session_trial1, pair_id2, session_trial2)
            assert (pair_id1, session_trial1) == (pair_id2, session_trial2), fail_msg
            assert int(cam_pair_id1) == 0 and int(cam_pair_id2) == 1

            if bonvideos is True:
                trial_dir = results_path / 'triangulated' / ('%s_%s_%s' % (animal, trial, date)) 
            else:
                trial_dir = results_path / 'triangulated' / exp_name

            destfolder = str(trial_dir / ('%d_%s_%s' % (PAIR_IDXS[pair], *pair)))
            destfolders[pair].append(destfolder)
            if not os.path.exists(os.path.realpath(destfolder)):
                os.makedirs(os.path.realpath(destfolder))

    dlc3d_cfgs = get_dlc3d_configs(config_path)
    for pair, dlc3d_cfg in dlc3d_cfgs.items():
        print('Triangulating videos belonging to %s' % (pair,))
        triangulate(
            dlc3d_cfg,
            videos[pair],
            gputouse=gputouse,
            destfolders=destfolders[pair]
        )

    return True


def calibrate_dlc_cameras(config_path, cbrow=9, cbcol=6, calibrate=False, alpha=0.9, skip=None, paralell=True):
    from deeplabcut import calibrate_cameras
    from .get import get_dlc3d_configs

    dlc3d_cfgs = get_dlc3d_configs(config_path)
    cpu_cores = cpu_count(logical=False)
    
    if paralell is False or cpu_cores < 2:
        for pair, dlc_config in dlc3d_cfgs.items():
            if skip is None or PAIR_IDXS[pair] not in skip:
                calibrate_cameras(dlc_config, cbrow, cbcol, calibrate, alpha)

    else:
        workers = 4 if cpu_cores < 8 else 8
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            if skip is None:
                result = {executor.submit(calibrate_cameras, dlc_config, cbrow, cbcol, calibrate, alpha): pair for pair, dlc_config in dlc3d_cfgs.items()}
            else:
                result = {}
                for pair, dlc_config in dlc3d_cfgs.items():
                    if skip is None or PAIR_IDXS[pair] not in skip:
                        result[executor.submit(calibrate_cameras, dlc_config, cbrow, cbcol, calibrate, alpha)] = pair

            for future in concurrent.futures.as_completed(result):
                pair = result[future]
                try:
                    data = future.result()
                except Exception as exc:
                    print('%s generated an exception: %s' % (pair, exc))
                else:
                    print('%s calibration matrices generated' % (pair,))
