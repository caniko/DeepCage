from psutil import cpu_count
import concurrent.futures

from shutil import copyfile
from pathlib import Path
from glob import glob
import os

from deepcage.auxiliary.constants import PAIR_IDXS

from .edit import read_config, get_dlc3d_configs


def triangulate_bonvideos(config_path, video_root, gputouse=0, vformat='avi'):
    '''
    Parameters
    ----------
    vid_format : string; default 'avi'
    '''

    from deeplabcut.pose_estimation_3d import triangulate

    cfg = read_config(config_path)
    results_path = Path(cfg['results_path'])

    pairs = []
    videos, destfolders = {}, {}
    experiment_dirs = glob(os.path.join(video_root, '*/'))
    for expdir in experiment_dirs:
        bon, animal, trial, date = os.path.basename(expdir[:-1]).split('_')
        assert bon == 'BonRecordings'

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

            destfolder = results_path / 'triangulated' / ('%s_%s_%s' % (animal, trial, date)) / ('%s_%s' % pair)
            destfolders[pair].append(destfolder)
            if not os.path.exists(destfolder):
                os.makedirs(destfolder)

    dlc3d_cfgs = get_dlc3d_configs(config_path)
    for pair in pairs:
        print('Triangulating videos belonging to %s' % (pair,))
        triangulate(
            dlc3d_cfgs[pair],
            videos[pair],
            gputouse=gputouse,
            destfolders=destfolders[pair]
        )

    return True


def calibrate_dlc_cameras(config_path, cbrow=9, cbcol=6, calibrate=False, alpha=0.9, skip=None, paralell=True):
    from deeplabcut import calibrate_cameras

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
