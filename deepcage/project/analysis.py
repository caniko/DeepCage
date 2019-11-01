import concurrent.futures

from pathlib import Path
from glob import glob
import os

from deepcage.auxiliary.detect import detect_cpu_number
from deepcage.auxiliary.constants import PAIR_IDXS
from .edit import read_config, get_dlc3d_configs


def add_new_videos(config_path, video_root, vformat='avi'):
    '''
    Parameters
    ----------
    vid_format : string; default 'avi'
    '''
    dlc3d_cfgs = get_dlc3d_configs(config_path)
    subdirs = glob(os.path.join(video_root, '*/'))
    for subdir in subdirs:
        bon, animal, trial, date = os.path.basename(subdir[:-1]).split('_')
        assert bon == 'BonRecordings'
        print('Moving videos of experiment with info:\nanimal %s; trial: %s; date %s' % (
            animal, trial, date
        ))

        video_dirs = glob(os.path.join(subdir, '*/'))
        for video_dir in video_dirs:
            pair_dirs = glob(os.path.join(subdir, '*/'))
            for pair_dir in pair_dirs:
                pair_id, cam1, cam2 = os.path.basename(pair_dir[:-1]).split('_')
                project_root = os.path.dirname(dlc3d_cfgs[(cam1, cam2)])

                video_dir = os.path.join(project_root, 'videos')
                if not os.path.exists(video_dir):
                    os.mkdir(video_dir)
                for video in glob(os.path.join(pair_dir, '*.'+vformat)):
                    cam = os.path.basename(video).split('_')[2]
                    new_name = '%s_%s_%s_%s.%s' % (animal, trial, cam, date, vformat)
                    copyfile(os.path.realpath(video), os.path.join(video_dir, new_name))

    return True


def triangulate_videos(config_path, gputouse=0, vformat='avi'):
    from deeplabcut.pose_estimation_3d import triangulate

    cfg = read_config(config_path)
    data_path = Path(cfg['data_path'])
    dlc3d_cfgs = get_dlc3d_configs(config_path)

    for pair, dlc_config in dlc3d_cfgs.items():
        video_path = os.path.join(os.path.dirname(dlc_config), 'videos')
        if not os.path.exists(video_path):
            print('No videos found in %s' % os.path.dirname(dlc_config))
            continue

        videos = glob(os.path.join(video_path, '*.'+vformat))
        for v in videos:
            animal, trial, cam, date = os.path.splitext(os.path.basename(v))[0].split('_')
            dest_folder = data_path / ('%s_%s_%s_%s' % (animal, trial, cam, date)) / str(pair)
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)

            triangulate(config_path, video_path, gputouse=gputouse, destfolder=dest_folder)


def calibrate_dlc_cameras(config_path, cbrow=9, cbcol=6, calibrate=False, alpha=0.9, skip=None, paralell=True):
    from deeplabcut import calibrate_cameras

    dlc3d_cfgs = get_dlc3d_configs(config_path)
    cpu_cores = detect_cpu_number(logical=False)
    
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
