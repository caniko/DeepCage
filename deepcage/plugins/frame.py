from datetime import datetime
import concurrent.futures
from copy import copy
import pandas as pd
import numpy as np
import pickle
import pathlib
import os

from read_exdir import get_network_events

from deepcage.auxiliary import read_config, get_pairs, detect_bonsai, CAMERAS

from .utils import get_closest_idxs


def stereocamera_frames(frames_dir, pair_tol=np.timedelta64(ms=40)):
    '''
    Create frame pairs from temporally close frames within DeepCage camera-pairs
    
    Parameters
    ----------
    frames_dir : string
        String containing the full path of the directory storing BonRecordings related to the project
    pair_tol : np.timedelta64; default np.timedelta64(ms=40)
        Maximum difference between closest frames given in np.timedelta64
    '''

    bon_projects, subs = detect_bonsai(frames_dir)

    root_dir = pathlib.Path(frames_dir) / 'DeepCage'
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    paired_i_timings = {}
    cameras = tuple(CAMERAS.keys())
    pairs = get_pairs()
    for info, bonpath in bon_projects.items():
        animal, date, trial = info
        print('Creating pairs for anime %s trial %s, date: %s' % (animal, trial, date))
        
        cam_timings = {}
        for camera in cameras:
            df = pd.read_csv(bonpath / (camera+'_0.csv'), names=('time', 'millisecond'))
            cam_timings[camera] = []
            for row in df.iterrows():
                macro, micro = row[1].time.split(' ')
                year, day, month = macro.split('/')
                hour, minute, second = micro.split(':')
                r_time = datetime(
                    int(year), int(month), int(day),
                    int(hour), int(minute), int(second),
                    int(row[1].millisecond) * 1000
                )
                cam_timings[camera].append(np.datetime64(r_time))

            cam_timings[camera] = np.array(cam_timings[camera], dtype='datetime64')

        df_timings = {}
        df_lengths = {}
        for pair in pairs:
            cam1, cam2 = pair
            closesti_cam1_timings = get_closest_idxs(cam_timings[cam1], cam_timings[cam2])
            valid_cam2_indeces = np.where(cam_timings[cam1][closesti_cam1_timings] - cam_timings[cam2] <= pair_tol)
            paired_i_timings[(animal, date, trial, pair)] = np.dstack(
                (closesti_cam1_timings[valid_cam2_indeces], valid_cam2_indeces[0])
            )

            df_timings[(pair, cam1)] = closesti_cam1_timings[valid_cam2_indeces]
            df_timings[(pair, cam2)] = valid_cam2_indeces[0]
            df_lengths[valid_cam2_indeces[0].shape[0]] = pair
    
        seq = [cam_timings[pair[1]][df_timings[(pair, pair[1])]] for pair in pairs]
        cam_combined = np.concatenate(seq)
        sorted_cc = np.sort(cam_combined)

        idxs = {}
        for pair in pairs:
            cam1, cam2 = pair
            idxs[pair] = get_closest_idxs(sorted_cc, cam_timings[cam2][df_timings[(pair, cam2)]])
            df_timings[(pair, cam1)] = pd.Series(df_timings[(pair, cam1)], index=idxs[pair])
            df_timings[(pair, cam2)] = pd.Series(df_timings[(pair, cam2)], index=idxs[pair])


        set_name = '%s_%s_%s' % (animal, date, trial)
        filename = 'paired_frames_%s' % set_name

        save_path = root_dir / set_name
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        df = pd.DataFrame(df_timings)
        df.to_hdf(save_path / ('%s.h5' % filename), filename)
        try:
            df.to_excel(save_path / ('%s.xlsx' % filename))
        except ModuleNotFoundError:
            print('openpyxl is not installed, saving readable version as csv instead of xlsx')
            df.to_csv(save_path / ('%s.csv' % filename))

    pickle_path = root_dir / 'stereocamera_frames.pickle'
    with open(pickle_path, 'wb') as outfile:
        pickle.dump(paired_i_timings, outfile)


def create_videos(frames_dir, width=1920, height=1080, start_pair=None):
    '''
    Create video using frame pairs that are temporally close frames within DeepCage camera-pairs
    
    Parameters
    ----------
    frames_dir : string
        String containing the full path of the directory storing BonRecordings related to the project
    '''
    # TODO: Finish
    import cv2

    bon_projects, subs = detect_bonsai(frames_dir)
    frame_root = pathlib.Path(frames_dir) / 'DeepCage'
    paired_i_timings_path = frame_root / 'stereocamera_frames.pickle'
    video_dir = frame_root / 'videos'
    
    with open(paired_i_timings_path, 'rb') as infile:
        paired_i_timings = pickle.load(infile)

    if not os.path.exists(paired_i_timings_path):
        msg = 'Does not exist: %s\ndeepcage.plugins.stereocamera_frames need to be run first' % paired_i_timings_path
        raise ValueError(msg)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for info, timings in paired_i_timings.items():
            animal, date, trial, pair = info

            
