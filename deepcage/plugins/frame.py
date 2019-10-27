from datetime import datetime
import concurrent.futures
from copy import copy
import pandas as pd
import numpy as np
import pickle

from glob import glob
import pathlib
import os
import re

from read_exdir import get_network_events

from deepcage.auxiliary import read_config, get_pairs, detect_bonsai, CAMERAS

from .utils import get_closest_idxs


def stereocamera_frames(frames_dir, pair_tol=np.timedelta64(100, 'ms')):
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

    i_second = np.timedelta64(1, 's')
    paired_timing_idxs = {}
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

        rem_timings = {}
        df_lengths = {}
        fps = []
        for pair in pairs:
            cam1, cam2 = pair
            closesti_cam1_timings = get_closest_idxs(cam_timings[cam1], cam_timings[cam2])
            valid_cam2_indeces = np.where(np.abs(cam_timings[cam1][closesti_cam1_timings] - cam_timings[cam2]) <= pair_tol)

            rem_timings[(pair, cam1)] = closesti_cam1_timings[valid_cam2_indeces]
            rem_timings[(pair, cam2)] = valid_cam2_indeces[0]
            paired_timing_idxs[(animal, date, trial, pair)] = np.dstack(
                (rem_timings[(pair, cam1)], rem_timings[(pair, cam2)])
            )
            
            fps.append(
                np.ceil( i_second / ( (cam_timings[cam1][rem_timings[(pair, cam1)]][-1]
                                       - cam_timings[cam1][rem_timings[(pair, cam1)]][0])
                                       / rem_timings[(pair, cam1)].shape[0] ) * 10) / 10 )
            fps.append(copy(fps[-1]))
            df_lengths[valid_cam2_indeces[0].shape[0]] = pair
    
        seq = [cam_timings[pair[1]][rem_timings[(pair, pair[1])]] for pair in pairs]
        cam_combined = np.concatenate(seq)
        sorted_cc = np.sort(cam_combined)

        idxs = {}
        for pair in pairs:
            cam1, cam2 = pair
            idxs[pair] = get_closest_idxs(sorted_cc, cam_timings[cam2][rem_timings[(pair, cam2)]])
            rem_timings[(pair, cam1)] = pd.Series(rem_timings[(pair, cam1)], index=idxs[pair])
            rem_timings[(pair, cam2)] = pd.Series(rem_timings[(pair, cam2)], index=idxs[pair])


        set_name = '%s_%s_%s' % (animal, date, trial)
        filename = 'paired_frames_%s' % set_name

        save_path = root_dir / set_name
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        df = pd.DataFrame(rem_timings)
        df.to_hdf(save_path / ('%s.h5' % filename), filename)
        try:
            df.to_excel(save_path / ('%s.xlsx' % filename))
        except ModuleNotFoundError:
            print('openpyxl is not installed, saving readable version as csv instead of xlsx')
            df.to_csv(save_path / ('%s.csv' % filename))

    print('FPS: %s\nMean FPS: %s' % (fps, np.mean(fps)))
    pickle_path = root_dir / 'stereocamera_frames.pickle'
    with open(pickle_path, 'wb') as outfile:
        pickle.dump((paired_timing_idxs, fps), outfile)


def create_videos(frames_dir, width=1920, height=1080, start_pair=None, notebook=False):
    '''
    Create video using frame pairs that are temporally close frames within DeepCage camera-pairs
    
    Parameters
    ----------
    frames_dir : string
        String containing the full path of the directory storing BonRecordings related to the project
    '''
    # TODO: Finish
    from .utils import encode_video
    import cv2

    bon_projects, subs = detect_bonsai(frames_dir)
    frame_root = pathlib.Path(frames_dir) / 'DeepCage'
    pickle_path = frame_root / 'stereocamera_frames.pickle'

    if not os.path.exists(pickle_path):
        msg = 'Does not exist: %s\ndeepcage.plugins.stereocamera_frames need to be run first' % pickle_path
        raise ValueError(msg)

    with open(pickle_path, 'rb') as infile:
        paired_timing_idxs, fps = pickle.load(infile)

    imgs = []
    save_paths = []
    for info, timings in paired_timing_idxs.items():
        animal, date, trial, pair = info
        set_name = '%s_%s_%s' % (animal, date, trial)
        video_dir = frame_root / set_name / 'videos' / ('%s_%s' % pair)
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)

        for i in range(len(pair)):
            cam = pair[i]
            img_path = glob(os.path.realpath(bon_projects[(animal, date, trial)] / ('*_%s' % cam)))[0]

            all_imgs = glob(os.path.join(img_path, '*.png'))

            this_imgs = []
            for fi in np.nditer(timings.T[i]):
                this_imgs.append(os.path.abspath(all_imgs[int(fi)]))
            imgs.append(this_imgs)

            # imgs.append( [ os.path.abspath(all_imgs[int(fi)]) for fi in np.nditer(timings.T[i]) ] )
            save_paths.append( video_dir / ('%s_%d-%s.avi' % (set_name, i, cam)) )

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        video_encoders = executor.map(encode_video, save_paths, imgs, fps)
        for future in video_encoders:
            print(future)

    # video_encoders = map(encode_video, save_paths, imgs, fps)
    # for result in video_encoders:
    #     print(result)
