from datetime import datetime
import concurrent.futures
from copy import copy
import pandas as pd
import numpy as np
import pickle

from pathlib import Path
from glob import glob
import os
import re

from read_exdir import get_network_events

from deepcage.auxiliary.constants import CAMERAS, PAIR_IDXS, get_pairs
from deepcage.project import (
    create_new_project_3d, create_dc_project,
    read_config,
    calibrate_dlc_cameras,
    detect_bonsai
)


def initialise_projects(project_name, experimenter, root, dlc_config, calib_root):
    '''
    Initialise a DeepCage project along with the DeepLabCut 3D for each stereo camera. Following the creation of these DLC 3D projects,
    detect calibration images located in a standard Bonsai project (can be found in examples in repo); move the images to their respective
    DLC 3D projects. Run deeplabcut.calibrate_cameras(calibrate=False) for each DLC 3D project,
    and prepare for quality assurance of the checkerboard detections.
    
    Parameters
    ----------
    project_name : str
        String containing the project_name of the project.
    experimenter : str
        String containing the project_name of the experimenter/scorer.
    root : string
        String containing the full path of to the directory where the new project files should be located
    dlc_config : string
        String containing the full path of to the dlc config.yaml file that will be used for the dlc 3D projects
    calib_root : string
        String containing the full path of to the root directory storing the calibration files for each dlc 3D project
    '''
    from deepcage.auxiliary.detect import detect_dlc_calibration_images
    from deeplabcut.create_project import create_new_project_3d
    from deeplabcut.utils.auxiliaryfunctions import write_config_3d

    dlc3d_project_configs = {}

    with concurrent.futures.ProcessPoolExecutor(max_worker=4) as executor:
        for pair, calib_paths in detect_dlc_calibration_images(calib_root).items():
            cam1, cam2 = pair

            name = '%d_%s_%s' % (PAIR_IDXS[pair], cam1, cam2)
            dlc3d_project_configs[pair] = create_new_project_3d(name, experimenter, num_cameras=2, working_directory=root)
            project_path = Path(os.path.dirname(dlc3d_project_configs[pair]))

            calibration_images_path = project_path / 'calibration_images'
            if not os.path.exists(calibration_images_path):
                os.makedirs(calibration_images_path)

            executor.submit(png_to_jpg, calibration_images_path, img_paths=calib_paths)
            
            cfg = read_config(dlc3d_project_configs[pair])
            cfg['config_file_camera-1'] = dlc_config
            cfg['config_file_camera-2'] = dlc_config
            cfg['camera_names'] = list(pair)
            cfg['trainingsetindex_'+cam1] = cfg.pop('trainingsetindex_camera-1')
            cfg['trainingsetindex_'+cam2] = cfg.pop('trainingsetindex_camera-2')
            write_config_3d(dlc3d_project_configs[pair], cfg)

    config_path = create_dc_project(project_name, experimenter, dlc_config, dlc3d_project_configs, working_directory=root)

    calibrate_dlc_cameras(config_path, cbrow=9, cbcol=6, calibrate=False, alpha=0.9)
    return config_path


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

    root_dir = Path(frames_dir) / 'DeepCage'
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
    import cv2

    bon_projects, subs = detect_bonsai(frames_dir)
    frame_root = Path(frames_dir) / 'DeepCage'
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


def get_closest_idxs(array, values):
    # Courtesy of @anthonybell https://stackoverflow.com/a/46184652
    
    # make sure array is a numpy array
    array = np.asarray(array)

    # get insert positions
    idxs = np.searchsorted(array, values, side="left")

    # find indexes where previous index is closer
    prev_idx_is_less = ( (idxs == len(array))|(np.abs(values - array[np.maximum(idxs-1, 0)]) < np.abs(values - array[np.minimum(idxs, len(array)-1)])) )
    idxs[prev_idx_is_less] -= 1

    return idxs


def encode_video(save_path, img_paths, fps, width=1920, height=1080):
    from tqdm import trange
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(filename=str(save_path), fourcc=fourcc, apiPreference=0, fps=float(fps), frameSize=(width, height))

    for i in trange(len(img_paths), desc='Encoding: %s' % os.path.basename(save_path)):
        img = cv2.imread(img_paths[i])
        video.write(img)

    cv2.destroyAllWindows()
    video.release()


def png_to_jpg(save_dir, img_paths=None, img_root=None, codec='cv'):
    assert os.path.exists(save_dir), 'Does not exist:\n%s' % save_dir

    if img_paths is None:
        img_paths = glob(os.path.join(img_root, '**/*.png'))
    elif img_root is None:
        msg = "Either img_paths or img_root has to be defined"
        ValueError(msg)

    for img in img_paths:
        img_path = os.path.realpath(img)

        if '\\' in img_path:
            separator = '\\'
        elif '/' in img_path:
            separator = '/'

        save_path = os.path.join(save_dir, os.path.basename(img).replace('png', 'jpg'))
        if codec == 'pil':
            from PIL import Image

            im = Image.open(img_path)
            rgb_im = im.convert('RGB')
            rgb_im.save(save_path)

        elif codec == 'cv':
            import cv2

            jpg = cv2.imread(img_path)
            cv2.imwrite(save_path, jpg)
