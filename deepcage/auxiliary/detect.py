import pandas as pd
import numpy as np

from shutil import copyfile
from copy import copy

from pathlib import Path
from glob import glob
import os

from deepcage.project.edit import read_config
from deepcage.project.get import get_dlc3d_configs


def detect_bonsai(root):
    '''
    Detect videos in bonsai projects, and return dict where key is info, and value is path.

    Parameters
    ----------
    root : string
        Absolute path of the directory storing BonRecordings related to the project
    '''
    bon_projects = {}
    subs = {}
    for folder in glob(os.path.join(root, 'BonRecordings*')):
        name, animal, trial, date = os.path.basename(folder).split('_')
        date = date.replace('2019', '19')
        assert name == 'BonRecordings'

        bon_projects[(animal, date, trial)] = Path(folder)

    return bon_projects


def detect_videos_in_hierarchy(video_root, deep_dict=False, video_dir_hierarchy=('trial', 'pair'), copy_video_dir=None):
    '''
    Find videos stored in a given hierarchy
    '''

    video_root_subdirs = glob(os.path.join(video_root, '*/'))
    hierarchy = {}
    videos = []
    if video_dir_hierarchy == ('trial', 'pair'):
        for trial in video_root_subdirs:
            trial_name = str(Path(trial).stem)
            pairs_cams_vids = {}
            for pair in glob(os.path.join(trial, '*/')):
                pair_name = str(Path(pair).stem)
                if deep_dict is True:
                    pair_vids = {}
                for vid in glob(os.path.join(pair, '*.avi')):
                    vid_path = Path(vid)
                    video_filename = vid_path.stem
                    pair_id, cam_id, cam, trial = video_filename.split('_')

                    if copy_video_dir is None:
                        vid_path = os.path.realpath(vid)
                    else:
                        new_video_dir = copy_video_dir / trial_name / pair_name
                        if not os.path.exists(str(new_video_dir)):
                            os.makedirs(new_video_dir)
                        vid_path = new_video_dir / vid_path.name
                        copyfile(vid, vid_path)
                    videos.append(vid_path)

                    if deep_dict is True:
                        pair_vids[cam] = vid_path
                    else:
                        pairs_cams_vids[(pair_name, cam)] = vid_path

                if deep_dict is True and pair_vids != {}:
                    pairs_cams_vids[pair_name] = copy(pair_vids)

            if pairs_cams_vids is not {}:
                hierarchy[trial_name] = copy(pairs_cams_vids)
    else:
        raise ValueError('video_root_depth must be ("trial", "pair")')

    return hierarchy, videos


def detect_cage_calibration_images(config_path, img_format='png', name_pos=0):
    '''
    Detect images in calibration_images folder, and return a dictionary with their paths

    Parameters
    ----------
    config_path : string
        Absolute path of the project config.yaml file.
    '''
    from .constants import CAMERAS
    
    camera_names = tuple(CAMERAS.keys())
    image_dir = os.path.realpath(read_config(config_path)['calibration_path'])
    
    cam_image_paths = {}
    for img in glob(os.path.join(image_dir, '*.'+img_format)):
        img_camera_name = Path(img).stem.split('_')[name_pos]

        if img_camera_name in camera_names:
            cam_image_paths[img_camera_name] = os.path.realpath(img)

    return cam_image_paths


def detect_dlc_calibration_images(root, img_format='png'):
    subdirs = glob(os.path.join(root, '*/'))
    result = {}
    for subdir in subdirs:
        subdir = os.path.dirname(subdir)
        idx, fcam1, fcam2 = os.path.basename(subdir).split('_')
        result[(fcam1, fcam2)] = [os.path.realpath(calib_img) for calib_img in glob(os.path.join(subdir, '*.png'))]

    return result


def detect_triangulation_result(config_path, filter_low_likelihood=True, pcutoff=0.1, undistorted=True, suffix='_DLC_3D.h5', change_basis=False, bonvideos=False):
    '''
    This function detects and returns the state of deeplabcut-triangulated coordinate h5 files (can be changed)

    Parameters
    ----------
    config_path : string
        Absolute path of the project config.yaml file.
    suffix : string
        The suffix in the DeepLabCut 3D project triangualtion result storage files
    change_basis : boolean
        Boolean stating wether the function is within a change basis workflow

    Example
    -------

    '''

    suffix_split = suffix.split('.')
    if len(suffix_split) > 1:
        if suffix_split[-1] != 'h5':
            msg = 'Invalid file extension in suffix: %s' % suffix
            raise ValueError(msg)
    else:
        suffix = suffix + '.h5'

    cfg = read_config(config_path)
    dlc3d_cfgs = get_dlc3d_configs(config_path)

    results_path = Path(cfg['results_path'])
    triangulated = results_path / ('undistorted' if undistorted is True else 'distorted') / 'triangulated'
    experiments = glob(str(triangulated / '*/'))
    if experiments == []:
        msg = 'Could not find any triangulated coordinates in %s' % triangulated
        raise ValueError(msg)

    # Detect triangulation results in related DeepLabCut 3D projects
    # Analyse the number of occurances of hdf across projects
    missing = 0
    status, coords, likelihoods, pairs = {}, {}, {}, {}
    for exp_path in experiments:
        exp_dir_name = os.path.basename(exp_path)
        if bonvideos is True:
            animal, trial, date = exp_dir_name.split('_')
            coords[(animal, trial, date)], likelihoods[(animal, trial, date)] = {}, {}
        else:
            coords[exp_dir_name], likelihoods[exp_dir_name] = {}, {}

        regions_of_interest = {}
        for hdf_path in glob(os.path.join(exp_path, '**/*'+suffix)):
            pair_dir = os.path.dirname(hdf_path)
            pair_info = os.path.basename(pair_dir).split('_')
            if len(pair_info) == 2:
                cam1, cam2 = pair_info
            else:
                idx_, cam1, cam2 = pair_info
            pair = (cam1, cam2)

            df = pd.read_hdf(os.path.realpath(hdf_path))['DLC_3D']
            exp_regions_of_interest = df.columns.levels[0]
            regions_of_interest[pair] = exp_regions_of_interest

            if filter_low_likelihood is True:
                try:
                    cam1_prediction_2d = pd.read_hdf(glob(os.path.join(pair_dir, f'*{cam1}*filtered.h5'))[0])
                    cam2_prediction_2d = pd.read_hdf(glob(os.path.join(pair_dir, f'*{cam2}*filtered.h5'))[0])
                except FileNotFoundError:
                    print("No filtered predictions found. Will use the unfiltered predictions.")
                    cam1_prediction_2d = pd.read_hdf(glob(os.path.join(pair_dir, f'*{cam1}*.h5'))[0])
                    cam2_prediction_2d = pd.read_hdf(glob(os.path.join(pair_dir, f'*{cam2}*.h5'))[0])

                coord, roi_likelihood = {}, {}
                for roi in exp_regions_of_interest:
                    cam1_likelihood = cam1_prediction_2d[ cam1_prediction_2d.keys()[0][0] ][roi]['likelihood'].values
                    cam2_likelihood = cam2_prediction_2d[ cam2_prediction_2d.keys()[0][0] ][roi]['likelihood'].values

                    coord_likelihood = np.dstack((cam1_likelihood, cam2_likelihood))[0]
                    coord_idxs = np.all(coord_likelihood < pcutoff, axis=1)
                    df[roi].loc[coord_idxs] = np.nan

                    coord[roi] = df[roi].values
                    roi_likelihood[roi] = coord_likelihood
            else:
                coord = {roi: df[roi].values for roi in exp_regions_of_interest}

            if bonvideos is True:
                coords[(animal, trial, date)][pair] = coord

                if filter_low_likelihood is True:
                    likelihoods[(animal, trial, date)][pair] = roi_likelihood
            else:
                coords[exp_dir_name][pair] = coord
                if filter_low_likelihood is True:
                    likelihoods[exp_dir_name][pair] = roi_likelihood
        # print(1)
        # print(all([exp_regions_of_interest == rsoi for rsoi in regions_of_interest]))

        # if not all(all([exp_regions_of_interest == rsoi for rsoi in regions_of_interest])):
        #     save_path = os.path.join(exp_path, 'rsoi_incom_%s_%s_%s.xlsx' % (animal, trial, date))
        #     pd.DataFrame.from_dict(regions_of_interest).to_excel(save_path)
        #     print('Inconsistencies in exp %s %s %s were found.\nAn overview was saved:\n%s\n' % (
        #         animal, trial, date, save_path
        #         )
        #     )
        #     missing += 1

    if missing == 0:
        print('Triangulations files detected, and verified')
        if change_basis is True:
            print('Proceeding to changing basis')
        else:
            print('The current DeepCage project is ready for changing basis')
        return coords, likelihoods if filter_low_likelihood is True else coords
    else:
        if missing == 1:
            msg = 'Inconsistencies in regions of interest was found in one experiment'
        else:
            msg = 'Inconsistencies in regions of interest were found in %d experiments' % missing
        raise ValueError(msg)


def detect_2d_coords(config_path, suffix='filtered.h5', bonvideos=False):
    '''
    This function detects and returns the state of deeplabcut-triangulated coordinate h5 files (can be changed)

    Parameters
    ----------
    config_path : string
        Absolute path of the project config.yaml file.
    suffix : string
        The suffix in the DeepLabCut 3D project triangualtion result storage files

    Example
    -------

    '''

    cfg = read_config(config_path)
    dlc3d_cfgs = get_dlc3d_configs(config_path)

    results_path = Path(cfg['results_path'])
    triangulated = results_path / 'triangulated'
    experiments = glob(str(triangulated / '*/'))
    if experiments == []:
        msg = 'Could not find any triangulated coordinates in %s' % triangulated
        raise ValueError(msg)

    coords = {}
    for exp_path in experiments:
        exp_dir_name = os.path.basename(exp_path)
        if bonvideos is True:
            animal, trial, date = exp_dir_name.split('_')
            coords[(animal, trial, date)] = {}
        else:
            coords[exp_dir_name] = {}

        regions_of_interest = {}
        for hdf_path in glob(os.path.join(exp_path, '**/*'+suffix)):
            pair_info = os.path.basename(os.path.dirname(hdf_path)).split('_')
            if len(pair_info) == 2:
                cam1, cam2 = pair_info
            else:
                idx_, cam1, cam2 = pair_info
            pair = (cam1, cam2)

            df = pd.read_hdf(os.path.realpath(hdf_path))
            exp_regions_of_interest = df.columns.levels[0]
            regions_of_interest[pair] = exp_regions_of_interest

            coord = {roi: df[roi].values for roi in exp_regions_of_interest}
            if bonvideos is True:
                coords[(animal, trial, date)][pair] = coord
            else:
                coords[exp_dir_name][pair] = coord

    return coords
