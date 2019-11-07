import pandas as pd

from pathlib import Path
from glob import glob
import os

from deepcage.project.edit import read_config, get_dlc3d_configs

from .constants import CAMERAS, get_pairs


def detect_bonsai(frames_dir):
    '''
    Parameters
    ----------
    frames_dir : string
        String containing the full path of the directory storing BonRecordings related to the project
    '''
    bon_projects = {}
    subs = {}
    for folder in glob(os.path.join(frames_dir, 'BonRecordings*')):
        name, animal, trial, date = os.path.basename(folder).split('_')
        date = date.replace('2019', '19')
        assert name == 'BonRecordings'

        bon_projects[(animal, date, trial)] = Path(folder)
        subs[(animal, date, trial)] = '%s-%s-%s' % (animal, date, trial)
        
    return bon_projects, subs


def detect_cage_calibration_images(config_path, img_format='png'):
    '''
    Detect images in calibration_images folder, and return a dictionary with their paths
    
    '''
    camera_names = tuple(CAMERAS.keys())
    image_dir = os.path.realpath(read_config(config_path)['calibration_path'])
    
    cam_image_paths = {}
    for img in glob(os.path.join(image_dir, '*.'+img_format)):
        img_camera_name = img.split('\\')[-1].split('_')[0]

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


def detect_triangulation_result(config_path, suffix='_DLC_3D.h5', change_basis=False):
    '''
    This function changes the basis of deeplabcut-triangulated that are 3D.

    Parameters
    ----------
    config_path : string
        String containing the full path of the project config.yaml file.
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
    triangulated = results_path / 'triangulated'
    experiments = glob(str(triangulated / '*/'))

    # Detect triangulation results in related DeepLabCut 3D projects
    # Analyse the number of occurances of hdf across projects
    missing = 0
    status, coords, pairs = {}, {}, {}
    for exp_path in experiments:
        animal, trial, date = os.path.basename(exp_path).split('_')
        coords[(animal, trial, date)] = {}

        regions_of_interest = {}
        for hdf_path in glob(os.path.join(exp_path, '**/*'+suffix)):
            print(hdf_path)
            cam1, cam2 = os.path.basename(os.path.dirname(hdf_path)).split('_')
            pair = (cam1, cam2)

            df = pd.read_hdf(os.path.realpath(hdf_path))['DLC_3D']
            exp_regions_of_interest = df.columns.levels[0]
            regions_of_interest[pair] = exp_regions_of_interest

            coords[(animal, trial, date)][pair] = {roi: df[roi].values for roi in exp_regions_of_interest}
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
        return coords
    else:
        if missing == 1:
            msg = 'Inconsistencies in regions of interest was found in one experiment'
        else:
            msg = 'Inconsistencies in regions of interest were found in %d experiments' % missing
        raise ValueError(msg)
