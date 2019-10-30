import pandas as pd
from pathlib import Path
from glob import glob
import os

from .constants import CAMERAS
from .project import read_config


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


def detect_images(config_path):
    '''
    Detect images in calibration_images folder, and return a dictionary with their paths
    
    '''
    camera_names = tuple(CAMERAS.keys())
    image_dir = os.path.realpath(read_config(config_path)['calibration_path'])
    
    cam_image_paths = {}
    for img in glob(os.path.join(image_dir, '*.png')):
        img_camera_name = img.split('\\')[-1].split('_')[0]

        if img_camera_name in camera_names:
            cam_image_paths[img_camera_name] = os.path.realpath(img)

    return cam_image_paths


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
    if len(suffix_split):
        if suffix_split[-1] != 'h5':
            msg = 'Invalid file extension in suffix: %s' % suffix
            raise ValueError(msg)
    else:
        suffix = suffix + '.h5'

    cfg = read_config(config_path)
    dlc3d_configs = os.path.realpath(cfg['dlc3d_project_configs'])
    data_path = os.path.realpath(cfg['data_path'])

    # Detect triangulation results in related DeepLabCut 3D projects
    # Analyse the number of occurances of hdf across projects
    rsoi = None
    missing = 0
    status = {}
    coords = {}
    pairs = set(get_pairs())
    for pair, config_path in dlc3d_configs.items():
        current_data_path = os.path.join(
            os.path.join(os.path.dirname(config_path), 'videos'),
            '*%s' % suffix
        )
        hdfs = glob(current_data_path)
        for hdf in hdfs:
            filename = os.path.basename(hdf)
            dframe = pd.read_hdf(os.path.realpath(hdf))['DLC_3D']
            current_rsoi = dframe.columns.levels[0]

            msg = 'The dataframes do not hold data from the same experiment.\nConsensus: %s\n%s: %s' % (
                rsoi, (pair, filename), current_rsoi
            )
            assert rsoi is None or rsoi == current_rsoi
            rsoi = current_rsoi

            coords[(pair, filename)] = {roi: dframe[roi].values for roi in rsoi}
            status[(filename, pair)] = 'X'
    
        pairs_with_hdf = set(kt[1] for kt in status.keys())
        missing = pairs.difference(pairs_with_hdf)
        for pair in missing:
            status[(filename, pair)] = '-'
            missing += 1

    if len(missing) != 0:
        save_path = os.path.join(data_path, 'missing_hdf.xlsx')
        pd.DataFrame.from_dict(status).to_excel(save_path)
        print('There are %d files missing. An overview of the inconsistencies have been saved:\n%s\n' % save_path)
        return False
    else:
        print('Triangulations files detected, and verified')
        if change_basis is True:
            print('Proceeding to changing basis')
            return coords
        else:
            print('The current DeepCage project is ready for changing basis')
