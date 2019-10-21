import os, pickle, yaml
from glob import glob
from pathlib import Path
import ruamel.yaml

import concurrent.futures
from warnings import warn
import pandas as pd

from .basis import change_basis
from .constants import CAMERAS


def detect_images(config_path):
    '''
    Detect images in calibration_images folder, and return a dictionary with their paths
    
    '''
    camera_names = tuple(CAMERAS.keys())
    image_dir = os.path.realpath(read_config(config_path)['calibration_path'])
    
    cam_image_paths = {}
    for img in glob(os.path.join(image_dir, '*.png')):
        img_camera_name = img.split('\\')[-1].split('_')[0]
        print(img_camera_name)
        if img_camera_name in camera_names:
            cam_image_paths[img_camera_name] = os.path.realpath(img)

    return cam_image_paths


def get_pairs(camera_names=CAMERAS):
    num = len(camera_names)
    pairs = []
    for i in range():
        cam1 = camera_names[i]
        cam2 = camera_names[i+1] if i != num else camera_names[0]
        pair = (cam1, cam2)
        pairs.append(pair)
        
    return pairs


def create_config_template():
    '''
    Augmented function from https://github.com/AlexEMG/DeepLabCut

    Creates a template for config.yaml file. This specific order is preserved while saving as yaml file.

    '''

    yaml_str = ''' \
# Project definitions
    Task:
    scorer:
    date:
    \n
# Data paths
    data_path:
    calibration_path:
    results_path:
# Project paths
    project_config:
    dlc_project_config:
    dlc3d_project_configs:
    '''
    ruamelFile = ruamel.yaml.YAML()
    cfg_file = ruamelFile.load(yaml_str)

    return (cfg_file, ruamelFile)


def write_config(config_path, cfg):
    '''
    Augmented function from https://github.com/AlexEMG/DeepLabCut

    Write structured config file.

    '''
    with open(config_path, 'w') as cf:
        ruamelFile = ruamel.yaml.YAML()
        cfg_file,ruamelFile = create_config_template()
        for key in cfg.keys():
            cfg_file[key]=cfg[key]

        ruamelFile.dump(cfg_file, cf)


def read_config(config_path):
    '''
    Augmented function from https://github.com/AlexEMG/DeepLabCut

    Reads structured config file

    '''
    ruamelFile = ruamel.yaml.YAML()
    path = Path(config_path)
    if os.path.exists(path):
        try:
            with open(path, 'r') as infile:
                cfg = ruamelFile.load(infile)
        except Exception as err:
            if err.args[2] == "could not determine a constructor for the tag '!!python/tuple'":
                with open(path, 'r') as ymlfile:
                    cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)
                    write_config(config_path,cfg)
    else:
        msg = "Config file is not pairs_with_hdf. Please make sure that the file in the path exists"
        raise FileNotFoundError (msg)

    return cfg


def create_dc_project(project_name, experimenter, dlc_project_config, dlc3d_project_configs, working_directory=None):
    '''
    Augmented function from https://github.com/AlexEMG/DeepLabCut
    
    Creates a new project directory, sub-directories and a basic configuration file.
    The configuration file is loaded with the default values. Change its parameters to your projects need.

    Parameters
    ----------
    project_name : str
        String containing the project_name of the project.

    experimenter : str
        String containing the project_name of the experimenter/scorer.

    dlc_project_config : str
        String containing the full path to the DeepLabCut project config.yaml file.

    dlc3d_project_configs : dict
        Dict with camera pair as keys, and values as strings containing the full path to the respective DeepLabCut 3D project config.yaml file.
        The camera pairs are derived from the following cameras:
        'NorthWest', 'NorthEast', 'EastNorth', 'EastSouth', 'SouthEast', 'SouthWest', 'WestSouth', 'WestNorth'

    working_directory : str, optional
        The directory where the project will be created. The default is the ``current working directory``.

    Example
    --------
    Linux/MacOs
    >>> deeplabcut.create_new_project('reaching-task', 'TorvaldsJobs', '/dlc_project/', '/dlc3d_project/' working_directory='/analysis/project/')
    >>> deeplabcut.create_new_project('reaching-task', 'TorvaldsJobs', '/dlc_project/', '/dlc3d_project/')

    Windows:
    >>> deeplabcut.create_new_project('reaching-task', 'Steve')
    >>> deeplabcut.create_new_project('reaching-task', 'Steve', Paths: r'C:\ OR 'C:\\ <- i.e. a double backslash \\\\)

    '''
    from datetime import datetime as dt

    # Initialise workflow variables
    date = dt.today()
    month = date.strftime("%B")
    day = date.day
    date = str(month[0:3] + str(day))

    if working_directory == None:
        working_directory = '.'
    working_directory = Path(working_directory).resolve()
    project_name = '{pn}-{exp}-{date}'.format(
        pn=project_name,
        exp=experimenter,
        date=dt.today().strftime('%Y-%m-%d')
    )
    project_path = working_directory / project_name
    conf_path = os.path.join(str(project_path), 'config.yaml')

    # Check if project directory is in use
    if os.path.exists(project_path):
        if os.path.exists(conf_path):
            print('There is already a project directory is in use by another project.\nPath: {}\nAborting...'.format(project_path))
            return conf_path
        else:
            print('Please empty the project directory before proceding.\nPath: {}\nAborting...')
            return False

    # Create project and sub-directories
    data_path = project_path / 'labeled-data'
    calibration_path = project_path / 'calibration-images'
    results_path = project_path / 'analysis_results'
    for path in (data_path, calibration_path, results_path):
        path.mkdir(parents=True)
        print('Created "{}"'.format(path))

    # Set values to config file:
    cfg_file, ruamelFile = create_config_template()
    cfg_file
    cfg_file['Task'] = project_name
    cfg_file['scorer'] = experimenter
    cfg_file['date'] = date

    cfg_file['data_path'] = str(data_path)
    cfg_file['results_path'] = str(results_path)
    cfg_file['calibration_path'] = str(calibration_path)

    cfg_file['project_config'] = str(project_path)
    cfg_file['dlc_project_config'] = str(dlc_project_config)
    cfg_file['dlc3d_project_configs'] = dlc3d_project_configs

    # Write dictionary to yaml config file
    write_config(conf_path, cfg_file)

    print('\nThe project has been created, and is located at:\n%s' % project_path)

    return conf_path


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
