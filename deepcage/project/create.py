from warnings import warn
import concurrent.futures
import yaml, ruamel.yaml

from shutil import copyfile
from pathlib import Path
from glob import glob
import os

from deepcage.auxiliary.constants import CAMERAS, PAIR_IDXS, get_pairs
from deepcage.project.edit import read_config, write_config, get_dlc3d_configs

from .utils import png_to_jpg


def initialise_prepare_projects(project_name, experimenter, root, dlc_config, calib_root):
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
        String containing the full path of to the directoy where the new project files should be located
    dlc_config : string
        String containing the full path of to the dlc config.yaml file that will be used for the dlc 3D projects
    calib_root : string
        String containing the full path of to the root directory storing the calibration files for each dlc 3D project
    '''
    from deeplabcut.create_project import create_new_project_3d

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
            write_config(dlc3d_project_configs[pair], cfg)

    config_path = create_dc_project(project_name, experimenter, dlc_config, dlc3d_project_configs, working_directory=root)

    calibrate_dlc(config_path, cbrow=9, cbcol=6, calibrate=False, alpha=0.9)
    return config_path


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
