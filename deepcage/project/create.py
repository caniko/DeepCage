from warnings import warn
import concurrent.futures
import yaml, ruamel.yaml

from shutil import copyfile, copytree
from pathlib import Path
from glob import glob
import os

from deeplabcut.generate_training_dataset.frame_extraction import extract_frames

from deepcage.auxiliary.constants import CAMERAS, PAIR_IDXS, get_pairs
from deepcage.auxiliary.logistics import dlc3d_video_migrate

from .edit import read_config, write_config
from .get import get_dlc3d_configs


def create_project_old_cage(
        new_project_name, old_cage_config, new_root,
        video_root=None, bonvideos=None, video_extension='avi',
        dlc_project_config='', dlc_working_dir=None,
        new_experimenter=None,
    ):
    """
    Create new DeepCageProject assembly with the use of a different cage proejct.
    Function used for migrating DeepCage constants to other projects
    
    The assembly migrations includes all DeepLabCut 3D projects, (optional) creating a new DLC 2D project,
    copying new videos to the new project.
    
    Parameters
    ----------
    new_project_name : str
        String containing the name of the new project.
        
    old_cage_config : str
        Absolute path to the old(!) DeepCage project config.yaml file.

    new_root : str-like
        Object containing the full path to the project, must have __str__() or method accepted by pathlib.Path()

    dlc_project_config : str; default '' (empty-string)
        Absolute path to the DeepLabCut project config.yaml file.

    dlc_working_dir : str; default, None -> project_path
        The directory where the optional DeepLabCut 2D project will be created.

    new_experimenter : str; default None
        String containing the project_name of the new experimenter/scorer.

    """

    from deeplabcut.utils.auxiliaryfunctions import write_config_3d

    # Prepare simple variables
    cfg = read_config(old_cage_config)
    dlcrd_cfg_paths = get_dlc3d_configs(old_cage_config)
    experimenter = cfg['scorer'] if new_experimenter is None else new_experimenter

    # Create DeepCage project
    new_cfg_path = create_dc_project(
        new_project_name, experimenter, dlc_project_config,
        dlc3d_project_configs={}, working_directory=new_root,
    )
    project_path = os.path.dirname(new_cfg_path)
    new_cfg = read_config(new_cfg_path)

    if video_root is not None:
        rem_videos = glob(os.path.join(video_root, ('**/*.%s' % video_extension) ))
    elif bonvideos is not None:
        from deepcage.auxiliary.detect import detect_bonsai

        rem_videos = detect_bonsai(bonvideos)
    else:
        msg = 'No DeepLabCut 2D project was provided along with no video source, can not create dlc 2d project'
        raise ValueError(msg)

    # Dedicate DeepLabCut 2D project (pre-defined or new; arg dependent)
    if dlc_project_config is '':
        from deeplabcut.create_project.new import create_new_project

        dlc_create_root = dlc_working_dir if dlc_working_dir is not None else project_path
        dlc_path = create_new_project(
            new_project_name, experimenter, rem_videos,
            working_directory=dlc_create_root, copy_videos=True, videotype='.avi'
        )
    else:
        dlc_path = dlc_project_config
    new_cfg['dlc_project_config'] = dlc_path    # Add DLC 2D project to new DeepCage config file

    # Create a copy of every DLC 3d project in the previous project
    new_dlc_path = os.path.join(os.path.dirname(new_cfg_path), 'DeepLabCut')
    if not os.path.exists(new_dlc_path):
        os.mkdir(new_dlc_path)

    new_dlc3d_cfg_paths = {}
    for pair, path in dlcrd_cfg_paths.items():
        cam1, cam2 = pair
        dlc3d_root = os.path.dirname(path)
        dlc3d_id = os.path.basename(dlc3d_root)

        # Create the project folder
        new_dlc3d_root = os.path.join(new_dlc_path, dlc3d_id)
        os.mkdir(new_dlc3d_root)
        copytree(
            os.path.join(dlc3d_root, 'camera_matrix'),
            os.path.join(new_dlc3d_root, 'camera_matrix')
        )
        copyfile(
            os.path.join(dlc3d_root, 'config.yaml'),
            os.path.join(new_dlc3d_root, 'config.yaml')
        )

        # Modify the copied config.yaml file
        new_dlc3d_cfg_path = os.path.join(new_dlc3d_root, 'config.yaml')
        new_dlc3d_cfg = read_config(new_dlc3d_cfg_path)
        new_dlc3d_cfg['project_path'] = new_dlc3d_root
        new_dlc3d_cfg['config_file_%s' % cam1] = dlc_path
        new_dlc3d_cfg['config_file_%s' % cam2] = dlc_path
        write_config_3d(new_dlc3d_cfg_path, new_dlc3d_cfg)

        new_dlc3d_cfg_paths[pair] = new_dlc3d_root

    new_cfg['dlc3d_project_configs'] = new_dlc3d_cfg_paths
    write_config(new_cfg_path, new_cfg)

    if video_root is not None:
        dlc3d_video_migrate(new_cfg_path, video_root)
        print('Copied videos from %s to new project' % video_root)

    print('Created new DeepCage project:\n%s' % project_path)
    if dlc_project_config == '':
        print('New DeepLabCut 2D project is located in\n%s' % dlc_create_root)

    # Extracting frames using k-means
    extract_frames(path_config_file, userfeedback=False)


def create_dc_project(
        project_name, experimenter, dlc_project_config,
        dlc3d_project_configs, working_directory=None, dlc_init=False
    ):
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
        Absolute path to the DeepLabCut project config.yaml file.

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
    if dlc_init is False and os.path.exists(project_path):
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
