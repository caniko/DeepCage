import yaml, ruamel.yaml
import os

from pathlib import Path


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