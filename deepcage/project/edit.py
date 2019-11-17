from shutil import copyfile
import yaml, ruamel.yaml
from pathlib import Path
from glob import glob
import os

from .get import get_dlc3d_configs


def write_config(config_path, cfg):
    '''
    Augmented function from https://github.com/AlexEMG/DeepLabCut

    Write structured config file.

    '''
    from deepcage.project.create import create_config_template

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
        msg = 'Config file not found in config_path:\n%s' % config_path
        raise FileNotFoundError (msg)

    return cfg


def add_bonvideos_dlc3d(config_path, video_root, vformat='avi'):
    '''
    Parameters
    ----------
    vid_format : string; default 'avi'
    '''
    dlc3d_cfgs = get_dlc3d_configs(config_path)
    subdirs = glob(os.path.join(video_root, '*/'))
    for subdir in subdirs:
        bon, animal, trial, date = os.path.basename(subdir[:-1]).split('_')
        assert bon == 'BonRecordings'
        print('Moving videos of experiment with info:\nanimal %s; trial: %s; date %s' % (
            animal, trial, date
        ))

        video_dirs = glob(os.path.join(subdir, '*/'))
        for video_dir in video_dirs:
            pair_dirs = glob(os.path.join(subdir, '*/'))
            for pair_dir in pair_dirs:
                pair_id, cam1, cam2 = os.path.basename(pair_dir[:-1]).split('_')
                project_root = os.path.dirname(dlc3d_cfgs[(cam1, cam2)])

                video_dir = os.path.join(project_root, 'videos')
                if not os.path.exists(video_dir):
                    os.mkdir(video_dir)
                for video in glob(os.path.join(pair_dir, '*.'+vformat)):
                    cam = os.path.basename(video).split('_')[2]
                    new_name = '%s_%s_%s_%s.%s' % (animal, trial, cam, date, vformat)
                    copyfile(os.path.realpath(video), os.path.join(video_dir, new_name))

    return True
