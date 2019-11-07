import pickle
import os

from .edit import read_config


def get_labels(config_path):
    cfg = read_config(config_path)
    data_path = os.path.realpath(cfg['data_path'])
    label_path = os.path.join(data_path, 'labels.pickle')
    try:
        with open(label_path, 'rb') as infile:
            return pickle.load(infile)
    except FileNotFoundError:
        msg = 'Could not find labels in {}\nThe label process has either not been performed, or was not successful'.format(label_path)
        raise FileNotFoundError(msg)
