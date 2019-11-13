import pickle
import os

from deepcage.auxiliary.constants import CAMERAS

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


def get_paired_labels(config_path, pair):
    cam1, cam2 = pair
    
    labels = get_labels(config_path)
    raw_cam1v, raw_cam2v = labels[cam1], labels[cam2]
    raw_cam1v[0]['positive']

    if CAMERAS[cam1][2] == CAMERAS[cam2][2]:
        assert CAMERAS[cam1] == CAMERAS[cam2], '%s != %s' % (CAMERAS[cam1], CAMERAS[cam2])
    else:
        assert CAMERAS[cam1][1][0] == CAMERAS[cam2][0][0], '%s != %s' % (CAMERAS[cam1][1][0], CAMERAS[cam2][0][0])
        assert CAMERAS[cam1][0][0] == CAMERAS[cam2][1][0], '%s != %s' % (CAMERAS[cam1][0][0], CAMERAS[cam2][1][0])

    return {
        'normal': {
            cam1: {
                (CAMERAS[cam1][0][0], 'positive'): raw_cam1v[0]['positive'],    # NorthNorth: x-axis
                (CAMERAS[cam1][0][0], 'negative'): raw_cam1v[0]['negative'],
                CAMERAS[cam1][1]: raw_cam1v[1],                                     # NorthNorth: y-axis
                'z-axis': raw_cam1v[2],
                'origin': raw_cam1v[3]
            },
            cam2: {
                (CAMERAS[cam2][0][0], 'positive'): raw_cam2v[0]['positive'],
                (CAMERAS[cam2][0][0], 'negative'): raw_cam2v[0]['negative'],
                CAMERAS[cam2][1]: raw_cam2v[1],
                'z-axis': raw_cam2v[2],
                'origin': raw_cam2v[3]
            }
        },
        'decrement': {
            cam1: {
                (CAMERAS[cam1][0][0], 'positive', 'apex'): raw_cam1v[0]['positive'][0],
                (CAMERAS[cam1][0][0], 'positive', 'decrement'): raw_cam1v[0]['positive'][1],
                (CAMERAS[cam1][0][0], 'negative', 'apex'): raw_cam1v[0]['negative'][0],
                (CAMERAS[cam1][0][0], 'negative', 'decrement'): raw_cam1v[0]['negative'][1],
                (*CAMERAS[cam1][1], 'apex'): raw_cam1v[1][0],
                (*CAMERAS[cam1][1], 'decrement'): raw_cam1v[1][1],
                ('z-axis', 'apex'): raw_cam1v[2][0],
                ('z-axis', 'decrement'): raw_cam1v[2][1]
            },
            cam2: {
                (CAMERAS[cam2][0][0], 'positive', 'apex'): raw_cam2v[0]['positive'][0],
                (CAMERAS[cam2][0][0], 'positive', 'decrement'): raw_cam2v[0]['positive'][1],
                (CAMERAS[cam2][0][0], 'negative', 'apex'): raw_cam2v[0]['negative'][0],
                (CAMERAS[cam2][0][0], 'negative', 'decrement'): raw_cam2v[0]['negative'][1],
                (*CAMERAS[cam2][1], 'apex'): raw_cam2v[1][0],
                (*CAMERAS[cam2][1], 'decrement'): raw_cam2v[1][1],
                ('z-axis', 'apex'): raw_cam2v[2][0],
                ('z-axis', 'decrement'): raw_cam2v[2][1]
            }
        }
    }
