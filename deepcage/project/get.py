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

    return {
        'normal': {
            cam1: {
                ('%s positive' % CAMERAS[cam1][0][0]): raw_cam1v[0]['positive'],
                ('%s negative' % CAMERAS[cam1][0][0]): raw_cam1v[0]['negative'],
                CAMERAS[cam1][1]: raw_cam1v[1],
                'z-axis': raw_cam1v[2],
                'origin': raw_cam1v[3]
            },
            cam2: {
                ('%s positive' % CAMERAS[cam2][0][0]): raw_cam2v[0]['positive'],
                ('%s negative' % CAMERAS[cam2][0][0]): raw_cam2v[0]['negative'],
                CAMERAS[cam2][1]: raw_cam2v[1],
                'z-axis': raw_cam2v[2],
                'origin': raw_cam2v[3]
            }
        },
        'decrement': {
            cam1: {
                ('%s positive apex' % CAMERAS[cam1][0][0]): raw_cam1v[0]['positive'][0],
                ('%s positive decrement' % CAMERAS[cam1][0][0]): raw_cam1v[0]['positive'][1],
                ('%s negative' % CAMERAS[cam1][0][0]): raw_cam1v[0]['negative'][0],
                ('%s negative decrement' % CAMERAS[cam1][0][0]): raw_cam1v[0]['negative'][1],
                ('%s apex' % (CAMERAS[cam1][1],)): raw_cam1v[1][0],
                ('%s decrement' % (CAMERAS[cam1][1],)): raw_cam1v[1][1],
                'z-axis apex': raw_cam1v[2][0],
                'z-axis decrement': raw_cam1v[2][1]
            },
            cam2: {
                ('%s positive apex' % CAMERAS[cam2][0][0]): raw_cam2v[0]['positive'][0],
                ('%s positive decrement' % CAMERAS[cam2][0][0]): raw_cam2v[0]['positive'][1],
                ('%s negative' % CAMERAS[cam2][0][0]): raw_cam2v[0]['negative'][0],
                ('%s negative decrement' % CAMERAS[cam2][0][0]): raw_cam2v[0]['negative'][1],
                ('%s apex' % (CAMERAS[cam2][1],)): raw_cam2v[1][0],
                ('%s decrement' % (CAMERAS[cam2][1],)): raw_cam2v[1][1],
                'z-axis apex': raw_cam2v[2][0],
                'z-axis decrement': raw_cam2v[2][1]
            }
        }
    }
