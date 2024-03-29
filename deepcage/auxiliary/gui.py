import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import gc

import pickle
from glob import glob
import os

from deepcage.project.edit import read_config

from .detect import detect_bonsai, detect_cage_calibration_images
from .constants import CAMERAS


def get_title(camera_name, axis_name, direction, istip):
    return '{camera_name}\nClick on {} tip of the {} on the {} side'.format(
        'the' if istip else 'a point an decrement from\nthe',
        axis_name, direction, camera_name=camera_name
    )
    

def get_coord(cam_image, n=-1, title=None):
    '''
    Helper function for triangulate_raw_2d_camera_coords.
    User manually selects points on the provided images
    
    Parameters
    ----------
    cam_image : string; default None
        Absolute path of the image from camera as a string.
    cam2_image : string; default None
        Absolute path of the image of camera 2 as a string.
    '''

    plt.imshow(mpimg.imread(cam_image))
    if title is not None:
        plt.title(title)

    pick = plt.ginput(n=n, timeout=-1, show_clicks=True)[0]

    return pick


def basis_label(config_path, decrement=False, detect=True, name_pos=0, format='png', image_paths=None):
    '''
    Parameters
    ----------
    config_path : string
        Absolute path of the project config.yaml file.
    image_paths : dict; optional
        Dictionary where the key is name of the camera, and the value is the full path to the image
        of the referance points taken with the camera
    '''
    if image_paths is None:
        camera_images = detect_cage_calibration_images(config_path, name_pos=name_pos)

    n = -1
    basis_labels = dict.fromkeys(camera_images.keys() if detect is True else CAMERAS.keys())
    for camera in basis_labels.keys():
        cam_img = camera_images[camera]

        if decrement is True:
            basis_labels[camera] = (
                {direction: [get_coord(cam_img, n=n, title=get_title(camera, CAMERAS[camera][0][0], direction, istip)) for istip in (True, False)] for direction in ('positive', 'negative')},
                [get_coord(cam_img, n=n, title=get_title(camera, CAMERAS[camera][1][0], CAMERAS[camera][1][1], istip)) for istip in (True, False)],
                [get_coord(cam_img, n=n, title=get_title(camera, 'z-axis', 'positive', istip)) for istip in (True, False)]
            )
        else:
            basis_labels[camera] = (
                {direction: get_coord(cam_img, n=n, title=get_title(camera, CAMERAS[camera][0][0], True, direction)) for direction in ('positive', 'negative')},
                get_coord(cam_img, n=n, title=get_title(camera, CAMERAS[camera][1][0], CAMERAS[camera][1][1], True)),
                get_coord(cam_img, n=n, title=get_title(camera, 'z-axis', 'positive', True)),
                get_coord(cam_img, n=n, title='Select origin')
            )

        plt.close()
        gc.collect()

    data_path = os.path.join(read_config(config_path)['data_path'], 'labels.pickle')
    with open(data_path, 'wb') as outfile:
        stereo_file = pickle.dump(basis_labels, outfile)

    return basis_labels


def alter_basis_label(config_path, camera, index=None, image_paths=None):
    data_path = os.path.join(read_config(config_path)['data_path'], 'labels.pickle')
    with open(data_path, 'rb') as infile:
        basis_labels = pickle.load(infile)
        
    if image_paths is None:
        camera_images = detect_cage_calibration_images(config_path)
