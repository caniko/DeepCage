import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import numpy as np
import ruamel.yaml

import os

from .constants import CAMERAS, PAIRS
from .auxilaryFunc import read_config, detect_images


def change_basis_func(coord_matrix, linear_map, origin):
    '''
    This function changes the basis of deeplabcut-triangulated that are 3D.
    Parameters
    ----------
    coord_matrix : numpy.array
        A 3D matrix that stores the coordinates row-wise
    linear_map : numpy.array
        (3, 3) array that stores the linear map for changing basis
    origin : numpy.array-like
        A 3D row vector, that represents the origin
    Example
    -------
    >>> deeplabcut.change_of_basis(coord_matrix, linear_map, origin=(1, 4.2, 3))
    '''
    origin = np.asarray(origin)

    assert origin.shape == (3,)
    assert len(coord_matrix.shape) == 2
    assert coord_matrix.shape[1] == 3
    assert linear_map.shape == (3, 3)

    # Change basis, and return result
    return np.apply_along_axis(
        lambda v: np.dot(linear_map, v - origin),
        1, coord_matrix
    )


def basis_label(config_path, image_paths=None):
    '''
    Parameters
    ----------
    image_paths : dict; optional
        Dictionary where the key is name of the camera, and the value is the full path to the image
        of the referance points taken with the camera
    '''
    if image_paths is None:
        camera_images = detect_images(config_path)

    axis_vectors = dict.fromkeys(CAMERAS)
    for camera, axis in CAMERAS.items():
        cam_img = camera_images[camera]

        axis_vectors[camera] = (
            {direction: [get_coord(cam_img, n=1, title=get_title(camera, axis[0][0], istip, direction)) for istip in (True, False)] for direction in ('positive', 'negative')},
            [get_coord(cam_img, n=1, title=get_title(camera, axis[1][0], istip, axis[1][1])) for istip in (True, False)],
            [get_coord(cam_img, n=1, title=get_title(camera, 'z-axis', istip, 'positive')) for istip in (True, False)]
        )

    data_path = os.path.join(read_config(config_path)['data_path'], 'labels.pickle')
    with open(data_path, 'wb') as outfile:
        stereo_file = pickle.dump(axis_vectors, outfile)

    return axis_vectors


def get_coord(cam_image, n=-1, title=None):
    '''
    Helper function for triangulate_raw_2d_camera_coords.
    User manually selects points on the provided images
    
    Parameters
    ----------
    cam_image : string; default None
        Full path of the image from camera as a string.
    cam2_image : string; default None
        Full path of the image of camera 2 as a string.
    '''
    plt.imshow(mpimg.imread(cam_image))
    if title is not None:
        plt.title(title)
    
    return plt.ginput(n=n, timeout=-1, show_clicks=True)

def get_title(camera_name, axis_name, input_istip, direction):
    return '{camera_name}\nClick on {} tip of the {} on the {} side'.format(
        'the' if input_istip else 'a point an decrement from\nthe',
        axis_name, direction, camera_name=camera_name
    )

def remove_close_zero(vector, tol=1e-16):
    ''' Returns the vector where values under the tolerance is set to 0 '''
    vector[np.abs(vector) < tol] = 0
    return vector

def unit_vector(vector):
    ''' Returns the unit vector of the vector. '''
    return vector / np.linalg.norm(vector)
