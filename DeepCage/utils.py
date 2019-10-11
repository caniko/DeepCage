import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


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

def get_title(axis_name, input_istip, direction):
    return 'Click on {} tip of the {} on the {} side'.format(
        'the' if input_istip else 'a point an decrement from\nthe',
        axis_name, direction
    )

def remove_close_zero(vector, tol=1e-16):
    ''' Returns the vector where values under the tolerance is set to 0 '''
    vector[np.abs(vector) < tol] = 0
    return vector

def unit_vector(vector):
    ''' Returns the unit vector of the vector. '''
    return vector / np.linalg.norm(vector)

def UNSAFE_get_stereo_cam_2d_units(camera_2d_coords, sec_axis_positive_dir_coord):
    units = {}
    if sec_axis_positive_dir_coord == 'close':
        origin = pos + (neg - pos) / 2
        units = {
            'x': (x - origin) / unit_vector(x - origin),
            'z': (origin - z) / unit_vector(origin - z)
        }
        units['y'] = - np.cross(units['x'], units['z'])
    else:
        origin = neg + (pos - neg) / 2
        units = {
            'y': (origin - neg) / unit_vector(origin - neg),
            'z': (origin - z) / unit_vector(origin - z)
        }
        units['x'] = np.cross(units['y'], units['z'])

    return units
