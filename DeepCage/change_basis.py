import numpy as np

from copy import copy
import os

from .utils import get_coord, unit_vector, get_title, get_stereo_cam_2d_units
from .triangualte import triangulate_raw_2d_camera_coords


# TODO: Create a jupyter notebook with an implementation of this workflow

def get_basis(config_dict, image_dict, camera_pairs, user_defined_axis=['x', 'z'], pixel_tolerance=2):
    '''
    Parameters
    ----------
    config_dict : dict
        Dictionary where the key is name of the camera, and the value is the full path of the config.yaml file
        as a string.
    image_dict : dict
        Dictionary where the key is name of the camera, and the value is the full path to the image
        of the referance points taken with the camera
    camera_pairs : list-like
        List of cameras that are pairs. Pairs usually have their own deeplabcut 3D project
    user_defined_axis : dictionary with max length 2
        Dictionary where the key are the axis to be selected by the user, choose between 'x-axis', 'y-axis', 'z-axis'; 1-, 2-, 3-dimension.
        There must be two axis names, the third axis is defined using the cross product.
        
        The values of the keys is the secondary_axis_locations of the defined axis. This is important for the calculation of the third axis (first right hand rule)
    pixel_tolerance : integer, float; default 2
        Defines the floor-tolerance for setting basis vector after coordinate components being set to 0. After being moved to the origin
    ''' 
    VALID_AXIS_NAMES = ['x', 'y', 'z', 1, 2, 3]
    coord_labels = ['origin']
    for axis_name in user_defined_axis:
        if axis_name in VALID_AXIS_NAMES:
            coord_labels.append(axis_name)
        else:
            msg = 'Invalid axis name, {}. Valid names: {}'.format(axis_name, VALID_AXIS_NAMES)
            raise ValueError(msg)
            
    cam_coords = dict.fromkeys(image_dict)
    for cam_name, cam_img in image_dict.items():
        cam_coords[cam_name] = []
        max_i = len(coord_labels) - 1
        for coord_name in coord_labels:
            title = '%s: left mouse click add point' % coord_name
            cam_coords[cam_name].append(get_coord(cam_img, n=1, title=title))

    basis_of_pairs = {}
    basis_dict = dict.fromkeys(coord_labels)
    for cam1_name, cam2_name in camera_pairs:
        coords = triangulate_raw_2d_camera_coords(
            config_dict[(cam1_name, cam2_name)],
            cam1_coords=cam_coords[cam1_name],
            cam2_coords=cam_coords[cam2_name],
            unit_keys=coord_labels
        )

        basis_of_pairs[(cam1_name, cam2_name)] = copy(basis_dict)
        basis_of_pairs[(cam1_name, cam2_name)]['origin'] = np.array(coords['origin'])
        for user_axis in user_defined_axis:
            basis_of_pairs[(cam1_name, cam2_name)][user_axis] = unit_vector(
                remove_close_zero(coords[user_axis] - coords['origin'], tol=pixel_tolerance)
            )

    return basis_of_pairs

def get_cage_basis(config_dict, image_dict, pixel_tolerance=2, save_path=None):
    '''
    Parameters
    ----------
    config_dict : dict
        Dictionary where the key is name of the stereo camera pair, and the value is the full path of the config.yaml file
    image_dict : dict
        Dictionary where the key is name of the camera, and the value is the full path to the image
        of the referance points taken with the camera
    camera_pairs : list-like
        List of cameras that are pairs. Pairs usually have their own deeplabcut 3D project
    pixel_tolerance : integer, float; default 2
        Defines the floor-tolerance for setting basis vector after coordinate components being set to 0. After being moved to the origin
    '''

    PAIRS = 8
    CAMERAS = {
            'NorthWest': ('x-axis', ('y-axis', 'positive'), 'close', 1), 'NorthEast': ('x-axis', ('y-axis', 'positive'), 'close', 1),
            'EastNorth': ('y-axis',  ('x-axis', 'positive'), 'far', 2), 'EastSouth': ('y-axis',  ('x-axis', 'positive'), 'far', 2),
            'SouthEast': ('x-axis', ('y-axis', 'negative'), 'far', 3), 'SouthWest': ('x-axis', ('y-axis', 'negative'), 'far', 3),
            'WestSouth': ('y-axis',  ('x-axis', 'negative'), 'close', 4), 'WestNorth': ('y-axis',  ('x-axis', 'negative'), 'close', 4)
    }

    axis_vectors = dict.fromkeys(CAMERAS)
    for camera, axis in CAMERAS.items():
        cam_img = image_dict[camera]
        
        axis_vectors[camera] = (
            {direction: [get_coord(cam_img, n=1, title=get_title(axis[0], istip, direction)) for istip in (True, False)] for direction in ('positive', 'negative')},
            [get_coord(cam_img, n=1, title=get_title(axis[1][0], istip, axis[1][1])) for istip in (True, False)],
            [get_coord(cam_img, n=1, title=get_title('z-axis', istip, 'positive')) for istip in (True, False)]
        )

    stereo_cam_units = {}    
    camera_names = tuple(CAMERAS.keys())
    for i in range(PAIRS):
        cam1 = camera_names[i]
        cam2 = camera_names[i+1] if i != PAIRS else camera_names[0]
        pair = (cam1, cam2)

        raw_cam1v = axis_vectors[cam1]
        raw_cam2v = axis_vectors[cam2]
        if CAMERAS[cam1][3] == CAMERAS[cam2][3]:
            i, ii, iii = CAMERAS[cam1][0], CAMERAS[cam1][1][0], 'z-axis'
            assert CAMERAS[cam1] == CAMERAS[cam2]
            unit_keys = ((CAMERAS[cam1][0], 'positive'), CAMERAS[cam1][1], 'z-axis')

            cam1v = (
                raw_cam1v[0]['positive'][0], raw_cam1v[0]['negative'][0],   # NorthNorth: x-axis
                raw_cam1v[1][0],                                            # NorthNorth: y-axis
                raw_cam1v[2][0]                                             # z-axis
            )
            cam2v = (
                raw_cam2v[0]['positive'][0], raw_cam2v[0]['negative'][0],
                raw_cam2v[1][0],
                raw_cam2v[2][0]
            )
        else:
            assert CAMERAS[cam1][1][0] == CAMERAS[cam2][0]
            assert CAMERAS[cam1][0] == CAMERAS[cam2][1][0]
            unit_keys = (CAMERAS[cam1][1], CAMERAS[cam2][1], 'z-axis')

            cam1v = (
                raw_cam1v[1][0], raw_cam1v[1][1],                                                   # EastSouth: x positive     SouthWest: y negative
                raw_cam1v[0][ CAMERAS[cam2][1][1] ][0], raw_cam1v[0][ CAMERAS[cam2][1][1] ][1],     # EastSouth: y negative     SouthWest: x negative
                raw_cam1v[2][0], raw_cam1v[2][1]
            )
            cam2v = (
                raw_cam2v[0][ CAMERAS[cam1][1][1] ][0], raw_cam2v[0][ CAMERAS[cam1][1][1] ][1],     # EastSouth: x positive     SouthWest: y negative
                raw_cam2v[1][0], raw_cam2v[1][1],                                                   # EastSouth: y negative     SouthWest: x negative
                raw_cam2v[2][0], raw_cam2v[2][1]
            )

        coords = triangulate_raw_2d_camera_coords(
            config_dict[pair], cam1_coords=cam1v, cam2_coords=cam2v
        )

        if CAMERAS[cam1][3] == CAMERAS[cam2][3]:
            last_axis = 'y-axis' if CAMERAS[cam1][0] == 'x-axis' else 'x-axis'
            if CAMERAS[cam1][2] == 'close':
                origin = coords[0] + (coords[1] - coords[0]) / 2    # pos + (coords[1] - pos) / 2
                stereo_cam_units[pair] = {
                    CAMERAS[cam1][0]: (coords[0] - origin) / unit_vector(coords[0] - origin),
                    'z-axis': (origin - coords[3]) / unit_vector(origin - coords[3])
                }
                stereo_cam_units[pair][last_axis] = - np.cross(
                    stereo_cam_units[pair][CAMERAS[cam1][0]],
                    stereo_cam_units[pair]['z-axis']
                )
            else:
                origin = coords[1] + (coords[0] - coords[1]) / 2
                stereo_cam_units[pair] = {
                    CAMERAS[cam1][0]: (origin - coords[1]) / unit_vector(origin - coords[1]),
                    'z': (origin - coords[3]) / unit_vector(origin - coords[3])
                }
                stereo_cam_units[pair][last_axis] = np.cross(
                    stereo_cam_units[pair][CAMERAS[cam1][0]],
                    stereo_cam_units[pair]['z-axis']
                )
            alt_y_unit = (origin - coords[2]) / unit_vector(origin - coords[2])
        

    return cam_units


def change_of_basis(column_matrix, x=None, y=None, z=None, origin=np.array((0, 0, 0)) ):
    '''
    This function changes the basis of deeplabcut-triangulated that are 3D.
    
    Note: Two out of three axis should be defined. The third axis is the normal vector

    Parameters
    ----------
    column_matrix : numpy.array
        A 3D array that holds the coordinates that will have their basis changed
    x : numpy.array-like; default None
        A 3D row vector, that represents the basis of the new x axis
    y : numpy.array-like; default None
        A 3D row vector, that represents the basis of the new y axis
    z : numpy.array-like; default None
        A 3D row vector, that represents the basis of the new z axis
    origin : numpy.array-like; default np.array((0, 0, 0))
        A 3D row vector, that represents the origin

    Example
    -------
    With random basis vectors:
    >>> deeplabcut.change_of_basis(column_matrix, x=(2.4, 0, 0), y=(0,  5.3, 0), z=None, origin=np.array((0, 0, 0))).

    '''
    basis_dict = {'x': x, 'y': y, 'z': z}
    known_basis = []
    last_axis = None
    for axis, basis in basis_dict.items():
        if basis is None:
            if last_axis is not None:
                msg = 'Two out of three axis should be defined. The third axis is the normal vector'
                raise AttributeError(msg)
            last_axis = axis
        else:
            basis = np.asarray(basis)
            known_basis.append(axis)

    if basis_dict[known_basis[0]].shape != basis_dict[known_basis[1]].shape:
        msg = 'The basis vectors (%s and %s) need to have the same shape' % known_basis
        raise AttributeError(msg)
    if basis_dict[known_basis[0]].shape != (3,):
        msg = 'The basis vectors (%s and %s) can only have the shape (3,)\n' \
              'In other words, they have to be row vectors' % known_basis
        raise AttributeError(msg)

    # Make sure each basis vector are unit vectors
    basis_dict[known_basis[0]] = unit_vector(basis_dict[known_basis[0]])
    basis_dict[known_basis[1]] = unit_vector(basis_dict[known_basis[1]])

    basis_dict[last_axis] = np.cross(basis_dict[known_basis[0]], basis_dict[known_basis[1]])
    linear_transformer = np.array((tuple(basis_dict.values())))

    # Change basis, and return result
    return np.apply_along_axis(
        lambda v: np.dot(linear_transformer, v - np.asarray(origin)),
        1, column_matrix
    )
