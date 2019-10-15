from numpy.linalg import solve
import numpy as np

from copy import copy
import pickle
import os

from .constants import CAMERAS, PAIRS
from .auxilaryFunc import read_config
from .utils import get_coord, unit_vector, get_title, basis_label
from .triangulate import triangulate_raw_2d_camera_coords


# TODO: Create a jupyter notebook with an implementation of this workflow

def get_cage_basis(config, pixel_tolerance=2, save_path=None):
    '''
    Parameters
    ----------
    config : string
        String containing the full path of the project config.yaml file.
    pixel_tolerance : integer, float; default 2
        Defines the floor-tolerance for setting basis vector after coordinate components being set to 0. After being moved to the origin
    '''
    
    # Read config file and prepare variables
    cfg = read_config(config)
    label_path = os.path.join(cfg['data_path'], 'labels.pickle')
    dlc3d_configs = os.path.realpath(cfg['dlc3d_configs'])
    
    # Get labels
    try:
        with open(label_path, 'rb') as infile:
            axis_vectors = pickle.load(infile)
    except FileNotFoundError:
        msg = 'Could not find labels in {}\nThe label process has either not been performed, or was not successful'.format(label_path)
        raise FileNotFoundError(msg)

    stereo_cam_units = {}
    camera_names = tuple(CAMERAS.keys())
    for i in range(PAIRS):
        cam1 = camera_names[i]
        cam2 = camera_names[i+1] if i != PAIRS else camera_names[0]
        pair = (cam1, cam2)
        print('Calculating the basis vectors of {}'.format(pair))

        # Preparing for triangualting image coordinates
        raw_cam1v = axis_vectors[cam1]
        raw_cam2v = axis_vectors[cam2]
        if CAMERAS[cam1][2] == CAMERAS[cam2][2]:
            i, ii, iii = CAMERAS[cam1][0][0], CAMERAS[cam1][1][0], 'z-axis'
            assert CAMERAS[cam1] == CAMERAS[cam2]
            unit_keys = ((CAMERAS[cam1][0][0], 'positive'), CAMERAS[cam1][1], 'z-axis')

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
            # Corner
            assert CAMERAS[cam1][1][0] == CAMERAS[cam2][0]
            assert CAMERAS[cam1][0][0] == CAMERAS[cam2][1][0]
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

        trian = triangulate_raw_2d_camera_coords(
            dlc3d_configs[pair], cam1_coords=cam1v, cam2_coords=cam2v
        )

        if CAMERAS[cam1][2] == CAMERAS[cam2][2]:
            axis_2nd_name = 'y-axis' if CAMERAS[cam1][0][0] == 'x-axis' else 'x-axis'
            if CAMERAS[cam1][0][1] == 'close':
                origin = trian[0] + (trian[1] - trian[0]) / 2    # pos + (trian[1] - pos) / 2
                z_axis = unit_vector(trian[3] - origin)
                axis_1st = unit_vector(trian[0] - origin)
                axis_2nd = unit_vector(np.cross(axis_1st, z_axis))
            else:
                origin = trian[1] + (trian[0] - trian[1]) / 2
                z_axis = unit_vector(trian[3] - origin)
                axis_1st = unit_vector(origin - trian[1])
                axis_2nd = unit_vector(np.cross(z_axis, axis_1st))

            if CAMERAS[cam2][1][1] == 'positive':
                alt_axis_2nd = origin - trian[2]
            else:
                alt_axis_2nd = trian[2] - origin
            
        else:
            # Corner
            first_axis_linev = unit_vector(trian[1] - trian[0])
            z_axis_linev = unit_vector(trian[5] - trian[4])
            tangent_v = unit_vector(np.corss(z_axis_linev, first_axis_linev))

            LHS = np.array((first_axis_linev, -z_axis_linev, tangent_v)).T
            RHS = trian[4] - trian[0]
            sol = solve(LHS, RHS)

            origin = trian[0] + first_axis_linev * sol[0] + tangent_v * sol[2]/2
            z_axis = unit_vector(trian[4] - origin)
            
            if CAMERAS[cam1][1][1] == 'positive':
                axis_1st = unit_vector(trian[0] - origin)
                axis_2nd = unit_vector(np.cross(axis_1st, z_axis))
            else:
                axis_1st = unit_vector(origin - trian[0])
                axis_2nd = unit_vector(np.cross(z_axis, axis_1st))

            if CAMERAS[cam2][1][1] == 'positive':
                alt_axis_2nd = trian[2] - origin
            else:
                alt_axis_2nd = origin - trian[2]
                
        stereo_cam_units[pair] = {
            CAMERAS[cam1][0][0]: axis_1st,
            axis_2nd_name: axis_2nd,
            'z-axis': z_axis
        }
        
        print('\nCross product derived {axis_2nd_name}: {cross}\n' \
        'Origin-subtraction derived {axis_2nd_name}: {orig}\n' \
        'Ration between cross and orig: {ratio}\n'.format(
            axis_2nd_name=axis_2nd_name,
            cross=stereo_cam_units[pair][axis_2nd_name],
            orig=alt_axis_2nd,
            ratio=stereo_cam_units[pair][axis_2nd_name] / alt_axis_2nd
        ))

    return stereo_cam_units


def change_basis(coord_matrix, origin, x, y, z):
    '''
    This function changes the basis of deeplabcut-triangulated that are 3D.
    
    Note: Two out of three axis should be defined. The third axis is the normal vector

    Parameters
    ----------
    coord_matrix : numpy.array
        A 3D matrix that stores the coordinates row-wise
    origin : numpy.array-like
        A 3D row vector, that represents the origin
    x : numpy.array-like; default None
        A 3D row vector, that represents the basis of the new x axis
    y : numpy.array-like; default None
        A 3D row vector, that represents the basis of the new y axis
    z : numpy.array-like; default None
        A 3D row vector, that represents the basis of the new z axis

    Example
    -------
    >>> deeplabcut.change_of_basis(coord_matrix, origin=(1, 4.2, 3), x=(2.4, 0, 0), y=(0,  5.3, 0), z=(1, 0, 4.1))

    '''
    transform_matrix = np.array((x, y, z))
    origin = np.asarray(origin)

    # Change basis, and return result
    return np.apply_along_axis(
        lambda v: np.dot(transform_matrix, v - origin),
        1, coord_matrix
    )


def get_basis(dlc3d_configs, image_paths, camera_pairs, user_defined_axis=['x', 'z'], pixel_tolerance=2):
    '''
    Parameters
    ----------
    dlc3d_configs : dict
        Dictionary where the key is name of the camera, and the value is the full path of the config.yaml file
        as a string.
    image_paths : dict
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
            
    cam_coords = dict.fromkeys(image_paths)
    for cam_name, cam_img in image_paths.items():
        cam_coords[cam_name] = []
        max_i = len(coord_labels) - 1
        for coord_name in coord_labels:
            title = '%s: left mouse click add point' % coord_name
            cam_coords[cam_name].append(get_coord(cam_img, n=1, title=title))

    basis_of_pairs = {}
    basis_dict = dict.fromkeys(coord_labels)
    for cam1_name, cam2_name in camera_pairs:
        coords = triangulate_raw_2d_camera_coords(
            dlc3d_configs[(cam1_name, cam2_name)],
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
