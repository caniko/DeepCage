from numpy.linalg import solve
import pandas as pd
import numpy as np

import concurrent.futures
from collections import namedtuple
from warnings import warn
from copy import copy, deepcopy
from glob import glob
import pickle
import os

from .auxilaryFunc import read_config, get_pairs, detect_triangulation_result
from .utils import get_coord, unit_vector, get_title, basis_label, change_basis_func
from .triangulate import triangulate_raw_2d_camera_coords

from .constants import CAMERAS, PAIRS


# TODO: Create a jupyter notebook with an implementation of this workflow

def calibrate_cage(config_path, pixel_tolerance=2, save_path=None):
    '''
    Parameters
    ----------
    config_path : string
        String containing the full path of the project config.yaml file.
    pixel_tolerance : integer, float; default 2
        Defines the floor-tolerance for setting basis vector after coordinate components being set to 0. After being moved to the origin
    '''

    # Read config file and prepare variables
    cfg = read_config(config_path)
    dlc3d_configs = cfg['dlc3d_configs']
    data_path = os.path.realpath(cfg['data_path'])

    basis_result_path = os.path.join(data_path, 'cb_result.pickle')
    dataframe_path = os.path.join(data_path, 'basis_vectors.xlsx')
    if os.path.exists(basis_result_path) and os.path.exists(dataframe_path):
        msg = 'Please remove old analysis files before proceeding. File paths:\n%s\n%s\n' % (
            basis_result_path, dataframe_path
        )
    elif os.path.exists(basis_result_path):
        msg = 'Please remove old analysis file before proceeding. File paths:\n%s\n' % basis_result_path
    elif os.path.exists(dataframe_path):
        msg = 'Please remove old analysis file before proceeding. File paths:\n%s\n' % dataframe_path

    # Get labels
    label_path = os.path.join(data_path, 'labels.pickle')
    try:
        with open(label_path, 'rb') as infile:
            axis_vectors = pickle.load(infile)
    except FileNotFoundError:
        msg = 'Could not find labels in {}\nThe label process has either not been performed, or was not successful'.format(label_path)
        raise FileNotFoundError(msg)

    unit = {'map': None, 'origin': None}

    orig_map = {}
    stereo_cam_units = {}
    camera_names = tuple(CAMERAS.keys())

    pairs = get_pairs()
    for pair in pairs:
        cam1, cam2 = pair
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

        stereo_cam_units[(pair, CAMERAS[cam1][0][0])] = axis_1st
        stereo_cam_units[(pair, axis_2nd_name)] = axis_2nd
        stereo_cam_units[(pair, 'alt_%s' % axis_2nd_name)] = copy(alt_axis_2nd)
        stereo_cam_units[(pair, 'z-axis')] = z_axis

        orig_map[pair] = copy(unit)
        orig_map[pair]['origin'] = origin
        orig_map[pair]['map'] = np.array((axis_1st, axis_2nd, z_axis)).T

        print('\nCross product derived {axis_2nd_name}: {cross}\n' \
        'Origin-subtraction derived {axis_2nd_name}: {orig}\n' \
        'Ration between cross and origing: {ratio}\n'.format(
            axis_2nd_name=axis_2nd_name,
            cross=stereo_cam_units[(pair, axis_2nd_name)],
            orig=alt_axis_2nd,
            ratio=stereo_cam_units[(pair, axis_2nd_name)] / alt_axis_2nd
        ))

    with open(basis_result_path, 'wb') as outfile:
        pickle.dump(orig_map, outfile)
        print('Saved linear map to:\n{}'.format(basis_result_path))

    pd.DataFrame.from_dict(stereo_cam_units).to_excel(dataframe_path)
    print('Saved excel file containing the computed basis vectors to:\n{}'.format(dataframe_path))

    print('Returning dictionary containing the computed linear maps')
    return orig_map


def change_basis(config_path, suffix='_DLC_3D.h5'):
    '''
    This function changes the basis of deeplabcut-triangulated that are 3D.

    Parameters
    ----------
    config_path : string
        String containing the full path of the project config.yaml file.
    linear_maps : {string: numpy.array}
        (3, 3) array that stores the linear map for changing basis
    suffix : string
        The suffix in the DeepLabCut 3D project triangualtion result storage files

    Example
    -------

    '''

    coords = detect_triangulation_result(config_path, suffix=suffix, change_basis=True)
    if coords is False:
        print('According to the DeepCage result detection algorithm this project is not ready for changing basis')
        return False

    cfg = read_config(config_path)
    data_path = os.path.realpath(cfg['data_path'])
    dlc3d_configs = os.path.realpath(cfg['dlc3d_project_configs'])
    result_path = cfg['results_path']

    if len(glob(os.path.join(result_path, '*.xlsx'))) or len(glob(os.path.join(result_path, '*.h5'))):
        msg = 'The result folder needs to be empty. Path: {}'.format(result_path)
        raise ValueError(msg)

    basis_result_path = os.path.join(data_path, 'cb_result.pickle')
    with open(basis_result_path, 'rb') as infile:
        orig_map = pickle.load(infile)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        submissions = {}
        for info, rsoi in coords.items():
            pair, filename = info
            origin, linear_map = orig_map[pair]['origin'], orig_map[pair]['map']
    
            for roi, array in rsoi.items():
                submissions[executor.submit(change_basis_func, array, linear_map, origin)] = (filename, roi, pair)

        new_coords = {}
        for future in submissions:
            filename, roi, pair = submissions[future]
            if filename not in new_coords:
                new_coord[filename] = {}

            try:
                new_coords[filename][(roi, pair)] = future.result()
            except Exception as exc:
                print('%s generated an exception: %s' % (submissions[future], exc))

    print('Attempting to save new coordinates to result folder:\n%s' % result_path)
    for filename, coords in new_coords.items():
        file_path = os.path.join(result_path, 'mapped_' + filename)
        dframe = pd.DataFrame.from_dict(coords)
        
        dframe.to_hdf(file_path+'.h5')
        dframe.to_csv(file_path+'.csv')
        dframe.to_excel(file_path+'.xlsx')
        
        print('Saved', filename)

    print('Done')
    return True


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
