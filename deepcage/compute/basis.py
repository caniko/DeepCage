from numpy.linalg import solve
import pandas as pd
import numpy as np
import vg

from psutil import cpu_count
from warnings import warn
import concurrent.futures
from glob import glob
import pickle
import os

from deepcage.auxiliary.detect import detect_triangulation_result
from deepcage.project.edit import read_config, get_dlc3d_configs
from deepcage.auxiliary.constants import CAMERAS, get_pairs

from .triangulate import triangulate_raw_2d_camera_coords
from .utils import unit_vector, change_basis_func


# TODO: Create a jupyter notebook with an implementation of this workflow

def stereo_cam_info(config_path, axis_vectors, pair, decrement=False):
    '''
    Using the labels from basis_label create 2d representations of the basis vectors for triangulation

    '''
    cam1, cam2 = pair
    print('Calculating the basis vectors of %s %s' % pair)
    # i, ii, iii = CAMERAS[cam1][0][0], CAMERAS[cam1][1][0], 'z-axis'

    # Preparing for triangualting image coordinates
    raw_cam1v = axis_vectors[cam1]
    raw_cam2v = axis_vectors[cam2]
    
    if CAMERAS[cam1][2] == CAMERAS[cam2][2]:
        assert CAMERAS[cam1] == CAMERAS[cam2], '%s != %s' % (CAMERAS[cam1], CAMERAS[cam2])
        unit_keys = [CAMERAS[cam1][0][0], CAMERAS[cam1][1][0], 'z-axis']
    else:
        assert CAMERAS[cam1][1][0] == CAMERAS[cam2][0][0], '%s != %s' % (CAMERAS[cam1][1][0], CAMERAS[cam2][0][0])
        assert CAMERAS[cam1][0][0] == CAMERAS[cam2][1][0], '%s != %s' % (CAMERAS[cam1][0][0], CAMERAS[cam2][1][0])
        unit_keys = [CAMERAS[cam1][1], CAMERAS[cam2][1], 'z-axis']
    if decrement is False:
        unit_keys.append('origin')

    if decrement is True:
        if CAMERAS[cam1][2] == CAMERAS[cam2][2]:
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
            cam1v = (
                raw_cam1v[1][0], raw_cam1v[1][1],                                                   # EastSouth: x positive
                raw_cam1v[0][ CAMERAS[cam2][1][1] ][0], raw_cam1v[0][ CAMERAS[cam2][1][1] ][1],     # EastSouth: y negative
                raw_cam1v[2][0], raw_cam1v[2][1]
            )
            cam2v = (
                raw_cam2v[0][ CAMERAS[cam1][1][1] ][0], raw_cam2v[0][ CAMERAS[cam1][1][1] ][1],     # SouthEast: x positive
                raw_cam2v[1][0], raw_cam2v[1][1],                                                   # SouthEast: y negative
                raw_cam2v[2][0], raw_cam2v[2][1]
            )
    else:
        if CAMERAS[cam1][2] == CAMERAS[cam2][2]:
            cam1v = (
                raw_cam1v[0]['positive'], raw_cam1v[0]['negative'],     # NorthNorth: x-axis
                raw_cam1v[1],                                           # NorthNorth: y-axis
                raw_cam1v[2],                                           # z-axis
                raw_cam1v[3]
            )
            cam2v = (
                raw_cam2v[0]['positive'], raw_cam2v[0]['negative'],
                raw_cam2v[1],
                raw_cam2v[2],
                raw_cam1v[3]
            )

        else:
            # Corner
            cam1v = (
                raw_cam1v[1],                           # EastSouth: x positive
                raw_cam1v[0][ CAMERAS[cam2][1][1] ],    # EastSouth: y negative
                raw_cam1v[2],
                raw_cam1v[3]
            )
            cam2v = (
                raw_cam2v[0][ CAMERAS[cam1][1][1] ],     # SouthEast: x positive
                raw_cam2v[1],                            # SouthEast: y negative
                raw_cam2v[2],
                raw_cam2v[3]
            )

    dlc3d_cfg = get_dlc3d_configs(config_path)[pair]
    trian = triangulate_raw_2d_camera_coords(
        dlc3d_cfg, cam1_coords=cam1v, cam2_coords=cam2v
    )

    if decrement is True:
        if CAMERAS[cam1][2] == CAMERAS[cam2][2]:
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
            tangent_v = unit_vector(np.cross(z_axis_linev, first_axis_linev))

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
    else:
        if CAMERAS[cam1][2] == CAMERAS[cam2][2]:
            if CAMERAS[cam1][0][1] == 'close':
                origin = trian[0] + (trian[1] - trian[0]) / 2    # pos + (trian[1] - pos) / 2
                z_axis = unit_vector(trian[3] - origin)
                axis_1st = unit_vector(trian[0] - origin)
                # axis_2nd = unit_vector(np.cross(z_axis, axis_1st))
                axis_2nd = unit_vector(np.cross(axis_1st, z_axis))
            else:
                origin = trian[1] + (trian[0] - trian[1]) / 2
                z_axis = unit_vector(trian[3] - origin)
                axis_1st = unit_vector(origin - trian[1])
                # axis_2nd = unit_vector(np.cross(axis_1st, z_axis))
                axis_2nd = unit_vector(np.cross(z_axis, axis_1st))

            if CAMERAS[cam2][1][1] == 'positive':
                alt_axis_2nd = origin - trian[2]
            else:
                alt_axis_2nd = trian[2] - origin
        else:
            origin = trian[-1]
            z_axis = unit_vector(trian[2] - origin)
            if CAMERAS[cam1][1][1] == 'positive':
                axis_1st = unit_vector(trian[0] - origin)
                # axis_2nd = unit_vector(np.cross(z_axis, axis_1st))
                axis_2nd = unit_vector(np.cross(axis_1st, z_axis))
            else:
                axis_1st = unit_vector(origin - trian[0])
                # axis_2nd = unit_vector(np.cross(axis_1st, z_axis))
                axis_2nd = unit_vector(np.cross(z_axis, axis_1st))

            if CAMERAS[cam2][1][1] == 'positive':
                alt_axis_2nd = trian[2] - origin
            else:
                alt_axis_2nd = origin - trian[2]

    axis_2nd_name = 'y-axis' if CAMERAS[cam1][0][0] == 'x-axis' else 'x-axis'
    stereo_cam_unit = {
        CAMERAS[cam1][0][0]: axis_1st,
        (axis_2nd_name, 'cross'): axis_2nd,
        (axis_2nd_name, 'alt'): alt_axis_2nd,
        'z-axis': z_axis
    }

    orig_map = {
        'origin': origin,
        'map': np.array((axis_1st, axis_2nd, z_axis)).T
    }

    print('\nCross product derived {axis_2nd_name}: {cross}\n' \
    'Origin-subtraction derived {axis_2nd_name}: {orig}\n' \
    'Angle between the two vectors: {angle}\n'.format(
        axis_2nd_name=axis_2nd_name, cross=axis_2nd,
        orig=alt_axis_2nd, angle=vg.angle(axis_2nd, alt_axis_2nd)
    ))

    return stereo_cam_unit, orig_map


def generate_linear_map(config_path, pixel_tolerance=2, paralell=False):
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
    dlc3d_cfgs = get_dlc3d_configs(config_path)
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

    camera_names = tuple(CAMERAS.keys())


    pairs = tuple(dlc3d_cfgs.keys())
    orig_maps, stereo_cam_units = {}, {}
    cpu_cores = cpu_count(logical=False)
    if paralell is False or cpu_cores < 2:
        for pair in pairs:
            orig_maps[pair], stereo_cam_units[pair] = stereo_cam_info(config_path, axis_vectors, pair)

    else:
        workers = 4 if cpu_cores < 8 else 8
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_info = {executor.submit(stereo_cam_info, pair): pair for pair in pairs}
            for future in concurrent.futures.as_completed(future_to_info):
                pair = future_to_info[future]
                try:
                    orig_maps[pair], stereo_cam_units[pair] = future.result()
                except Exception as exc:
                    print('%s generated an exception: %s' % (pair, exc))

    with open(basis_result_path, 'wb') as outfile:
        pickle.dump((orig_maps, stereo_cam_units), outfile)
        print('Saved linear map to:\n{}'.format(basis_result_path))

    pd.DataFrame.from_dict(stereo_cam_units).to_excel(dataframe_path)
    print('Saved excel file containing the computed basis vectors to:\n{}'.format(dataframe_path))

    print('Returning dictionary containing the computed linear maps')
    return orig_maps, stereo_cam_units


def change_basis_experiment_coords(pair_roi_df, orig_maps):
    coords = {}
    for pair, roi_df in pair_roi_df.items():
        origin, linear_map = orig_maps[pair]['origin'], orig_maps[pair]['map']
        for roi, df in roi_df.items():
            x, y, z = change_basis_func(df, linear_map, origin).T
            coords[(roi, pair, 'x')] = pd.Series(x)
            coords[(roi, pair, 'y')] = pd.Series(y)
            coords[(roi, pair, 'z')] = pd.Series(z)
    return pd.DataFrame.from_dict(coords, orient='columns').sort_index(axis=1, level=0)


def map_coords(config_path, suffix='_DLC_3D.h5', paralell=False):
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
        print('According to the DeepCage triangulated coordinates detection algorithm this project is not ready for changing basis')
        return False

    cfg = read_config(config_path)
    data_path = os.path.realpath(cfg['data_path'])
    result_path = cfg['results_path']

    dlc3d_cfgs = get_dlc3d_configs(config_path)

    if len(glob(os.path.join(result_path, '*.xlsx'))) or len(glob(os.path.join(result_path, '*.h5'))):
        msg = 'The result folder needs to be empty. Path: {}'.format(result_path)
        raise ValueError(msg)

    basis_result_path = os.path.join(data_path, 'cb_result.pickle')
    try:
        with open(basis_result_path, 'rb') as infile:
            stereo_cam_units, orig_maps = pickle.load(infile)
    except FileNotFoundError:
        msg = 'Could not detect results from deepcage.compute.generate_linear_map() in:\n%s' % basis_result_path
        raise FileNotFoundError(msg)

    dfs = {}
    cpu_cores = cpu_count(logical=False)
    if paralell is False or cpu_cores < 2:
        for info, pair_roi_df in coords.items():
            animal, trial, date = info
            dfs[(animal, trial, date)] = change_basis_experiment_coords(pair_roi_df, orig_maps)

    else:
        submissions = {}
        workers = 4 if cpu_cores < 8 else 8
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            for info, pair_roi_df in coords.items():
                # info = (animal, trial, date)
                submissions[executor.submit(change_basis_experiment_coords, pair_roi_df, orig_maps)] = info

            for future in submissions:
                info = submissions[future]
                if info not in dfs:
                    dfs[info] = {}
                try:
                    dfs[info][(roi, pair)] = future.result()
                except Exception as exc:
                    print('%s generated an exception: %s' % (submissions[future], exc))

    print('Attempting to save new coordinates to result folder:\n%s' % result_path)
    for info, df in dfs.items():
        file_path = os.path.join(result_path, 'mapped_%s_%s_%s' % info)

        df.to_hdf(file_path+'.h5', key='a%st%sd%s' % info)
        df.to_csv(file_path+'.csv')
        df.to_excel(file_path+'.xlsx')

        print('The mapped coordinates of %s saved to\n%s\n' % (info, file_path))

    print('Done')
    return True


def cage_calibrate_triangulated(config_path):
    cfg = read_config(config_path)
    results_path = Path(cfg['results_path'])
    triangulated = results_path / 'triangulated'
    if not os.path.exists(triangulated):
        msg = 'Could detect triangulated coords in %s' % triangulated
        raise ValueError(msg)

    experiments = glob(triangulated / '*/')
    for exp in experiments:
        map_coords(config_path)
