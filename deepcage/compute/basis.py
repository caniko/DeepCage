from numpy.linalg import solve, norm
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
from deepcage.auxiliary.constants import CAMERAS, get_pairs, cage_order_pairs
from deepcage.project.get import get_dlc3d_configs, get_labels, get_paired_labels
from deepcage.project.edit import read_config

from .triangulate import triangulate_raw_2d_camera_coords, triangulate_basis_labels
from .utils import unit_vector, change_basis_func

# TODO: Create a jupyter notebook with an implementation of this workflow


def compute_basis_vectors(trian, pair, use_cross=True, normalize=True, decrement=False):
    cam1, cam2 = pair
    if decrement is True:
        if CAMERAS[cam1][2] == CAMERAS[cam2][2]:
            if CAMERAS[cam1][0][1] == 'left':
                origin = trian[0] + (trian[1] - trian[0]) / 2    # pos + (trian[1] - pos) / 2
                z_axis = trian[3] - origin
                axis_1st = trian[0] - origin
                axis_2nd = np.cross(axis_1st, z_axis)
            else:
                origin = trian[1] + (trian[0] - trian[1]) / 2
                z_axis = trian[3] - origin
                axis_1st = origin - trian[1]
                axis_2nd = np.cross(z_axis, axis_1st)

            if CAMERAS[cam2][1][1] == 'positive':
                alt_axis_2nd = origin - trian[2]
            else:
                alt_axis_2nd = trian[2] - origin

        else:
            # Corner
            first_axis_linev = trian[1] - trian[0]
            z_axis_linev = trian[5] - trian[4]
            tangent_v = np.cross(z_axis_linev, first_axis_linev)

            LHS = np.array((first_axis_linev, -z_axis_linev, tangent_v)).T
            RHS = trian[4] - trian[0]
            sol = solve(LHS, RHS)

            origin = trian[0] + first_axis_linev * sol[0] + tangent_v * sol[2]/2
            z_axis = trian[4] - origin

            if CAMERAS[cam1][1][1] == 'positive':
                axis_1st = trian[0] - origin
                axis_2nd = np.cross(axis_1st, z_axis)
            else:
                axis_1st = origin - trian[0]
                axis_2nd = np.cross(z_axis, axis_1st)

            if CAMERAS[cam2][1][1] == 'positive':
                alt_axis_2nd = trian[2] - origin
            else:
                alt_axis_2nd = origin - trian[2]
    else:
        if CAMERAS[cam1][2] == CAMERAS[cam2][2]:
            if CAMERAS[cam1][0][1] == 'left':
                origin = trian[0] + (trian[1] - trian[0]) / 2    # pos + (trian[1] - pos) / 2
                z_axis = trian[3] - origin
                axis_1st = trian[0] - origin
                axis_2nd = np.cross(axis_1st, z_axis)
            else:
                origin = trian[1] + (trian[0] - trian[1]) / 2
                z_axis = trian[3] - origin
                axis_1st = origin - trian[1]
                axis_2nd = np.cross(z_axis, axis_1st)

            if CAMERAS[cam2][1][1] == 'positive':
                alt_axis_2nd = origin - trian[2]
            else:
                alt_axis_2nd = trian[2] - origin
        else:
            origin = trian[-1]
            z_axis = trian[2] - origin
            if CAMERAS[cam1][1][1] == 'positive':
                axis_1st = trian[0] - origin
                axis_2nd = np.cross(axis_1st, z_axis)
            else:
                axis_1st = origin - trian[0]
                axis_2nd = np.cross(z_axis, axis_1st)

            if CAMERAS[cam2][1][1] == 'positive':
                alt_axis_2nd = trian[2] - origin
            else:
                alt_axis_2nd = origin - trian[2]
    
    # Calculate axis_len before potential normalization
    axis_len = np.mean((norm(axis_1st), norm(alt_axis_2nd), norm(z_axis)))

    if normalize is True:
        axis_1st = unit_vector(axis_1st)
        axis_2nd = unit_vector(axis_2nd)
        alt_axis_2nd = unit_vector(alt_axis_2nd)
        z_axis = unit_vector(z_axis)
    else:
        axis_2nd = unit_vector(axis_2nd) * np.average((norm(z_axis), norm(axis_1st)))

    axis_2nd_name = 'y-axis' if CAMERAS[cam1][0][0] == 'x-axis' else 'x-axis'
    if use_cross is True:
        stereo_cam_unit = {
            CAMERAS[cam1][0][0]: axis_1st,
            axis_2nd_name: axis_2nd,
            'z-axis': z_axis,
            (axis_2nd_name, 'origin_subtract'): alt_axis_2nd
        }
    else:
        stereo_cam_unit = {
            CAMERAS[cam1][0][0]: axis_1st,
            axis_2nd_name: alt_axis_2nd,
            'z-axis': z_axis,
            (axis_2nd_name, 'cross'): axis_2nd
        }

    orig_map = {
        'axis_len': axis_len,
        'origin': origin,
        'map': np.array((
            stereo_cam_unit['x-axis'],
            stereo_cam_unit['y-axis'],
            z_axis
        )).T
    }

    if normalize is False:
        print('Vector magnitude/np.linalg.norm')
        for axis in ('x-axis', 'y-axis', 'z-axis'):
            print(f'{axis}: {np.linalg.norm(stereo_cam_unit[axis])}')

    print(
        f'\nCross product derived {axis_2nd_name}: {axis_2nd}\n' \
        f'Origin-subtraction derived {axis_2nd_name}: {alt_axis_2nd}\n' \
        f'Angle between the two vectors: {vg.angle(axis_2nd, alt_axis_2nd)}\n'
    )

    return stereo_cam_unit, orig_map


def create_stereo_cam_origmap(config_path, undistort=True, decrement=False, save=True, **cbv_kwargs):
    '''
    Parameters
    ----------
    config_path : string
        Absolute path of the project config.yaml file.
    cbv_kwargs : dictionary
        Keyword arguments for compute_basis_vectors()
    '''
    cfg = read_config(config_path)
    dlc3d_cfgs = get_dlc3d_configs(config_path)
    data_path = os.path.realpath(cfg['data_path'])

    if save is True:
        basis_result_path = os.path.join(data_path, 'cb_result.pickle')
        dataframe_path = os.path.join(data_path, 'basis_vectors.xlsx')
        if os.path.exists(basis_result_path) and os.path.exists(dataframe_path):
            msg = f'Please remove old analysis files before proceeding. File paths:\n{basis_result_path}\n{dataframe_path}'
        elif os.path.exists(basis_result_path):
            msg = f'Please remove old analysis file before proceeding. File paths:\n{basis_result_path}'
        elif os.path.exists(dataframe_path):
            msg = f'Please remove old analysis file before proceeding. File paths:\n{dataframe_path}\n'

    pairs = tuple(dlc3d_cfgs.keys())
    stereo_cam_units, orig_maps = {}, {}
    for pair in pairs:
        cam1, cam2 = pair
        dlc3d_cfg = dlc3d_cfgs[pair]
    
        # Get triangulated points for computing basis vectors
        basis_labels = get_paired_labels(config_path, pair)['decrement' if decrement is True else 'normal']
        trian = triangulate_basis_labels(dlc3d_cfg, basis_labels, pair, undistort=undistort, decrement=decrement)

        stereo_cam_units[pair], orig_maps[pair] = compute_basis_vectors(trian, pair, decrement=decrement, **cbv_kwargs)

    if save is True:
        with open(basis_result_path, 'wb') as outfile:
            pickle.dump((stereo_cam_units, orig_maps), outfile)
            print('Saved linear map to:\n{}'.format(basis_result_path))

        pd.DataFrame.from_dict(stereo_cam_units).to_excel(dataframe_path)
        print('Saved excel file containing the computed basis vectors to:\n{}'.format(dataframe_path))

    print('Returning dictionary containing the computed linear maps')

    return stereo_cam_units, orig_maps


def map_experiment(
        config_path, undistort=True, percentiles=(5, 95), use_saved_origmap=True, normalize=True,
        suffix='_DLC_3D.h5', bonvideos=False, save=True, paralell=False, **cbv_kwargs
    ):
    '''
    This function changes the basis of deeplabcut-triangulated that are 3D.

    Parameters
    ----------
    config_path : string
        Absolute path of the project config.yaml file.
    linear_maps : {string: numpy.array}
        (3, 3) array that stores the linear map for changing basis
    suffix : string
        The suffix in the DeepLabCut 3D project triangualtion result storage files
    cbv_kwargs : dictionary
        Keyword arguments for compute_basis_vectors()

    Example
    -------

    '''
    coords = detect_triangulation_result(
        config_path, undistorted=undistort, suffix=suffix, change_basis=True, bonvideos=bonvideos
    )
    if coords is False:
        print('According to the DeepCage triangulated coordinates detection algorithm this project is not ready for changing basis')
        return False

    cfg = read_config(config_path)
    data_path = os.path.realpath(cfg['data_path'])
    result_path = cfg['results_path']

    dlc3d_cfgs = get_dlc3d_configs(config_path)

    if use_saved_origmap is True:
        basis_result_path = os.path.join(data_path, 'cb_result.pickle')
        try:
            with open(basis_result_path, 'rb') as infile:
                stereo_cam_units, orig_maps = pickle.load(infile)
        except FileNotFoundError:
            msg = 'Could not detect results from deepcage.compute.generate_linear_map() in:\n' \
                  + basis_result_path
            raise FileNotFoundError(msg)
    else:
        stereo_cam_units, orig_maps = create_stereo_cam_origmap(
            config_path, undistort=undistort, decrement=False, save=False, normalize=normalize, **cbv_kwargs
        )

    dfs = {}
    cpu_cores = cpu_count(logical=False)
    if paralell is False or cpu_cores < 2:
        for info, pair_roi_df in coords.items():
            # info = (animal, trial, date)
            dfs[info] = map_coords(pair_roi_df, orig_maps, percentiles)

    else:
        submissions = {}
        workers = 4 if cpu_cores < 8 else 8
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            for info, pair_roi_df in coords.items():
                # info = (animal, trial, date)
                submissions[executor.submit(map_coords, pair_roi_df, orig_maps)] = info

            for future in submissions:
                info = submissions[future]
                if info not in dfs:
                    dfs[info] = {}
                try:
                    dfs[info][(roi, pair)] = future.result()
                except Exception as exc:
                    print('%s generated an exception: %s' % (submissions[future], exc))

    if save is True:
        # print('Attempting to save new coordinates to result folder:\n%s' % result_path)
        for info, df in dfs.items():
            df_name = 'mapped'
            for i in info.split('_'):
                df_name += '_'+i

            file_path = os.path.join(result_path, df_name)

            df.to_hdf(file_path+'.h5', key=df_name if bonvideos is False else 'a%st%sd%s' % info)
            df.to_csv(file_path+'.csv')
            df.to_excel(file_path+'.xlsx')

            print('The mapped coordinates of %s have been saved to\n%s\n' % (info, file_path))

    print('DONE: Basis changed')
    return dfs


def map_coords(pair_roi_df, orig_maps, percentiles=None):
    '''
    Helper function for map_experiment()
    '''

    pairs = tuple(pair_roi_df.keys())
    pair_order = cage_order_pairs(pairs)

    columns = []
    pre_df = {}
    for pair in pair_order:
        roi_df = pair_roi_df[pair]
        for roi, coords in roi_df.items():
            x, y, z = change_basis_func(
                coords,
                orig_maps[pair]['map'],
                orig_maps[pair]['origin'],
                orig_maps[pair]['axis_len']
            ).T

            pre_df[(roi, pair, 'x')] = pd.Series(x)
            pre_df[(roi, pair, 'y')] = pd.Series(y)
            pre_df[(roi, pair, 'z')] = pd.Series(z)
            columns.extend(( (roi, pair, 'x'), (roi, pair, 'y'), (roi, pair, 'z') ))
    df = pd.DataFrame.from_dict(pre_df, orient='columns').sort_index(axis=1, level=0)
    return df.loc[np.logical_not(np.all(np.isnan(df.values), axis=1))]
