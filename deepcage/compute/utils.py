from numpy.linalg import norm
import pandas as pd
import numpy as np
import ruamel.yaml
import pickle
import os

from deepcage.auxiliary.constants import CAMERAS, cage_order_pairs


def change_basis_func(coord_matrix, linear_map, origin, axis_lenght, calibrator_lenght=4.7, percentiles=None):
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

    assert isinstance(calibrator_lenght, (int, float))
    assert isinstance(axis_lenght, (int, float))

    # subtract "new origin" -> change basis -> divide by mean axis_length
    return calibrator_lenght * (((coord_matrix - origin) @ linear_map) / axis_lenght)


def duovec_midpoint(v1, v2):
    # Find vector with highest magnitude
    v1sp, v2sp = norm(v1), norm(v2)

    if v1sp >= v2sp:
        return v1 + (v2 - v1) / 2
    else:
        return v2 + (v1 - v2) / 2
    

def triangulate(apex, b1, b2):
    bm = duovec_midpoint(b1, b2)
    return duovec_midpoint(bm, apex)

    
def unit_vector(vector):
    ''' Returns the unit vector of the vector. '''
    return vector / norm(vector)


def rad_to_deg(value):
    return (value * 180) / np.pi


def remove_close_zero(vector, tol=1e-16):
    ''' Returns the vector where values under the tolerance is set to 0 '''
    vector[np.abs(vector) < tol] = 0
    return vector


def equalise_3daxes(ax, coord_set):
    max_ = np.nanmax(coord_set)
    min_ = np.nanmin(coord_set)

    ax.set_xlim(min_, max_)
    ax.set_ylim(min_, max_)
    ax.set_zlim(min_, max_)

    return ax


def create_df_from_coords(pair_roi_df, orig_maps, remove_nans=False):
    pairs = tuple(pair_roi_df.keys())
    pair_order = cage_order_pairs(pairs)

    columns = []
    coords = {}
    for pair in pair_order:
        roi_df = pair_roi_df[pair]
        for roi, df in roi_df.items():
            x, y, z = df.T
            coords[(roi, pair, 'x')] = pd.Series(x)
            coords[(roi, pair, 'y')] = pd.Series(y)
            coords[(roi, pair, 'z')] = pd.Series(z)
            columns.extend(( (roi, pair, 'x'), (roi, pair, 'y'), (roi, pair, 'z') ))
    df = pd.DataFrame.from_dict(coords, orient='columns').sort_index(axis=1, level=0)

    return df if remove_nans is False else df.loc[np.logical_not(np.all(np.isnan(df.values), axis=1))]
