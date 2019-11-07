from numpy.linalg import norm
import numpy as np
import ruamel.yaml
import pickle
import os

from deepcage.auxiliary.constants import CAMERAS


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
