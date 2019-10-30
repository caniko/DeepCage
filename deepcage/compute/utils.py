from numpy.linalg import norm
import numpy as np
import ruamel.yaml
import pickle
import os

from deppcage.auxiliary import CAMERAS, read_config, detect_images


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


def remove_close_zero(vector, tol=1e-16):
    ''' Returns the vector where values under the tolerance is set to 0 '''
    vector[np.abs(vector) < tol] = 0
    return vector


def clockwise_angle(vector_1, vector_2, is_unit=False):
    '''
    Returns the inner_angle in radians between vectors 'vector_1' and 'vector_2'

    Example
    -------
    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    '''

    if is_unit is False:
        vector_1_u = np.apply_along_axis(unit_vector, 1, vector_1)
        vector_2_u = np.apply_along_axis(unit_vector, 1, vector_2)
    else:
        vector_1_u, vector_2_u = vector_1, vector_2
    
    inner_angle = np.arccos(np.clip(np.einsum('ij,ij->i', vector_1_u, vector_2_u), -1.0, 1.0))

    # Find determinant
    determinants = np.zeros((len(vector_1_u), 1))
    for i in range(len(vector_1_u)):
        determinants[i] = np.linalg.det(np.vstack((vector_1_u[i], vector_2_u[i])))

    clockwise_angle = np.where(determinants.T >= 0, inner_angle, 2*np.pi - inner_angle)[0]

    # The degree range is [pi, -pi]
    clockwise_angle -= np.pi

    return clockwise_angle
