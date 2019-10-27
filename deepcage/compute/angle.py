from numpy.linalg import norm
import numpy as np

from .utils import *


def non_sessile_angle_change(rp, p):
    """
    Parameters
    ----------
    p1 : numpy.array-like
    p2 : numpy.array-like
    """

    rp_smaller_idxs = np.where(norm(rb) < norm(p))
    line_vector_smaller = p[rp_smaller_idxs] - rp[rp_smaller_idxs]

    rp_larger_idxs = np.where(norm(rb) > norm(p))
    line_vector_larger = rp[rp_larger_idxs] - p[rp_larger_idxs]

    line_vectors = np.zeros((len(rb), 3))
    line_vectors[rp_smaller_idxs] = line_vector_smaller
    line_vectors[rp_larger_idxs] = line_vector_larger
    line_units = unit(line_vectors)

    baseline_unit_vector = np.median(line_units, axis=0)
    # baseline_uv_idx = line_units.index(np.percentile(line_units, 50, interpolation='nearest'))
    
