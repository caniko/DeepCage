from numpy.linalg import norm
import numpy as np
import vg


def non_sessile_angle_change(rp, p):
    """
    Parameters
    ----------
    p1 : numpy.array-like
    p2 : numpy.array-like
    """
    from .utils import unit_vector

    rp_smaller_idxs = np.where(norm(rp) < norm(p))
    line_vector_smaller = p[rp_smaller_idxs] - rp[rp_smaller_idxs]

    rp_larger_idxs = np.where(norm(rp) > norm(p))
    line_vector_larger = rp[rp_larger_idxs] - p[rp_larger_idxs]

    line_vectors = np.zeros((len(rp), 3))
    line_vectors[rp_smaller_idxs] = line_vector_smaller
    line_vectors[rp_larger_idxs] = line_vector_larger
    line_units = unit_vector(line_vectors)

    baseline_unit_vector = np.median(line_units, axis=0)
    # baseline_uv_idx = line_units.index(np.percentile(line_units, 50, interpolation='nearest'))
    return vg.angle(baseline_unit_vector, line_units, is_unit=True)


def compute_rigid_acceleration(*args, apex=None):
    from .utils import duovec_midpoint

    ar_lenght = len(args)
    if ar_lenght == 1:
        return np.diff(*args)
    elif ar_lenght == 2 and apex is not None:
        return np.diff(duovec_midpoint(duovec_midpoint(*args), apex))
    elif ar_lenght == 2:
        return np.diff(duovec_midpoint(*args))
