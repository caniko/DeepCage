import matplotlib.pyplot as plt
import numpy as np
import vg

import pickle
import os

from deepcage.project.edit import read_config

from .utils import rad_to_deg


def visualize_basis_vectors(config_path, lines=True):
    cfg = read_config(config_path)
    data_path = os.path.realpath(cfg['data_path'])

    basis_result_path = os.path.join(data_path, 'cb_result.pickle')
    try:
        with open(basis_result_path, 'rb') as infile:
            stereo_cam_units, orig_maps = pickle.load(infile)
    except FileNotFoundError:
        msg = 'Could not detect results from deepcage.compute.generate_linear_map() in:\n%s' % basis_result_path
        raise FileNotFoundError(msg)

    figure_dir = os.path.join(data_path, 'linear_map_visual')
    if not os.path.exists(figure_dir):
        os.mkdir(figure_dir)

    n = np.linspace(0, 5, 100)
    for pair, info in orig_maps.items():
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        if lines is True:
            r = []
            for axis in info['map'].T:
                r.append(n * axis[np.newaxis, :].T)

            r_1, r_2, r_3 = r
            ax.plot(*r_1, label='r_1')
            ax.plot(*r_2, label='r_2')
            ax.plot(*r_3, label='z')
        else:
            ax.plot()
        
        # angles
        i, ii, iii = info['map'].T
        i_ii = vg.angle(i, ii)
        i_iii = vg.angle(i, iii)
        ii_iii = vg.angle(ii, iii)

        title_text = '%s %s\nr1-r2: %3f r1-r3: %3f\nr2-r3: %3f' % (pair[0], pair[1], i_ii, i_iii, ii_iii)
        ax.set_title(title_text).set_y(1.005)
        ax.legend()

        plt.savefig(os.path.join(figure_dir, '%s_%s.png' % pair))
