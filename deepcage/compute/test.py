import matplotlib.pyplot as plt
import numpy as np
import vg

import pickle
import os

from deepcage.project.edit import read_config, get_dlc3d_configs
from deepcage.project.get import get_labels
from deepcage.auxiliary.constants import CAMERAS, PAIR_IDXS

from .triangulate import triangulate_basis_labels
from .basis import compute_basis_vectors
from .utils import rad_to_deg


def visualize_basis_vectors(config_path, decrement=False):
    cfg = read_config(config_path)
    data_path = os.path.realpath(cfg['data_path'])

    dlc3d_cfgs = get_dlc3d_configs(config_path)
    basis_labels = get_labels(config_path)

    figure_dir = os.path.join(data_path, 'test figures')
    if not os.path.exists(figure_dir):
        os.mkdir(figure_dir)

    n = np.linspace(-1, 5, 100)
    pairs = tuple(dlc3d_cfgs.keys())
    for pair in pairs:
        # Get pair info
        cam1, cam2 = pair
        raw_cam1v = basis_labels[cam1]
        raw_cam2v = basis_labels[cam2]

        # Create figure skeleton
        fig = plt.figure(figsize=(12, 10))
        ax1 = fig.add_subplot(221, projection='3d')
        ax2 = fig.add_subplot(222, projection='3d')
        ax3 = fig.add_subplot(223)  # cam1
        ax4 = fig.add_subplot(224)  # cam2
        
        # Prepare camera plot labels
        if decrement is False:
            cam_labels = {
                cam1: {
                    ('%s positive' % CAMERAS[cam1][0][0]): raw_cam1v[0]['positive'],
                    ('%s negative' % CAMERAS[cam1][0][0]): raw_cam1v[0]['negative'],
                    CAMERAS[cam1][1]: raw_cam1v[1],
                    'z-axis': raw_cam1v[2],
                    'origin': raw_cam1v[3]
                },
                cam2: {
                    ('%s positive' % CAMERAS[cam2][0][0]): raw_cam2v[0]['positive'],
                    ('%s negative' % CAMERAS[cam2][0][0]): raw_cam2v[0]['negative'],
                    CAMERAS[cam2][1]: raw_cam2v[1],
                    'z-axis': raw_cam2v[2],
                    'origin': raw_cam2v[3]
                }
            }

        else:
            cam_labels = {
                cam1: {
                    ('%s positive apex' % CAMERAS[cam1][0][0]): raw_cam1v[0]['positive'][0],
                    ('%s positive decrement' % CAMERAS[cam1][0][0]): raw_cam1v[0]['positive'][1],
                    ('%s negative' % CAMERAS[cam1][0][0]): raw_cam1v[0]['negative'][0],
                    ('%s negative decrement' % CAMERAS[cam1][0][0]): raw_cam1v[0]['negative'][1],
                    ('%s apex' % CAMERAS[cam1][1]): raw_cam1v[1][0],
                    ('%s decrement' % CAMERAS[cam1][1]): raw_cam1v[1][1],
                    'z-axis apex': raw_cam1v[2][0],
                    'z-axis decrement': raw_cam1v[2][1]
                },
                cam2: {
                    ('%s positive apex' % CAMERAS[cam2][0][0]): raw_cam2v[0]['positive'][0],
                    ('%s positive decrement' % CAMERAS[cam2][0][0]): raw_cam2v[0]['positive'][1],
                    ('%s negative' % CAMERAS[cam2][0][0]): raw_cam2v[0]['negative'][0],
                    ('%s negative decrement' % CAMERAS[cam2][0][0]): raw_cam2v[0]['negative'][1],
                    ('%s apex' % CAMERAS[cam2][1]): raw_cam2v[1][0],
                    ('%s decrement' % CAMERAS[cam2][1]): raw_cam2v[1][1],
                    'z-axis apex': raw_cam2v[2][0],
                    'z-axis decrement': raw_cam2v[2][1]
                }
            }
        # Plot manually created labels
        for cam, cax in zip(pair, (ax3, ax4)):
            colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(cam_labels[cam]))))
            for label, coord in cam_labels[cam].items():
                cax.set_title('%s labels' % cam).set_y(1.005)
                cax.scatter(*coord, c=next(colors), label=label)
                cax.legend()

        # Triangulate the two sets of labels, and map them to 3D
        dlc3d_cfg = dlc3d_cfgs[pair]
        trian_dict, trian = triangulate_basis_labels(
            dlc3d_cfg, basis_labels, pair, decrement=decrement, keys=True
        )
        colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(trian_dict))))
        for (label, coord), color in zip(trian_dict.items(), colors):
            x, y, z = coord - trian_dict['origin']
            ax1.scatter(x, y, z, c=color, label=label)

        ax1.set_title('Triangualted').set_y(1.005)
        ax1.legend()

        _, orig_map = compute_basis_vectors(trian, pair, decrement=decrement)

        r = []
        for axis in orig_map['map'].T:
            r.append(n * axis[np.newaxis, :].T)

        r_1, r_2, r_3 = r
        ax2.plot(*r_1, label='r1')
        ax2.plot(*r_2, label='r2')
        ax2.plot(*r_3, label='r3/z')
        
        # angles
        i, ii, iii = orig_map['map'].T
        i_ii = vg.angle(i, ii)
        i_iii = vg.angle(i, iii)
        ii_iii = vg.angle(ii, iii)

        title_text2 = 'r1-r2: %3f r1-r3: %3f\nr2-r3: %3f' % (i_ii, i_iii, ii_iii)
        ax2.set_title(title_text2).set_y(1.005)
        ax2.legend()

        # fig.subplots_adjust(wspace=0.5)
        fig.savefig(os.path.join(figure_dir, '%d_%s_%s.png' % (PAIR_IDXS[pair], *pair)))
