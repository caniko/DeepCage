import matplotlib.pyplot as plt
import numpy as np
import vg

import pickle
import os

from deepcage.project.edit import read_config, get_dlc3d_configs
from deepcage.project.get import get_labels, get_paired_labels
from deepcage.auxiliary.constants import CAMERAS, PAIR_IDXS

from .triangulate import triangulate_basis_labels, triangulate_raw_2d_camera_coords
from .basis import compute_basis_vectors
from .utils import rad_to_deg


def visualize_basis_vectors(config_path, decrement=False):
    dlc3d_cfgs = get_dlc3d_configs(config_path)
    basis_labels = get_labels(config_path)

    cfg = read_config(config_path)
    test_dir = os.path.join(cfg['data_path'], 'test')
    figure_dir = os.path.join(test_dir, 'visualize_basis_vectors')
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

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
        cam_labels = get_paired_labels(config_path, pair)['decrement' if decrement is True else 'normal']

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


def visualize_triangulation(config_path, decrement=False):
    dlc3d_cfgs = get_dlc3d_configs(config_path)
    basis_labels = get_labels(config_path)

    cfg = read_config(config_path)
    test_dir = os.path.join(cfg['data_path'], 'test')
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    fig = plt.figure(figsize=(12, 10))

    # Get non-corner pairs by splicing
    pairs = tuple(PAIR_IDXS.keys())[::2]
    for i, pair in enumerate(pairs):
        dlc3d_cfg = dlc3d_cfgs[pair]
        cam1, cam2 = pair

        # Prepare camera plot labels
        cam_labels = get_paired_labels(config_path, pair)['decrement' if decrement is True else 'normal']

        # Triangulate the two sets of labels, and map them to 3D
        raw_cam1v = basis_labels[cam1]
        raw_cam2v = basis_labels[cam2]
        trian_dict, _ = triangulate_raw_2d_camera_coords(
            dlc3d_cfg,
            cam1_coords=(*raw_cam1v[0].values(), *raw_cam1v[1:]),
            cam2_coords=(*raw_cam2v[0].values(), *raw_cam2v[1:]),
            keys=cam_labels[cam1]
        )

        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        for label, coord in trian_dict.items():
            ax.scatter(*coord, label=label)
        ax.legend()
        ax.set_title('%s %s' % pair).set_y(1.005)

    fig.suptitle('Triangulation visualization')
    fig.savefig(os.path.join(test_dir, 'visualize_triangulation.png'))
