import matplotlib.pyplot as plt
import numpy as np

import pickle
from glob import glob
import os

from deepcage.auxiliary.constants import CAMERAS
from deepcage.project.edit import read_config


""" Label """
def OE_basis_label(config_path, detect=True, name_pos=0, format='png', image_paths=None):
    '''
    See deepcage.auxiliary.detect.basis_label()
    '''
    import matplotlib.pyplot as plt
    import gc

    from deepcage.auxiliary.detect import detect_cage_calibration_images
    from deepcage.auxiliary.gui import get_title, get_coord


    if image_paths is None:
        camera_images = detect_cage_calibration_images(config_path, name_pos=name_pos)

    n = -1
    basis_labels = dict.fromkeys(camera_images.keys() if detect is True else CAMERAS.keys())
    for camera in basis_labels.keys():
        cam_img = camera_images[camera]

        basis_labels[camera] = (
            get_coord(cam_img, n=n, title=get_title(camera, 'x-axis', 'positive', True)),
            get_coord(cam_img, n=n, title=get_title(camera, 'y-axis', 'negative', True)),
            get_coord(cam_img, n=n, title=get_title(camera, 'z-axis', 'positive', True)),
            get_coord(cam_img, n=n, title='Select origin')
        )

        plt.close()
        gc.collect()

    data_path = os.path.join(read_config(config_path)['data_path'], 'labels.pickle')
    with open(data_path, 'wb') as outfile:
        stereo_file = pickle.dump(basis_labels, outfile)

    return basis_labels


def OE_get_paired_labels(config_path, pair):
    from deepcage.project.get import get_dlc3d_configs, get_labels

    cam1, cam2 = pair

    labels = get_labels(config_path)
    raw_cam1v, raw_cam2v = labels[cam1], labels[cam2]
    return {
        cam1: {
            'x-axis': labels[cam1][0],
            'negative y-axis': labels[cam1][1],
            'z-axis': labels[cam1][2],
            'origin': labels[cam1][3]
        },
        cam2: {
            'x-axis': labels[cam2][0],
            'negative y-axis': labels[cam2][1],
            'z-axis': labels[cam2][2],
            'origin': labels[cam2][3]
        }
    }


def OE_get_triangulated_basis(config_path, pair, undistort=True, keys=False):
    from deepcage.compute.triangulate import triangulate_raw_2d_camera_coords
    from deepcage.project.get import get_dlc3d_configs, get_labels

    cam1, cam2 = pair
    labels = get_labels(config_path)
    basis_labels = OE_get_paired_labels(config_path, pair)
    cam1_labels, cam2_labels = basis_labels.values()

    return triangulate_raw_2d_camera_coords(
        get_dlc3d_configs(config_path)[pair],
        cam1_coords=tuple(cam1_labels.values()),
        cam2_coords=tuple(cam2_labels.values()),
        undistort=undistort,
        keys=None if keys is False else tuple(cam1_labels.keys())
    )


""" Basis """
def OE_compute_basis_vectors(trian, pair, use_cross=False, normalize=True, decrement=False):
    '''
    See deepcage.compute.basis.compute_basis_vectors()
    
    Appropriate usecase deepcage.compute.basis.map_experiment(basis_computer=compute_basis_vectors)
    '''
    from numpy.linalg import norm
    from deepcage.compute.utils import unit_vector

    origin = trian[3]
    x_axis = trian[0] - origin
    y_axis = origin - trian[1]
    z_axis = trian[2] - origin

    axis_len = np.array( (norm(x_axis), norm(y_axis), norm(z_axis)) )
    if normalize is True:
        x_axis = unit_vector(x_axis)
        y_axis = unit_vector(y_axis)
        z_axis = unit_vector(z_axis)

    stereo_cam_unit = {'x_axis': x_axis, 'y_axis': y_axis, 'z-axis': z_axis}
    
    orig_map = {
        'axis_len': axis_len,
        'origin': origin,
        'map': np.array((x_axis, y_axis, z_axis)).T
    }

    return stereo_cam_unit, orig_map


def OE_map_experiment(
    config_path, percentiles=(5, 95), normalize=True, suffix='_DLC_3D.h5',
    bonvideos=False, save=True, paralell=False, undistort=True
):
    from deepcage.compute.basis import map_experiment

    return map_experiment(
        config_path, undistort=undistort, percentiles=percentiles, normalize=normalize,
        suffix=suffix, bonvideos=bonvideos, save=save, paralell=paralell,
        use_saved_origmap=False, basis_computer=OE_compute_basis_vectors,
        labels_getter=OE_get_triangulated_basis, labels_are2d=False
    )


""" Test """
def OE_visualize_workflow(config_path, undistort=True, normalize=True, save=True):
    '''
    Parameters
    ----------
    config_path : string
        Absolute path of the project config.yaml file.
    '''
    from deepcage.project.get import get_labels, get_paired_labels, get_dlc3d_configs
    import matplotlib.image as image
    import vg

    dlc3d_cfgs = get_dlc3d_configs(config_path)
    basis_labels = get_labels(config_path)

    cfg = read_config(config_path)
    test_dir = os.path.join(cfg['data_path'], 'test')
    figure_dir = os.path.join(test_dir, 'visualize_workflow')
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    n = np.linspace(-1, 5, 100)
    pairs = tuple(dlc3d_cfgs.keys())
    for pair in pairs:
        cam1, cam2 = pair

        fig = plt.figure(figsize=(12, 10))
        ax = {
            'trian': fig.add_subplot(221, projection='3d'),
            'basis': fig.add_subplot(222, projection='3d'),
            'cam1': fig.add_subplot(223),
            'cam2': fig.add_subplot(224)
        }
        
        # Get camera plot labels
        cam_labels = OE_get_paired_labels(config_path, pair)

        # Plot manual labels
        for cam, cax in zip(pair, (ax['cam1'], ax['cam2'])):
            # Add respective calibration image as background
            img_cam = image.imread(glob(os.path.join(cfg['calibration_path'], '*'+cam+'*'))[0])
            cax.imshow(img_cam)

            cmap = plt.cm.rainbow(np.linspace( 0, 1, len(cam_labels[cam]) ))
            for (label, coord), color in zip(cam_labels[cam].items(), cmap):
                cax.set_title((cam, 'labels')).set_y(1.005)
                cax.scatter(*coord, c=color, label=label)
                cax.legend()

        # Triangulate the two sets of labels, and map them to 3D
        dlc3d_cfg = dlc3d_cfgs[pair]
        trian_dict, trian = OE_get_triangulated_basis(config_path, pair, undistort=True, keys=True)

        cmap = iter(plt.cm.rainbow(np.linspace( 0, 1, len(trian_dict)) ))
        for (label, coord), color in zip(trian_dict.items(), cmap):
            ax['trian'].scatter(*(coord - trian_dict['origin']), c=color, label=label)

        ax['trian'].set_title('Triangualted').set_y(1.005)
        ax['trian'].legend()

        ax['trian'].set_xlabel('X', fontsize=10)
        ax['trian'].set_ylabel('Y', fontsize=10)
        ax['trian'].set_zlabel('Z', fontsize=10)

        _, orig_map = OE_compute_basis_vectors(trian, pair, normalize=normalize)

        r = []
        for axis in orig_map['map'].T:
            r.append(n * axis[np.newaxis, :].T)

        r_1, r_2, r_3 = r
        ax['basis'].plot(*r_1, label='r1/x')
        ax['basis'].plot(*r_2, label='r2/y')
        ax['basis'].plot(*r_3, label='r3/z')

        # Angles
        i, ii, iii = orig_map['map'].T
        i_ii = vg.angle(i, ii)
        i_iii = vg.angle(i, iii)
        ii_iii = vg.angle(ii, iii)

        title_text2 = 'r1-r2: %3f r1-r3: %3f\nr2-r3: %3f' % (i_ii, i_iii, ii_iii)

        ax['basis'].set_title(title_text2).set_y(1.005)
        ax['basis'].legend()

        ax['basis'].set_xticklabels([])
        ax['basis'].set_yticklabels([])
        ax['basis'].set_zticklabels([])
        ax['basis'].set_xlabel('X', fontsize=10)
        ax['basis'].set_ylabel('Y', fontsize=10)
        ax['basis'].set_zlabel('Z', fontsize=10)

        if save is True:
            fig.savefig( os.path.join(figure_dir, '%s_%s.png' % pair) )

    return fig, ax


def OE_visualize_basis_vectors():
    from deepcage.compute.test import visualize_basis_vectors


def OE_visualize_basis_vectors_single():
    from deepcage.compute.test import visualize_basis_vectors_single


def OE_plot_3d_trajectories(
    config_path, cm_is_real_idx=True, cols=2,
    normalize=True, save=True, **me_kwargs
):
    from deepcage.compute.test import plot_3d_trajectories

    dfs = OE_map_experiment(config_path, save=False, **me_kwargs)
    return plot_3d_trajectories(config_path, cm_is_real_idx=True, cols=cols, dfs=dfs, save=True)
