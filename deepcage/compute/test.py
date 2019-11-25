import matplotlib.pyplot as plt
from pandas import read_hdf
import numpy as np
import vg

from tqdm import tqdm
import cv2

import concurrent.futures
import subprocess
import pickle

from pathlib import Path
from glob import glob
import os

from deeplabcut.pose_estimation_3d.plotting3D import plot2D

from deepcage.project.get import get_labels, get_paired_labels, get_dlc3d_configs
from deepcage.project.edit import read_config
from deepcage.auxiliary.constants import CAMERAS, PAIR_IDXS, pair_cycler

from .triangulate import triangulate_basis_labels, triangulate_raw_2d_camera_coords
from .basis import compute_basis_vectors, create_stereo_cam_origmap
from .utils import rad_to_deg, unit_vector


def visualize_workflow(config_path, decrement=False):
    '''
    Parameters
    ----------
    config_path : string
        String containing the full path of the project config.yaml file.
    '''
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
        # Get pair info
        cam1, cam2 = pair

        # Create figure skeleton
        fig = plt.figure(figsize=(12, 10))
        ax1 = fig.add_subplot(221, projection='3d')
        ax2 = fig.add_subplot(222, projection='3d')
        ax3 = fig.add_subplot(223)  # cam1
        ax4 = fig.add_subplot(224)  # cam2
        
        # Prepare camera plot labels
        cam_labels = get_paired_labels(config_path, pair)['decrement' if decrement is True else 'normal']

        # Plot manual labels
        for cam, cax in zip(pair, (ax3, ax4)):
            cmap = iter(plt.cm.rainbow(np.linspace(0, 1, len(cam_labels[cam]))))
            for (label, coord), color in zip(cam_labels[cam].items(), cmap):
                cax.set_title('%s labels' % cam).set_y(1.005)
                cax.scatter(*coord, c=color, label=label)
                cax.legend()

        # Triangulate the two sets of labels, and map them to 3D
        dlc3d_cfg = dlc3d_cfgs[pair]
        trian_dict, trian = triangulate_basis_labels(
            dlc3d_cfg, cam_labels, pair, decrement=decrement, keys=True
        )

        cmap = iter(plt.cm.rainbow(np.linspace(0, 1, len(trian_dict)+1)))
        for (label, coord), color in zip(trian_dict.items(), cmap):
            ax1.scatter(*(coord - trian_dict['origin']), c=color, label=label)

        if CAMERAS[cam1][0][1] == 'close':
            c_origin = trian[0] + (trian[1] - trian[0]) / 2
        else:
            c_origin = trian[1] + (trian[0] - trian[1]) / 2
        c_origin -= trian_dict['origin']
        ax1.scatter(*c_origin, c=next(cmap), label='computed origin')

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

        fig.savefig( os.path.join(figure_dir, '%d_%s_%s.png' % (PAIR_IDXS[pair], *pair)) )


def visualize_triangulation(config_path, decrement=False):
    dlc3d_cfgs = get_dlc3d_configs(config_path)
    basis_labels = get_labels(config_path)

    cfg = read_config(config_path)
    test_dir = os.path.join(cfg['data_path'], 'test')
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    fig = plt.figure(figsize=(14, 10))

    # Get non-corner pairs by splicing
    pairs = tuple(PAIR_IDXS.keys())[::2]
    for i, pair in enumerate(pairs):
        dlc3d_cfg = dlc3d_cfgs[pair]
        cam1, cam2 = pair

        # Prepare camera plot labels
        cam_labels = get_paired_labels(config_path, pair)['decrement' if decrement is True else 'normal']

        # Triangulate the two sets of labels, and map them to 3D
        trian_dict, trian_coord = triangulate_raw_2d_camera_coords(
            dlc3d_cfg,
            cam1_coords=tuple(cam_labels[cam1].values()),
            cam2_coords=tuple(cam_labels[cam2].values()),
            keys=cam_labels[cam1]
        )

        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        for label, coord in trian_dict.items():
            ax.scatter(*coord, label=label)

        if CAMERAS[cam1][0][1] == 'close':
            c_origin = trian_coord[0] + (trian_coord[1] - trian_coord[0]) / 2
        else:
            c_origin = trian_coord[1] + (trian_coord[0] - trian_coord[1]) / 2
        ax.scatter(*c_origin, label='computed origin')

        ax.legend()
        angle_origins = vg.angle(trian_dict['origin'], c_origin)
        ax.set_title('%s %s\nInnerAngle(orgin, c_origin): %.2f deg' % (*pair, angle_origins)).set_y(1.005)

    fig.suptitle('Triangulation visualization', fontsize=20)
    fig.savefig(os.path.join(test_dir, 'visualize_triangulation.png'))


def visualize_basis_vectors(config_path, decrement=False):
    '''
    Parameters
    ----------
    config_path : string
        String containing the full path of the project config.yaml file.
    '''
    stereo_cam_units, orig_maps = create_stereo_cam_origmap(config_path, decrement=False, save=False)

    dlc3d_cfgs = get_dlc3d_configs(config_path)

    cfg = read_config(config_path)
    test_dir = os.path.join(cfg['data_path'], 'test')
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    fig = plt.figure(figsize=(12, 10))
    pairs = tuple(dlc3d_cfgs.keys())
    pair_num = int(len(pairs) / 2)
    for i in range(pair_num):
        pair1 = pairs[i]
        reds = iter(plt.cm.Reds(np.linspace(0.40, 0.60, 3)))

        pair2 = pair_cycler(i+4, pairs=pairs)
        blues = iter(plt.cm.Blues(np.linspace(0.40, 0.60, 3)))

        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        for pair, color in zip((pair1, pair2), (reds, blues)):
            origin = unit_vector(orig_maps[pair]['origin'])
            axes = np.apply_along_axis(
                lambda a: unit_vector(a - origin),
                arr=orig_maps[pair]['map'].T, axis=1
            )
            initials = pair[0][0] + pair[1][0]
            for i, axis in enumerate(axes):
                ax.plot3D(
                    [0, axis[0]], [0, axis[1]], [0, axis[2]],'-',
                    c=next(color), label='%s: r_%d' % (initials, i)
                )
        ax.legend(loc=2)
        ax.set_title('%s %s and %s %s' % (*pair1, *pair2)).set_y(1.015)

    fig.savefig( os.path.join(test_dir, 'visualize_basis_vectors.png') )
    plt.show()


def visualize_basis_vectors_single(config_path, decrement=False):
    '''
    Parameters
    ----------
    config_path : string
        String containing the full path of the project config.yaml file.
    '''
    stereo_cam_units, orig_maps = create_stereo_cam_origmap(config_path, decrement=False, save=False)

    dlc3d_cfgs = get_dlc3d_configs(config_path)

    cfg = read_config(config_path)
    test_dir = os.path.join(cfg['data_path'], 'test')
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    pairs = tuple(dlc3d_cfgs.keys())
    cmap = iter(plt.cm.rainbow(np.linspace(0, 1, 2*len(pairs)-2)))

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    n = np.linspace(0, 5, 100)
    c_spacing = 1 / (len(pairs) - 2)
    for i, pair in enumerate(pairs):
        cam1, cam2 = pair
        basis_vectors = []
        text_locations = []
        for axis in orig_maps[pair]['map'].T:
            basis_vectors.append(n * (axis - orig_maps[pair]['origin'])[np.newaxis, :].T)
            text_locations.append(5.5 * (axis - orig_maps[pair]['origin'])[np.newaxis, :].T)

        rem_space = c_spacing * i
        cmap = plt.cm.rainbow(np.linspace(rem_space, rem_space+0.12, 3))
        r_1, r_2, r_3 = basis_vectors
        t_1, t_2, t_3 = text_locations

        ax.plot(*r_1, label='%s %s r1' % pair, c=cmap[0])
        ax.text(*t_1, label='r1', c=cmap[0])

        ax.plot(*r_2, label='%s %s r2' % pair, c=cmap[1])
        ax.text(*t_2, label='r2', c=cmap[1])

        ax.plot(*r_3, label='%s %s r3/z' % pair, c=cmap[2])
        ax.text(*t_3, label='r3', c=cmap[2])

    ax.set_title('Basis comparison', fontsize=20).set_y(1.005)
    ax.legend(loc=2)
    fig.savefig( os.path.join(test_dir, 'visualize_basis_vectors.png') )


def plot_trajectories(config_path, cm_is_real_idx=True, cols=2):
    '''
    Parameters
    ----------
    config_path : string
        String containing the full path of the project config.yaml file.
    '''
    from matplotlib import cm, colors
    from math import ceil

    from .basis import map_experiment    

    cfg = read_config(config_path)
    tracjetory_dir = Path(cfg['data_path']) / 'test' / 'trajectories'
    if not os.path.exists(tracjetory_dir):
        os.makedirs(tracjetory_dir)

    dfs = map_experiment(config_path, save=False)
    # Experimen DFs
    for exp_info, df in dfs.items():
        print('Plotting the trajectories of experiment "%s"' % exp_info)
        # Region of interest DFs
        exp_dir = tracjetory_dir / exp_info
        if not os.path.exists(exp_dir):
            os.mkdir(exp_dir)

        roi_groups = df.groupby(level=0, axis=1)
        rows_all = ceil(len(roi_groups) / cols)

        fig_all = plt.figure(figsize=(16, 9))
        fig_all.suptitle(exp_info)
        for roi_idx, (roi, df_roi) in enumerate(roi_groups):
            ax_all = fig_all.add_subplot(rows_all, cols, roi_idx+1, projection='3d')
            ax_all.set_title(roi)

            # Compute number of rows_sep that will be allocated to respective figure
            fig_separate = plt.figure(figsize=(16, 9))
            fig_separate.suptitle(roi)
            
            cam_groups = df_roi[roi].groupby(level=0, axis=1)
            rows_sep = ceil(len(cam_groups) / cols)

            frame_ids = df_roi[roi].index.values
            # Create color map for plot
            if cm_is_real_idx is True:
                frames = frame_ids[-1]
                cmap = plt.cm.rainbow(np.linspace(0, 1, frames))
            # Camera DFs
            for cam_idx, (cam, df_cam) in enumerate(cam_groups):
                coords = df_cam.values
                if cm_is_real_idx is False:
                    cmap = plt.cm.rainbow(np.linspace(0, 1, len(coords.shape[0])))
                ax = fig_separate.add_subplot(rows_sep, cols, cam_idx+1,  projection='3d')
                ax.set_title(cam)

                # Get trajectories
                not_nan_locs = np.logical_not(np.any(np.isnan(coords), axis=1))
                # trajectory_start_locs = np.all(np.dstack((not_nan_locs[:-1], not_nan_locs[1:]))[0], axis=1)
                incremental_delta_trajectory_start_locs = {}
                for delta in range(1, 5+1):
                    # Check if potential point has an end node at <delta> number of increments from it, [i, delta]
                    incremental_delta_trajectory_start_locs[delta] = np.append(
                        np.all(np.dstack( (not_nan_locs[:-delta], not_nan_locs[delta:]) )[0], axis=1),
                        [False] * delta     # Mark the last indeces <delta> as False, because they are either end points or invalid end points (can't be start points)
                    )
                # Only use points that have a valid end node, using "incremental_delta_trajectory_start_locs"
                valid_start_locs = np.any(np.dstack(tuple(incremental_delta_trajectory_start_locs.values()))[0], axis=1)
                # valid_coords = coords[valid_start_locs]

                trajectory_start_idxs = np.where(valid_start_locs == True)[0]

                # meta_idx is for "cm_is_real_idx is True"
                for meta_idx, idx in enumerate(trajectory_start_idxs):
                    delta = 1
                    while incremental_delta_trajectory_start_locs[delta][trajectory_start_idxs[meta_idx]] is False:
                        delta += 1
                    ax.plot(
                        (coords[idx][0], coords[idx+delta][0]),
                        (coords[idx][1], coords[idx+delta][1]),
                        (coords[idx][2], coords[idx+delta][2]),
                        'x-', c=cmap[frame_ids[idx if cm_is_real_idx is True else meta_idx]]
                    )
                    ax_all.plot(
                        (coords[idx][0], coords[idx+delta][0]),
                        (coords[idx][1], coords[idx+delta][1]),
                        (coords[idx][2], coords[idx+delta][2]),
                        '-', label=cam
                    )

            fig_separate.savefig(str( exp_dir / ('%s.png' % roi) ))
        fig_all.savefig(str( exp_dir / 'all.png'))


def dlc3d_create_labeled_video(config_path, video_root=None):
    '''
    Augmented function from https://github.com/AlexEMG/DeepLabCut

    Create pairwise videos
    
    '''

    start_path = os.getcwd()

    cfg = read_config(config_path)
    triangulate_path = os.path.join(cfg['results_path'], 'triangulated')
    print(triangulate_path)
    if not os.path.exists(triangulate_path) or 0 == len(glob(os.path.join(triangulate_path, '*'))):
        msg = 'Could not detect triangulated coordinates in %s' % triangulate_path
        raise ValueError(msg)
    triangulate_path = Path(triangulate_path)

    skipped = []
    dlc3d_cfgs = get_dlc3d_configs(config_path)
    futures = {}
    # with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
    for pair, dlc3d_cfg_path in dlc3d_cfgs.items():
        dlc3d_cfg = read_config(dlc3d_cfg_path)
        pcutoff = dlc3d_cfg['pcutoff']
        markerSize = dlc3d_cfg['dotsize']
        alphaValue = dlc3d_cfg['alphaValue']
        cmap = dlc3d_cfg['colormap']
        skeleton_color = dlc3d_cfg['skeleton_color']
        scorer_3d = dlc3d_cfg['scorername_3d']

        bodyparts2connect = dlc3d_cfg['skeleton']
        bodyparts2plot = list(np.unique([val for sublist in bodyparts2connect for val in sublist]))
        color = plt.cm.get_cmap(cmap, len(bodyparts2plot))

        cam1, cam2 = pair

        if video_root is None:
            video_root = os.path.join(os.path.dirname(dlc3d_cfg_path), 'videos')
        else:
            video_root = os.path.realpath(video_root)
            
        cam1_videos = glob(os.path.join(video_root, ('*%s*' % cam1)))
        cam2_videos = glob(os.path.join(video_root, ('*%s*' % cam2)))

        for i, v_path in enumerate(cam1_videos):
            _, video_name = os.path.split(v_path)
            cam1_video, cam2_video = cam1_videos[i], cam2_videos[i]
            a_id, trial, vcam, date = video_name.replace('.avi', '').split('_')
            futures[create_video(
                # Paths
                triangulate_path, cam1_video, cam2_video,
                # ID
                a_id, trial, vcam, date, pair,
                # Config
                dlc3d_cfg, pcutoff, markerSize, alphaValue, cmap, skeleton_color, scorer_3d,
                bodyparts2plot, bodyparts2connect, color
            )] = (a_id, trial, vcam, date, pair)

    # for future in concurrent.futures.as_completed(futures):
    #     video_id = futures[future]
    #     try:
    #         result = future.result()
    #     except Exception as exc:
    #         print('%s generated an exception: %s' % (video_id, exc))
    #     else:
    #         print('%s = %s' % (video_id, result))

    os.chdir(start_path)


def create_video(
        # Paths
        triangulate_path, cam1_video, cam2_video,
        # ID
        a_id, trial, vcam, date, pair,
        # Config
        dlc3d_cfg, pcutoff, markerSize, alphaValue, cmap, skeleton_color, scorer_3d,
        bodyparts2plot, bodyparts2connect, color
    ):
    cam1, cam2 = pair

    trial_trian_result_path = triangulate_path / ('%s_%s_%s' % (a_id, trial, date)) / ('%s_%s' % pair)

    xyz_path = glob(str(trial_trian_result_path / '*_DLC_3D.h5'))[0]
    xyz_df = read_hdf(xyz_path, 'df_with_missing')

    try:
        df_cam1 = read_hdf(glob(str(trial_trian_result_path / ('*%s*filtered.h5' % cam1)))[0])
        df_cam2 = read_hdf(glob(str(trial_trian_result_path / ('*%s*filtered.h5' % cam2)))[0])
    except FileNotFoundError:
        df_cam1 = read_hdf(glob(str(trial_trian_result_path / ('*%s*.h5' % cam1)))[0])
        df_cam2 = read_hdf(glob(str(trial_trian_result_path / ('*%s*.h5' % cam2)))[0])

    vid_cam1 = cv2.VideoCapture(cam1_video)
    vid_cam2 = cv2.VideoCapture(cam2_video)
    file_name = '%s_%s_%s_%s_%s' % (a_id, trial, date, cam1, cam2)
    for k in tqdm(tuple(range(0, len(xyz_df)))):
        output_folder, num_frames = plot2D(
            dlc3d_cfg, k, bodyparts2plot, vid_cam1, vid_cam2,
            bodyparts2connect, df_cam1, df_cam2, xyz_df, pcutoff,
            markerSize,alphaValue, color, trial_trian_result_path,
            file_name, skeleton_color, view=[-113, -270],
            draw_skeleton=True, trailpoints=0,
            xlim=(None, None), ylim=(None, None), zlim=(None, None)
        )
    
    cwd = os.getcwd()
    os.chdir(str(output_folder))
    subprocess.call([
        'ffmpeg',
        '-threads', '0',
        '-start_number', '0',
        '-framerate', '30',
        '-i', str('img%0' + str(num_frames) + 'd.png'),
        '-r', '30', '-vb', '20M',
        os.path.join(output_folder, str('../' + file_name + '.mpg')),
    ])
    os.chdir(cwd)
