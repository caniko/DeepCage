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
from .utils import rad_to_deg, unit_vector, equalise_3daxes, create_df_from_coords


def visualize_workflow(config_path, normalize=True, decrement=False, save=True):
    '''
    Parameters
    ----------
    config_path : string
        Absolute path of the project config.yaml file.
    '''
    import matplotlib.image as image

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
        cam_labels = get_paired_labels(config_path, pair)['decrement' if decrement is True else 'normal']

        # Plot manual labels
        for cam, cax in zip(pair, (ax['cam1'], ax['cam2'])):
            # Add respective calibration image as background
            img_cam = image.imread(glob(os.path.join(cfg['calibration_path'], cam+'*'))[0])
            cax.imshow(img_cam)

            cmap = iter(plt.cm.rainbow(np.linspace( 0, 1, len(cam_labels[cam]) )))
            for (label, coord), color in zip(cam_labels[cam].items(), cmap):
                cax.set_title((cam, 'labels')).set_y(1.005)
                cax.scatter(*coord, c=color, label=label)
                cax.legend()

        # Triangulate the two sets of labels, and map them to 3D
        dlc3d_cfg = dlc3d_cfgs[pair]
        trian_dict, trian = triangulate_basis_labels(
            dlc3d_cfg, cam_labels, pair, decrement=decrement, keys=True
        )

        cmap = iter(plt.cm.rainbow(np.linspace( 0, 1, len(trian_dict)+1) ))
        for (label, coord), color in zip(trian_dict.items(), cmap):
            ax['trian'].scatter(*(coord - trian_dict['origin']), c=color, label=label)

        if CAMERAS[cam1][0][1] == 'left':
            c_origin = trian[0] + (trian[1] - trian[0]) / 2
        else:
            c_origin = trian[1] + (trian[0] - trian[1]) / 2
        c_origin -= trian_dict['origin']
        ax['trian'].scatter(*c_origin, c=next(cmap), label='computed origin')

        ax['trian'].set_title('Triangualted').set_y(1.005)
        ax['trian'].legend()

        ax['trian'].set_xlabel('X', fontsize=10)
        ax['trian'].set_ylabel('Y', fontsize=10)
        ax['trian'].set_zlabel('Z', fontsize=10)

        _, orig_map = compute_basis_vectors(trian, pair, normalize=normalize, decrement=decrement)

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
            fig.savefig( os.path.join(figure_dir, '%d_%s_%s.png' % (PAIR_IDXS[pair], *pair)) )

    return fig, ax


def visualize_triangulation(config_path, decrement=False, save=True):
    dlc3d_cfgs = get_dlc3d_configs(config_path)
    basis_labels = get_labels(config_path)

    cfg = read_config(config_path)
    test_dir = os.path.join(cfg['data_path'], 'test')
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    fig = plt.figure(figsize=(14, 10))

    # Get non-corner pairs by splicing
    pairs = tuple(PAIR_IDXS.keys())[::2]
    pair_ax = {}
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

        pair_ax[pair] = fig.add_subplot(2, 2, i+1, projection='3d')
        for label, coord in trian_dict.items():
            pair_ax[pair].scatter(*coord, label=label)

        if CAMERAS[cam1][0][1] == 'left':
            c_origin = trian_coord[0] + (trian_coord[1] - trian_coord[0]) / 2
        else:
            c_origin = trian_coord[1] + (trian_coord[0] - trian_coord[1]) / 2
        pair_ax[pair].scatter(*c_origin, label='computed origin')

        pair_ax[pair].legend()
        angle_origins = vg.angle(trian_dict['origin'], c_origin)
        pair_ax[pair].set_title('%s %s\nInnerAngle(orgin, c_origin): %.2f deg' % (*pair, angle_origins)).set_y(1.005)

    fig.suptitle('Triangulation visualization', fontsize=20)

    if save is True:
        fig.savefig(os.path.join(test_dir, 'visualize_triangulation.png'))

    return fig, pair_ax


def visualize_basis_vectors(config_path, normalize=True, decrement=False, save=True):
    '''
    Parameters
    ----------
    config_path : string
        Absolute path of the project config.yaml file.
    '''
    stereo_cam_units, orig_maps = create_stereo_cam_origmap(config_path, normalize=normalize, decrement=decrement, save=False)

    cfg = read_config(config_path)
    dlc3d_cfgs = get_dlc3d_configs(config_path)

    test_dir = os.path.join(cfg['data_path'], 'test')
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    pairs = tuple(dlc3d_cfgs.keys())
    pair_num = int(len(pairs) / 2)

    fig = plt.figure(figsize=(14, 8))
    ax_duo = {}
    for i in range(pair_num):
        pair1 = pairs[i]
        reds = iter(plt.cm.Reds(np.linspace(0.38, 0.62, 3)))

        pair2 = pair_cycler(i+4, pairs=pairs)
        blues = iter(plt.cm.Blues(np.linspace(0.38, 0.62, 3)))

        ax_duo[(pair1, pair2)] = fig.add_subplot(2, 2, i+1, projection='3d')
        for pair, color in zip((pair1, pair2), (reds, blues)):
            axes = list(stereo_cam_units[pair].values())[:3]
            initials = pair[0][0] + pair[1][0]
            for i, axis in enumerate(axes):
                ax_duo[(pair1, pair2)].plot(
                    [0, axis[0]], [0, axis[1]], [0, axis[2]],'-',
                    c=next(color), label=f'{initials}: r{i}'
                )
        ax_duo[(pair1, pair2)].legend(loc=2)
        ax_duo[(pair1, pair2)].set_title('%s %s and %s %s' % (*pair1, *pair2)).set_y(1.015)

    if save is True:
        fig.savefig( os.path.join(test_dir, 'visualize_basis_vectors.png') )

    return fig, ax_duo


def visualize_basis_vectors_single(config_path, normalize=True, decrement=False, save=True):
    '''
    Parameters
    ----------
    config_path : string
        Absolute path of the project config.yaml file.
    '''
    stereo_cam_units, orig_maps = create_stereo_cam_origmap(config_path, normalize=normalize, decrement=False, save=False)

    dlc3d_cfgs = get_dlc3d_configs(config_path)

    cfg = read_config(config_path)
    test_dir = os.path.join(cfg['data_path'], 'test')
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    pairs = tuple(dlc3d_cfgs.keys())
    cmap = iter(plt.cm.rainbow(np.linspace(0, 1, 2*len(pairs)-2)))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    c_spacing = 1 / (len(pairs) - 2)
    for i, pair in enumerate(pairs):
        cam1, cam2 = pair

        rem_space = c_spacing * i
        cmap = plt.cm.rainbow(np.linspace(rem_space, rem_space+0.12, 3))

        ax.plot(
            [0, stereo_cam_units[pair]['x-axis'][0]],
            [0, stereo_cam_units[pair]['x-axis'][1]],
            [0, stereo_cam_units[pair]['x-axis'][2]],
            label='%s %s r1/x' % pair, c=cmap[0]
        )
        # ax.text(*t_1, label='r1', c=cmap[0])

        ax.plot(
            [0, stereo_cam_units[pair]['y-axis'][0]],
            [0, stereo_cam_units[pair]['y-axis'][1]],
            [0, stereo_cam_units[pair]['y-axis'][2]],
            label='%s %s r2/y' % pair, c=cmap[1]
        )
        # ax.text(*t_2, label='r2', c=cmap[1])

        ax.plot(
            [0, stereo_cam_units[pair]['z-axis'][0]],
            [0, stereo_cam_units[pair]['z-axis'][1]],
            [0, stereo_cam_units[pair]['z-axis'][2]],
            label='%s %s r3/z' % pair, c=cmap[2]
        )
        # ax.text(*t_3, label='r3', c=cmap[2])

    ax.set_title('Basis comparison', fontsize=20).set_y(1.005)
    ax.legend(loc=2)

    if save is True:
        fig.savefig( os.path.join(test_dir, 'visualize_basis_vectors.png') )

    return fig, ax


def plot_2d_trajectories(config_path, cm_is_real_idx=True, save=True):
    pass


def plot_3d_trajectories(config_path, cm_is_real_idx=True, cols=2, remap=True, normalize=True, use_saved_origmap=True, save=True):
    '''
    Parameters
    ----------
    config_path : string
        Absolute path of the project config.yaml file.
    cm_is_real_idx : bool
        The trajectories are color-mapped with basis on their position in the array, rainbow (red->green->blue)
    '''
    from matplotlib import cm, colors
    from math import ceil

    from deepcage.auxiliary.detect import detect_triangulation_result
    from .basis import map_experiment    

    cfg = read_config(config_path)
    data_path = cfg['data_path']
    tracjetory_dir = Path(data_path) / 'test' / 'trajectories'
    if not os.path.exists(tracjetory_dir):
        os.makedirs(tracjetory_dir)

    if remap is True:
        dfs = map_experiment(config_path, save=False, use_saved_origmap=use_saved_origmap, normalize=normalize)
    else:
        pass
        # basis_result_path = os.path.join(data_path, 'cb_result.pickle')
        # try:
        #     with open(basis_result_path, 'rb') as infile:
        #         stereo_cam_units, orig_maps = pickle.load(infile)
        # except FileNotFoundError:
        #     msg = f'Could not detect results from deepcage.compute.generate_linear_map() in:\n{basis_result_path}' 
        #     raise FileNotFoundError(msg)

        # dfs = create_df_from_coords(detect_triangulation_result(config_path, change_basis=False))
        # if dfs is False:
        #     print('According to the DeepCage triangulated coordinates detection algorithm this project is not ready for changing basis')
        #     return False

    # Experimen DFs
    for exp_info, df in dfs.items():
        print(f'Plotting the trajectories of experiment "{exp_info}"')
        # Region of interest DFs
        exp_info = exp_info if remap else f'unmapped_{exp_info}'
        exp_dir = tracjetory_dir / exp_info
        if not os.path.exists(exp_dir):
            os.mkdir(exp_dir)

        roi_groups = df.groupby(level=0, axis=1)
        rows_all = ceil(len(roi_groups) / cols)


        fig_all = plt.figure(figsize=(16, 8))
        fig_all.suptitle(exp_info)

        ax_all, ax_sep = {}, {}
        for roi_idx, (roi, df_roi) in enumerate(roi_groups):
            pair_groups = df_roi[roi].groupby(level=0, axis=1)
            num_pairs = len(pair_groups)
            rows_sep = ceil(num_pairs / cols)

            # Plot related chores
            ax_all[roi] = fig_all.add_subplot(rows_all, cols, roi_idx+1, projection='3d')
            ax_all[roi].set_title(roi)

            ax_sep[roi] = {}

            # Create color map
            # All plot figure
            cmap_all = plt.cm.rainbow(np.linspace(0, 1, num_pairs))
            # Separate plot figures
            frame_ids = df_roi[roi].index.values
            if cm_is_real_idx is True:
                frames = frame_ids[-1]
                cmap_sep = plt.cm.rainbow(np.linspace(0, 1, frames))
            
            # Compute number of rows_sep that will be allocated to respective figure
            fig_sep = plt.figure(figsize=(16, 8))       # sep -> separate
            fig_sep.suptitle(roi)

            # Pair DFs
            for pair_idx, (pair, df_pair) in enumerate(pair_groups):
                coords = df_pair.values
                if cm_is_real_idx is False:
                    cmap_sep = plt.cm.rainbow(np.linspace(0, 1, len(coords.shape[0])))
                ax_sep[roi][pair] = fig_sep.add_subplot(rows_sep, cols, pair_idx+1,  projection='3d')
                ax_sep[roi][pair].set_title(pair)

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
                    ax_all[roi].plot(
                        (coords[idx][0], coords[idx+delta][0]),
                        (coords[idx][1], coords[idx+delta][1]),
                        (coords[idx][2], coords[idx+delta][2]),
                        '-', c=cmap_all[pair_idx]
                    )
                    ax_sep[roi][pair].plot(
                        (coords[idx][0], coords[idx+delta][0]),
                        (coords[idx][1], coords[idx+delta][1]),
                        (coords[idx][2], coords[idx+delta][2]),
                        'x-', c=cmap_sep[frame_ids[idx if cm_is_real_idx is True else meta_idx]]
                    )

            if save is True:
                fig_sep.savefig(str( exp_dir / f'{roi}.png' ))

        if save is True:
            fig_all.savefig(str( exp_dir / 'all.png'))

    return (fig_all, ax_all), (fig_sep, ax_sep)


def dlc3d_create_labeled_video(config_path, video_root=None, video_dir_hierarchy=False, remove_origin=False):
    '''
    Augmented function from https://github.com/AlexEMG/DeepLabCut

    Create pairwise videos
    
    '''
    from deepcage.auxiliary.detect import detect_videos_in_hierarchy

    start_path = os.getcwd()

    cfg = read_config(config_path)
    triangulate_path = os.path.join(cfg['results_path'], 'triangulated')
    if not os.path.exists(triangulate_path) or 0 == len(glob(os.path.join(triangulate_path, '*'))):
        msg = f'Could not detect triangulated coordinates in {triangulate_path}'
        raise ValueError(msg)
    triangulate_path = Path(triangulate_path)
    
    if remove_origin is True:
        basis_result_path = os.path.join(cfg['data_path'], 'cb_result.pickle')
        try:
            with open(basis_result_path, 'rb') as infile:
                stereo_cam_units, orig_maps = pickle.load(infile)
        except FileNotFoundError:
            msg = f'Could not detect results from deepcage.compute.generate_linear_map() in:\n{basis_result_path}'
            raise FileNotFoundError(msg)

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

        if video_dir_hierarchy is True:
            hierarchy, _ = detect_videos_in_hierarchy(
                video_root, deep_dict=True
            )
            for exp_id, pairs in hierarchy.items():
                for pair_info, cams in pairs.items():
                    pair_idx, cam1, cam2 = pair_info.split('_')
                    pair = (cam1, cam2)
                    cam1_video, cam2_video = cams.values()
                    info = exp_id
                    futures[create_video(
                        # Paths
                        (triangulate_path / exp_id / pair_info), cam1_video, cam2_video,
                        # ID
                        info, pair,
                        # Config
                        dlc3d_cfg, pcutoff, markerSize, alphaValue, cmap, skeleton_color, scorer_3d,
                        bodyparts2plot, bodyparts2connect, color,
                        # Style
                        origin_to_remove=orig_maps[pair]['origin'] if remove_origin is True else None,
                        new_path=True
                    )] = (*info, pair)

        else:
            if video_root is None:
                video_root = os.path.join(os.path.dirname(dlc3d_cfg_path), 'videos')
            else:
                video_root = os.path.realpath(video_root)

            cam1_videos = glob(os.path.join(video_root, (f'*{cam1}*')))
            cam2_videos = glob(os.path.join(video_root, (f'*{cam2}*')))

            for i, v_path in enumerate(cam1_videos):
                _, video_name = os.path.split(v_path)
                cam1_video, cam2_video = cam1_videos[i], cam2_videos[i]
                info = video_name.replace('.avi', '').split('_')
                futures[create_video(
                    # Paths
                    triangulate_path, cam1_video, cam2_video,
                    # ID
                    info, pair,
                    # Config
                    dlc3d_cfg, pcutoff, markerSize, alphaValue, cmap, skeleton_color, scorer_3d,
                    bodyparts2plot, bodyparts2connect, color,
                    # Style
                    origin_to_remove=orig_maps[pair]['origin'] if remove_origin is True else None,
                    new_path=True
                )] = (*info, pair)

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
        info, pair,
        # Config
        dlc3d_cfg, pcutoff, markerSize, alphaValue, cmap, skeleton_color, scorer_3d,
        bodyparts2plot, bodyparts2connect, color,
        # Style
        origin_to_remove=None, new_path=True
    ):
    cam1, cam2 = pair

    stringified_info = info if isinstance(info, str) else '_'.join(info)
    if new_path is False:
        trial_trian_result_path = triangulate_path / stringified_info / ('%s_%s' % pair)
    else:
        trial_trian_result_path = triangulate_path

    xyz_path = glob(str(trial_trian_result_path / '*_DLC_3D.h5'))[0]
    xyz_df = read_hdf(xyz_path, 'df_with_missing')

    try:
        df_cam1 = read_hdf(glob(str(trial_trian_result_path / (f'*{cam1}*filtered.h5')))[0])
        df_cam2 = read_hdf(glob(str(trial_trian_result_path / (f'*{cam2}*filtered.h5')))[0])
    except FileNotFoundError:
        df_cam1 = read_hdf(glob(str(trial_trian_result_path / (f'*{cam1}*.h5' % cam1)))[0])
        df_cam2 = read_hdf(glob(str(trial_trian_result_path / (f'*{cam2}*.h5' % cam2)))[0])

    vid_cam1 = cv2.VideoCapture(cam1_video)
    vid_cam2 = cv2.VideoCapture(cam2_video)

    if origin_to_remove is not None:
        for roi, df_roi in xyz_df['DLC_3D'].groupby(level=0, axis=1):
            for axis_num, (param, param_df) in enumerate(df_roi[roi].groupby(level=0, axis=1)):
                xyz_df.loc[:, ('DLC_3D', roi, param)] = xyz_df.loc[:, ('DLC_3D', roi, param)] - origin_to_remove[axis_num]

        file_name = f'new_origin_{stringified_info}_{cam1}_{cam2}'
    else:
        file_name = f'{stringified_info}_{cam1}_{cam2}'

    video_output = os.path.join(trial_trian_result_path, file_name+'.mpg')
    if not os.path.exists(video_output):
        for k in tqdm(tuple(range(0, len(xyz_df)))):
            output_folder, num_frames = plot2D(
                dlc3d_cfg, k, bodyparts2plot, vid_cam1, vid_cam2,
                bodyparts2connect, df_cam1, df_cam2, xyz_df, pcutoff,
                markerSize, alphaValue, color, trial_trian_result_path,
                file_name, skeleton_color, view=[-113, -270],
                draw_skeleton=True, trailpoints=0,
                xlim=[None, None], ylim=[None, None], zlim=[None, None]
            )
        
        cwd = os.getcwd()
        os.chdir(str(output_folder))
        subprocess.call([
            'ffmpeg',
            '-start_number', '0',
            '-framerate', '30',
            '-i', f'img%0{num_frames}d.png',
            '-r', '30', '-vb', '20M',
            os.path.join(output_folder, f'../{file_name}.mpg'),
        ])
        os.chdir(cwd)
    else:
        print(f'SKIPPING! Video already exists: {video_output}')
