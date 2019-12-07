import concurrent.futures

from pathlib import Path
from glob import glob
import os

from shutil import copyfile
from copy import copy

from deepcage.auxiliary.constants import CAMERAS, PAIR_IDXS, get_pairs
from deepcage.project.analysis import calibrate_dlc_cameras
from deepcage.project.create import create_dc_project
from deepcage.project.edit import read_config


def create_dlc_dc_projects(
        project_name, experimenter,
        project_root, calib_root, video_root, dlc_config=None,
        video_dir_hierarchy=('trial', 'pair'), copy_videos=True, image_format='png'
    ):
    '''
    Initialise a DeepCage project along with the DeepLabCut 3D for each stereo camera. Following the creation of these DLC 3D projects,
    detect calibration images located in a standard Bonsai project (can be found in examples in repo); move the images to their respective
    DLC 3D projects. Run deeplabcut.calibrate_cameras(calibrate=False) for each DLC 3D project,
    and prepare for quality assurance of the checkerboard detections.
    
    Parameters
    ----------
    project_name : str
        String containing the project_name of the project.
    experimenter : str
        String containing the project_name of the experimenter/scorer.
    project_root : string
        Absolute path of to the directory where the new project files should be located
    calib_root : string
        Absolute path of to the project_root directory storing the calibration files for each dlc 3D project
    dlc_config : string or None; default None
        Absolute path of to the dlc config.yaml file that will be used for the dlc 3D projects
    '''
    from deepcage.auxiliary.detect import detect_dlc_calibration_images, detect_videos_in_hierarchy
    from deeplabcut.generate_training_dataset import extract_frames
    from deeplabcut.create_project import create_new_project_3d, create_new_project
    from deeplabcut.utils.auxiliaryfunctions import write_config_3d

    from .create import create_dc_project, create_dlc3d_projects
    from .utils import png_to_jpg

    if 'jpg' in image_format or 'jpeg' in image_format:
        raise ValueError('Not %s supported, yet' % image_format)

    if not os.path.exists(video_root):
        raise ValueError('The path to the video project_root, does not exist:\n%s' % video_root)
    
    if not os.path.exists(project_root):
        raise ValueError('The given "project_root" does not exist:\n%s' % project_root)

    root_path = Path(project_root) / project_name
    video_dir = root_path / 'videos'

    # Get videos
    hierarchy, videos = detect_videos_in_hierarchy(
        video_root,
        video_dir_hierarchy=video_dir_hierarchy,
        copy_video_dir=video_dir if copy_videos is True else None
    )

    # Create a project specific DeepLabCut markerless pose estimation project, if not defined
    if dlc_config is None:
        dlc_config = create_new_project(
            f'dlc2d_{project_name}', experimenter, videos,
            working_directory=root_path,
            copy_videos=False,
            videotype='.avi'
        )
        extract_frames(str(dlc_config), userfeedback=False)

    dlc3d_root = root_path / 'DeepLabCut3D'
    if not os.path.exists(dlc3d_root):
        os.mkdir(dlc3d_root)

    dlc3d_project_configs = create_dlc3d_projects(
        project_name, experimenter, calib_root, dlc3d_root, dlc_config, image_format='png'
    )

    config_path = create_dc_project(project_name, experimenter, dlc_config, dlc3d_project_configs, working_directory=root_path)

    calibrate_dlc_cameras(config_path, cbrow=9, cbcol=6, calibrate=False, alpha=0.9)
    return config_path


def run_all_tests(
        config_path, save=True, show=False,
        # Triangulation settings
        undistort=True, remap=True, normalize=True,
        # Label settings
        decrement=False, cm_is_real_idx=True
    ):
    '''Function for running all compute.test functions simultaniously'''

    from matplotlib.pyplot import show
    from deepcage.compute.test import (
        visualize_workflow, visualize_triangulation, visualize_basis_vectors,
        visualize_basis_vectors_single, plot_3d_trajectories
    )

    vw_figures_axes = visualize_workflow(
        config_path, undistort=undistort, normalize=normalize, decrement=decrement, save=save
    )
    if show is True:
        show()

    vt_figures_axes = visualize_triangulation(config_path, undistort=undistort, decrement=decrement, save=save)
    if show is True:
        show()

    bv__figures_axes = visualize_basis_vectors(
        config_path, undistort=undistort, normalize=normalize, decrement=decrement, save=save
    )
    if show is True:
        show()

    bvs_figures_axes = visualize_basis_vectors_single(
        config_path, undistort=undistort, normalize=normalize, decrement=decrement, save=save
    )
    if show is True:
        show()

    trajectory3d_figures_axes = plot_3d_trajectories(
        config_path, undistort=undistort, cm_is_real_idx=cm_is_real_idx, cols=2, remap=remap,
        normalize=normalize, use_saved_origmap=True, save=save
    )
    if show is True:
        show()
