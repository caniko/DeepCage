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



def initialise_projects(
        project_name, experimenter,
        root, calib_root, video_root, dlc_config=None,
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
    root : string
        String containing the full path of to the directory where the new project files should be located
    calib_root : string
        String containing the full path of to the root directory storing the calibration files for each dlc 3D project
    dlc_config : string or None; default None
        String containing the full path of to the dlc config.yaml file that will be used for the dlc 3D projects
    '''
    from deeplabcut.create_project import create_new_project_3d, create_new_project
    from deepcage.auxiliary.detect import detect_dlc_calibration_images
    from deeplabcut.utils.auxiliaryfunctions import write_config_3d
    from .create import create_dc_project
    from .utils import png_to_jpg

    if 'jpg' in image_format or 'jpeg' in image_format:
        raise ValueError('Not %s supported, yet' % image_format)

    if not os.path.exists(video_root):
        raise ValueError('The path to the video root, does not exist:\n%s' % video_root)
    
    if not os.path.exists(root):
        raise ValueError('The given "root" does not exist:\n%s' % root)

    root_path = Path(root)
    video_dir = root_path / 'videos'

    video_root_subdirs = glob(os.path.join(video_root, '*/'))
    hierarchy = {}
    videos = []
    if video_dir_hierarchy == ('trial', 'pair'):
        for trial in video_root_subdirs:
            trial_name = str(Path(trial).stem)
            pairs_cams_vids = {}
            for pair in glob(os.path.join(trial, '*/')):
                pair_name = str(Path(pair).stem)
                for vid in glob(os.path.join(pair, '*.avi')):
                    video_filename = Path(vid).stem
                    pair_id, cam_id, cam, trial = video_filename.split('_')

                    if copy_videos is True:
                        new_video_dir = video_dir / trial_name / pair_name
                        if not os.exists(str(new_video_dir)):
                            os.makedirs(new_video_dir)
                        vid_path = new_video_dir / vid
                        copyfile(vid, vid_path)
                    else:
                        vid_path = os.path.realpath(vid)

                    pairs_cams_vids[(pairs, cam)] = vid_path
                    videos.append(vid_path)
            hierarchy[trial_name] = copy(pairs_cams_vids)
    else:
        raise ValueError('video_root_depth must be ("trial", "pair")')

    if dlc_config is None:
        dlc_config = create_new_project(project_name, experimenter, videos, working_directory=None, copy_videos=False,videotype='.avi')

    dlc3d_project_configs = {}
    with concurrent.futures.ProcessPoolExecutor(max_worker=4) as executor:
        for pair, calib_paths in detect_dlc_calibration_images(calib_root).items():
            cam1, cam2 = pair

            name = '%d_%s_%s' % (PAIR_IDXS[pair], cam1, cam2)
            dlc3d_project_configs[pair] = create_new_project_3d(name, experimenter, num_cameras=2, working_directory=root)
            project_path = Path(os.path.dirname(dlc3d_project_configs[pair]))

            calibration_images_path = project_path / 'calibration_images'
            if not os.path.exists(calibration_images_path):
                os.makedirs(calibration_images_path)

            if 'png' in image_format:
                executor.submit(png_to_jpg, calibration_images_path, img_paths=calib_paths)
            elif 'jpg' in image_format or 'jpeg' in image_format:
                # TODO: Implement solution
                pass

            cfg = read_config(dlc3d_project_configs[pair])
            cfg['config_file_camera-1'] = dlc_config
            cfg['config_file_camera-2'] = dlc_config
            cfg['camera_names'] = list(pair)
            cfg['trainingsetindex_'+cam1] = cfg.pop('trainingsetindex_camera-1')
            cfg['trainingsetindex_'+cam2] = cfg.pop('trainingsetindex_camera-2')
            write_config_3d(dlc3d_project_configs[pair], cfg)

    config_path = create_dc_project(project_name, experimenter, dlc_config, dlc3d_project_configs, working_directory=root)

    calibrate_dlc_cameras(config_path, cbrow=9, cbcol=6, calibrate=False, alpha=0.9)
    return config_path