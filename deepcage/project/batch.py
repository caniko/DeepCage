import concurrent.futures

from pathlib import Path
from glob import glob
import os

from deepcage.project.analysis import calibrate_dlc_cameras
from deepcage.project.create import create_dc_project
from deepcage.project.edit import read_config
from deepcage.auxiliary.constants import CAMERAS, PAIR_IDXS, get_pairs
from deepcage.auxiliary.detect import detect_dlc_calibration_images


def initialise_projects(project_name, experimenter, root, dlc_config, calib_root):
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
    dlc_config : string
        String containing the full path of to the dlc config.yaml file that will be used for the dlc 3D projects
    calib_root : string
        String containing the full path of to the root directory storing the calibration files for each dlc 3D project
    '''
    from deeplabcut.create_project import create_new_project_3d
    from deeplabcut.utils.auxiliaryfunctions import write_config_3d
    from .utils import png_to_jpg

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

            executor.submit(png_to_jpg, calibration_images_path, img_paths=calib_paths)
            
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