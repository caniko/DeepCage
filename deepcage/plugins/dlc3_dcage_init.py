from shutil import copyfile
from pathlib import Path
from glob import glob
import os

from deeplabcut.create_project import create_new_project_3d

from deepcage.auxiliary import create_dc_project, get_pairs, CAMERAS, PAIR_IDXS


def dlc3d_deepcage_calibration(project_name, experimenter, root, dlc_config, calib_root, vid_format='avi'):
    '''
    Create frame pairs from temporally close frames within DeepCage camera-pairs
    
    Parameters
    ----------
    project_name : str
        String containing the project_name of the project.
    experimenter : str
        String containing the project_name of the experimenter/scorer.
    root : string
        String containing the full path of to the directoy where the new project files should be located
    dlc_config : string
        String containing the full path of to the dlc config.yaml file that will be used for the dlc 3D projects
    calib_root : string
        String containing the full path of to the root directory storing the calibration files for each dlc 3D project
    vid_format : string; default 'avi'
    '''
    dlc3d_project_configs = {}
    for pair, calib_paths in detect_calibration_images(calib_root).items():
        cam1, cam2 = pair

        name = '%d_%s_%s' % (PAIR_IDXS[pair], cam1, cam2)
        dlc3d_project_configs[pair] = create_new_project_3d(name, experimenter, num_cameras=2, working_directory=root)
        project_path = Path(os.path.dirname(dlc3d_project_configs[pair]))

        calibration_images_path = project_path / 'calibration_images'
        if not os.path.exists(calibration_images_path):
            os.makedirs(calibration_images_path)

        png_to_jpg(calibration_images_path, img_paths=calib_paths)

    return create_dc_project(project_name, experimenter, dlc_config, dlc3d_project_configs, working_directory=root)


def detect_calibration_images(root, img_format='png'):
    subdirs = glob(os.path.join(root, '*/'))
    result = {}
    for subdir in subdirs:
        subdir = os.path.dirname(subdir)
        idx, fcam1, fcam2 = os.path.basename(subdir).split('_')
        result[(fcam1, fcam2)] = [os.path.realpath(calib_img) for calib_img in glob(os.path.join(subdir, '**/*.png'))]

    return result


def png_to_jpg(save_dir, img_paths=None, img_root=None, codec='cv'):
    assert os.path.exists(save_dir), 'Does not exist:\n%s' % save_dir

    if img_paths is None:
        img_paths = glob(os.path.join(img_root, '**/*.png'))
    elif img_root is None:
        msg = "Either img_paths or img_root has to be defined"
        ValueError(msg)

    for img in img_paths:
        img_path = os.path.realpath(img)

        if '\\' in img_path:
            separator = '\\'
        elif '/' in img_path:
            separator = '/'

        save_path = os.path.join(save_dir, img.replace('png', 'jpg'))
        if codec == 'pil':
            from PIL import Image

            im = Image.open(img_path)
            rgb_im = im.convert('RGB')
            rgb_im.save(save_path)

        elif codec == 'cv':
            import cv2

            jpg = cv2.imread(img_path)
            cv2.imwrite(save_path, jpg)
