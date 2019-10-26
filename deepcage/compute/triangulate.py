import numpy as np
import pickle
import cv2
import os

from deeplabcut.utils import auxiliaryfunctions_3d
from deepcage.auxiliary import read_config

from .utils import get_coord


def triangulate_raw_2d_camera_coords(config_path, cam1_coords=None, cam2_coords=None, cam1_image=None, cam2_image=None, unit_keys=None):
    """
    Augmented deeplabcut.triangulate() for DeepCage workflow
    
    This function triangulates user-defined coordinates from the two camera views using the camera matrices (derived from calibration) to calculate 3D predictions.
    Optionally, the user can define the coordiantes from images.
    
    Used for changing basis operations.
    
    Note: cam1 is the first camera on the 'camera_names' list located in the project 'config.yaml' file; cam2 is the second camera on the same list

    Parameters
    ----------
    config_path : string
        Full path of the config.yaml file as a string.
    cam1_image : string; default None
        Full path of the image of camera 1 as a string.
    cam2_image : string; default None
        Full path of the image of camera 2 as a string.
    cam1_coords : numpy.array-like; default None
        List of vectors that are coordinates in the camera 1 image
    cam2_coords : numpy.array-like; default None
        List of vectors that are coordinates in the camera 2 image
    unit_keys : list-like; default None
        List of names or dictionary keys that can be associated with the 3d-coordinate with the identical index

    Example
    -------
    To analyze a set of coordinates:
    >>> deeplabcut.triangulate_raw_2d_camera_coords(config_path, cam1_coords=((1, 2), (20, 50), ...), cam2_coords=((3, 5), (14, 2), ...) )

    Linux/MacOS
    To analyze a set of images in a directory:
    >>> deeplabcut.triangulate_raw_2d_camera_coords(config_path, cam1_image='/image_directory/cam1.png', cam2_image='/image_directory/cam2.png')
    
    Windows
    To analyze a set of images in a directory:
    >>> deeplabcut.triangulate_raw_2d_camera_coords(config_path, cam1_image='<drive_letter>:\\<image_directory>\\cam1.png', cam2_image='\\image_directory\\cam2.png')

    """
    # if ((cam1_coords is None and cam2_coords is None) and (cam1_image is None and cam2_image is None)) or (
    #     (cam1_coords is not None and cam2_coords is not None) and (cam1_image is not None and cam2_image is not None)):
    #     msg = 'Must include a set of camera images or 2d-coordinates'
    #     raise ValueError(msg)
    
    if cam1_coords is not None and cam2_coords is not None:
        coords_defined = True
        
    if cam1_image is not None and cam2_image is not None:
        if coords_defined is True:
            msg = 'Must include a set of camera images or 2d-coordinates'
            raise ValueError(msg)
        cam1_coords = get_coord(cam1_image, n=-1)
        cam2_coords = get_coord(cam2_image, n=-1)
        
        if len(cam1_coords) != len(cam2_coords):
            msg = 'Each image must have the same number of selections'
            raise ValueError(msg)

    cam1_coords = np.array(cam1_coords, dtype=np.float64)
    cam2_coords = np.array(cam1_coords, dtype=np.float64)

    if cam1_coords.shape != cam2_coords.shape:
        msg = "Camera coordinate arrays have different dimensions"
        raise ValueError(msg)

    if not cam1_coords[0].shape == (1, 2):
        if cam1_coords[0].shape == (2,):
            print("Attempting to fix coordinate-array by np.expand_dims(<array>, axis=1)")
            cam1_coords = np.expand_dims(cam1_coords, axis=1)
            cam2_coords = np.expand_dims(cam2_coords, axis=1)
        else:
            msg = "Coordinate-array has an invalid format"
            raise ValueError(msg)

    cfg_3d = read_config(config_path)
    img_path, path_corners, path_camera_matrix, path_undistort = auxiliaryfunctions_3d.Foldernames3Dproject(cfg_3d)

    cam_names = cfg_3d['camera_names']
    camera_pair_key = cam_names[0]+'-'+cam_names[1]

    # Create an empty dataFrame to store the undistorted 2d coordinates and likelihood
    stereo_path = os.path.join(path_camera_matrix, 'stereo_params.pickle')
    with open(stereo_path, 'rb') as infile:
        stereo_file = pickle.load(infile)

    mtx_l = stereo_file[camera_pair_key]['cameraMatrix1']
    dist_l = stereo_file[camera_pair_key]['distCoeffs1']

    mtx_r = stereo_file[camera_pair_key]['cameraMatrix2']
    dist_r = stereo_file[camera_pair_key]['distCoeffs2']

    R1 = stereo_file[camera_pair_key]['R1']
    P1 = stereo_file[camera_pair_key]['P1']

    R2 = stereo_file[camera_pair_key]['R2']
    P2 = stereo_file[camera_pair_key]['P2']

    cam1_undistorted_coords = cv2.undistortPoints(
        src=cam1_coords, cameraMatrix=mtx_l, distCoeffs=dist_l, P=P1, R=R1
    )
    cam2_undistorted_coords = cv2.undistortPoints(
        src=cam2_coords, cameraMatrix =mtx_r, distCoeffs = dist_r,P=P2,R=R2
    )
    homogenous_coords = auxiliaryfunctions_3d.triangulatePoints(P1, P2, cam1_undistorted_coords, cam2_undistorted_coords)
    triangulated_coords = np.array((homogenous_coords[0], homogenous_coords[1], homogenous_coords[2])).T

    if unit_keys is not None:
        return {label: coord for label, coord in zip(unit_keys, triangulated_coords)}
    else:
        return triangulated_coords