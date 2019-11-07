import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle
import gc
import os

from deepcage.project.edit import read_config

from .detect import detect_bonsai, detect_cage_calibration_images
from .constants import CAMERAS


def get_title(camera_name, axis_name, input_istip, direction):
    return '{camera_name}\nClick on {} tip of the {} on the {} side'.format(
        'the' if input_istip else 'a point an decrement from\nthe',
        axis_name, direction, camera_name=camera_name
    )
    

def get_coord(cam_image, n=-1, title=None):
    '''
    Helper function for triangulate_raw_2d_camera_coords.
    User manually selects points on the provided images
    
    Parameters
    ----------
    cam_image : string; default None
        Full path of the image from camera as a string.
    cam2_image : string; default None
        Full path of the image of camera 2 as a string.
    '''

    plt.imshow(mpimg.imread(cam_image))
    if title is not None:
        plt.title(title)

    pick = plt.ginput(n=n, timeout=-1, show_clicks=True)[0]

    return pick


def basis_label(config_path, image_paths=None, decrement=False):
    '''
    Parameters
    ----------
    config_path : string
        String containing the full path of the project config.yaml file.
    image_paths : dict; optional
        Dictionary where the key is name of the camera, and the value is the full path to the image
        of the referance points taken with the camera
    '''
    if image_paths is None:
        camera_images = detect_cage_calibration_images(config_path)

    n = -1
    axis_vectors = dict.fromkeys(CAMERAS)
    for camera, axis in CAMERAS.items():
        cam_img = camera_images[camera]

        if decrement is True:
            axis_vectors[camera] = (
                {direction: [get_coord(cam_img, n=n, title=get_title(camera, axis[0][0], istip, direction)) for istip in (True, False)] for direction in ('positive', 'negative')},
                [get_coord(cam_img, n=n, title=get_title(camera, axis[1][0], istip, axis[1][1])) for istip in (True, False)],
                [get_coord(cam_img, n=n, title=get_title(camera, 'z-axis', istip, 'positive')) for istip in (True, False)]
            )
        else:
            axis_vectors[camera] = (
                {direction: get_coord(cam_img, n=n, title=get_title(camera, axis[0][0], True, direction)) for direction in ('positive', 'negative')},
                get_coord(cam_img, n=n, title=get_title(camera, axis[1][0], True, axis[1][1])),
                get_coord(cam_img, n=n, title=get_title(camera, 'z-axis', True, 'positive')),
                get_coord(cam_img, n=n, title='Select origin')
            )

        plt.close()
        gc.collect()

    data_path = os.path.join(read_config(config_path)['data_path'], 'labels.pickle')
    with open(data_path, 'wb') as outfile:
        stereo_file = pickle.dump(axis_vectors, outfile)

    return axis_vectors


def stereo_cam_info(axis_vectors, pair, decrement=False):
    '''
    Using the labels from basis_label create 2d representations of the basis vectors for triangulation
    
    '''
    cam1, cam2 = pair
    print('Calculating the basis vectors of %s %s' % pair)
    # i, ii, iii = CAMERAS[cam1][0][0], CAMERAS[cam1][1][0], 'z-axis'

    # Preparing for triangualting image coordinates
    raw_cam1v = axis_vectors[cam1]
    raw_cam2v = axis_vectors[cam2]

    if decrement is True:
        if CAMERAS[cam1][2] == CAMERAS[cam2][2]:
            assert CAMERAS[cam1] == CAMERAS[cam2], '%s != %s' % (CAMERAS[cam1], CAMERAS[cam2])
            unit_keys = ((CAMERAS[cam1][0][0], 'positive'), CAMERAS[cam1][1], 'z-axis')

            cam1v = (
                raw_cam1v[0]['positive'][0], raw_cam1v[0]['negative'][0],   # NorthNorth: x-axis
                raw_cam1v[1][0],                                            # NorthNorth: y-axis
                raw_cam1v[2][0]                                             # z-axis
            )
            cam2v = (
                raw_cam2v[0]['positive'][0], raw_cam2v[0]['negative'][0],
                raw_cam2v[1][0],
                raw_cam2v[2][0]
            )
        else:
            # Corner
            assert CAMERAS[cam1][1][0] == CAMERAS[cam2][0][0], '%s != %s' % (CAMERAS[cam1][1][0], CAMERAS[cam2][0][0])
            assert CAMERAS[cam1][0][0] == CAMERAS[cam2][1][0], '%s != %s' % (CAMERAS[cam1][0][0], CAMERAS[cam2][1][0])
            unit_keys = (CAMERAS[cam1][1], CAMERAS[cam2][1], 'z-axis')

            cam1v = (
                raw_cam1v[1][0], raw_cam1v[1][1],                                                   # EastSouth: x positive
                raw_cam1v[0][ CAMERAS[cam2][1][1] ][0], raw_cam1v[0][ CAMERAS[cam2][1][1] ][1],     # EastSouth: y negative
                raw_cam1v[2][0], raw_cam1v[2][1]
            )
            cam2v = (
                raw_cam2v[0][ CAMERAS[cam1][1][1] ][0], raw_cam2v[0][ CAMERAS[cam1][1][1] ][1],     # EastSouth: x positive
                raw_cam2v[1][0], raw_cam2v[1][1],                                                   # EastSouth: y negative
                raw_cam2v[2][0], raw_cam2v[2][1]
            )
    else:
        if CAMERAS[cam1][2] == CAMERAS[cam2][2]:
            assert CAMERAS[cam1] == CAMERAS[cam2], '%s != %s' % (CAMERAS[cam1], CAMERAS[cam2])
            unit_keys = ((CAMERAS[cam1][0][0], 'positive'), CAMERAS[cam1][1], 'z-axis')

            cam1v = (
                raw_cam1v[0]['positive'], raw_cam1v[0]['negative'],     # NorthNorth: x-axis
                raw_cam1v[1],                                           # NorthNorth: y-axis
                raw_cam1v[2]                                            # z-axis
            )
            cam2v = (
                raw_cam2v[0]['positive'], raw_cam2v[0]['negative'],
                raw_cam2v[1],
                raw_cam2v[2]
            )
        else:
            # Corner
            assert CAMERAS[cam1][1][0] == CAMERAS[cam2][0][0], '%s != %s' % (CAMERAS[cam1][1][0], CAMERAS[cam2][0][0])
            assert CAMERAS[cam1][0][0] == CAMERAS[cam2][1][0], '%s != %s' % (CAMERAS[cam1][0][0], CAMERAS[cam2][1][0])
            unit_keys = (CAMERAS[cam1][1], CAMERAS[cam2][1], 'z-axis')

            cam1v = (
                raw_cam1v[1],                           # EastSouth: x positive
                raw_cam1v[0][ CAMERAS[cam2][1][1] ],    # EastSouth: y negative
                raw_cam1v[2],
            )
            cam2v = (
                raw_cam2v[0][ CAMERAS[cam1][1][1] ],     # EastSouth: x positive
                raw_cam2v[1],                            # EastSouth: y negative
                raw_cam2v[2]
            )

    dlc3d_cfgs = get_dlc3d_configs(config_path)
    trian = triangulate_raw_2d_camera_coords(
        dlc3d_cfgs[pair], cam1_coords=cam1v, cam2_coords=cam2v
    )

    if decrement is True:
        if CAMERAS[cam1][2] == CAMERAS[cam2][2]:
            if CAMERAS[cam1][0][1] == 'close':
                origin = trian[0] + (trian[1] - trian[0]) / 2    # pos + (trian[1] - pos) / 2
                z_axis = unit_vector(trian[3] - origin)
                axis_1st = unit_vector(trian[0] - origin)
                axis_2nd = unit_vector(np.cross(axis_1st, z_axis))
            else:
                origin = trian[1] + (trian[0] - trian[1]) / 2
                z_axis = unit_vector(trian[3] - origin)
                axis_1st = unit_vector(origin - trian[1])
                axis_2nd = unit_vector(np.cross(z_axis, axis_1st))

            if CAMERAS[cam2][1][1] == 'positive':
                alt_axis_2nd = origin - trian[2]
            else:
                alt_axis_2nd = trian[2] - origin

        else:
            # Corner
            first_axis_linev = unit_vector(trian[1] - trian[0])
            z_axis_linev = unit_vector(trian[5] - trian[4])
            tangent_v = unit_vector(np.cross(z_axis_linev, first_axis_linev))

            LHS = np.array((first_axis_linev, -z_axis_linev, tangent_v)).T
            RHS = trian[4] - trian[0]
            sol = solve(LHS, RHS)

            origin = trian[0] + first_axis_linev * sol[0] + tangent_v * sol[2]/2
            z_axis = unit_vector(trian[4] - origin)

            if CAMERAS[cam1][1][1] == 'positive':
                axis_1st = unit_vector(trian[0] - origin)
                axis_2nd = unit_vector(np.cross(axis_1st, z_axis))
            else:
                axis_1st = unit_vector(origin - trian[0])
                axis_2nd = unit_vector(np.cross(z_axis, axis_1st))

            if CAMERAS[cam2][1][1] == 'positive':
                alt_axis_2nd = trian[2] - origin
            else:
                alt_axis_2nd = origin - trian[2]

        axis_2nd_name = 'y-axis' if CAMERAS[cam1][0][0] == 'x-axis' else 'x-axis'
        stereo_cam_unit = {
            CAMERAS[cam1][0][0]: axis_1st,
            (axis_2nd_name, 'cross'): axis_2nd,
            (axis_2nd_name, 'alt'): alt_axis_2nd,
            'z-axis': z_axis
        }
        
        orig_map = {
            'origin': origin,
            'map': np.array((axis_1st, axis_2nd, z_axis)).T
        }

        print('\nCross product derived {axis_2nd_name}: {cross}\n' \
        'Origin-subtraction derived {axis_2nd_name}: {orig}\n' \
        'Ration between cross and origing: {ratio}\n'.format(
            axis_2nd_name=axis_2nd_name, cross=axis_2nd,
            orig=alt_axis_2nd, ratio=axis_2nd / alt_axis_2nd
        ))
        
        return stereo_cam_unit, orig_map
    return cam1v, cam2v


def alter_basis_label(config_path, camera, index=None, image_paths=None):
    data_path = os.path.join(read_config(config_path)['data_path'], 'labels.pickle')
    with open(data_path, 'rb') as infile:
        axis_vectors = pickle.load(infile)
        
    if image_paths is None:
        camera_images = detect_cage_calibration_images(config_path)
