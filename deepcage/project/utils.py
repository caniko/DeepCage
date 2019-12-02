from glob import glob
import os


def png_to_jpg(save_dir, img_paths=None, img_root=None, codec='cv'):
    '''
    Converts .PNG files to .JPEG files, and moves them to the desired path

    Parameters
    ----------
    save_dir : string-like
        Absolute path of the directory where the new images are saved
    img_paths : list-like
        List of absolute paths of images that will be converted
    img_root : string-like
        Absolute path of the root directory shared by the images to be converted
    codec : string; default 'cv'
        The package that will be used for converting the images
    '''

    assert os.path.exists(save_dir), 'Does not exist:\n%s' % save_dir

    if img_paths is None:
        if img_root is None:
            msg = "Either img_paths or img_root has to be defined"
            ValueError(msg)
        img_paths = glob(os.path.join(img_root, '**/*.png'))

    for img in img_paths:
        img_path = os.path.realpath(img)

        if '\\' in img_path:
            separator = '\\'
        elif '/' in img_path:
            separator = '/'

        save_path = os.path.join(save_dir, os.path.basename(img).replace('png', 'jpg'))
        if codec == 'pil':
            from PIL import Image

            im = Image.open(img_path)
            rgb_im = im.convert('RGB')
            rgb_im.save(save_path)

        elif codec == 'cv':
            import cv2

            jpg = cv2.imread(img_path)
            cv2.imwrite(save_path, jpg)

