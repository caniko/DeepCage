import numpy as np
import os


def get_closest_idxs(array, values):
    # Courtesy of @anthonybell https://stackoverflow.com/a/46184652
    
    # make sure array is a numpy array
    array = np.asarray(array)

    # get insert positions
    idxs = np.searchsorted(array, values, side="left")

    # find indexes where previous index is closer
    prev_idx_is_less = ( (idxs == len(array))|(np.abs(values - array[np.maximum(idxs-1, 0)]) < np.abs(values - array[np.minimum(idxs, len(array)-1)])) )
    idxs[prev_idx_is_less] -= 1

    return idxs


def encode_video(save_path, img_paths, fps, width=1920, height=1080):
    from tqdm import trange
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(filename=str(save_path), fourcc=fourcc, apiPreference=0, fps=float(fps), frameSize=(width, height))

    for i in trange(len(img_paths), desc='Encoding: %s' % os.path.basename(save_path)):
        img = cv2.imread(img_paths[i])
        video.write(img)

    cv2.destroyAllWindows()
    video.release()
    return True


def jp_encode_video(save_path, img_paths, fps, width=1920, height=1080):
    from tqdm.notebook import trange
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(filename=str(save_path), fourcc=fourcc, apiPreference=0, fps=float(fps), frameSize=(width, height))

    for i in trange(len(img_paths), desc='Encoding: %s' % os.path.basename(save_path)):
        img = cv2.imread(img_paths[i])
        video.write(img)

    cv2.destroyAllWindows()
    video.release()
    return True
