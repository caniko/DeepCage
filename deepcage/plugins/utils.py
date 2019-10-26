import numpy as np


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


def assemble_video(save_path, width=1920, height=1080):
    import cv2

    # choose codec according to format needed
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video=cv2.VideoWriter('video.avi', fourcc, 1,(width, height))

    for j in range(0,5):
        img = cv2.imread(str(i)+'.png')
        video.write(img)

    cv2.destroyAllWindows()
    video.release()
