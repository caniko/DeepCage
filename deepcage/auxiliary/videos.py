def move_videos_3dlc():
    vid_path = project_path / 'videos'
    name = '%d_%s_%s' % (PAIR_IDXS[pair], cam1, cam2)
    if not os.path.exists(vid_path):
        os.mkdir(vid_path)


def detect_videos(root, vid_format='avi'):
    root = Path(root)
    
    result = {}
    for subdir in glob(str(root)+'/*/'):
        idx, fcam1, fcam2 = os.path.basename(os.path.dirname(subdir)).split('_')
        path = Path(subdir)
        result['%s_%s' % (fcam1, fcam2)] = path / ('%d_%s_%s' % (idx, fcam1, fcam2)) / ('*.%s/' % vid_format)

    return result