

PAIRS = 8

CAMERAS = {
    'NorthWest': (('x-axis', 'close'), ('y-axis', 'positive'), 1),  'NorthEast': (('x-axis', 'close'), ('y-axis', 'positive'), 1),
    'EastNorth': (('y-axis', 'far'),  ('x-axis', 'positive'), 2),   'EastSouth': (('y-axis', 'far'),  ('x-axis', 'positive'), 2),
    'SouthEast': (('x-axis', 'far'), ('y-axis', 'negative'), 3),    'SouthWest': (('x-axis', 'far'), ('y-axis', 'negative'), 3),
    'WestSouth': (('y-axis', 'close'),  ('x-axis', 'negative'), 4), 'WestNorth': (('y-axis', 'close'),  ('x-axis', 'negative'), 4)
}

PAIR_IDXS = {
    ('NorthWest', 'NorthEast'): 0, ('NorthEast', 'EastNorth'): 1, ('EastNorth', 'EastSouth'): 2, ('EastSouth', 'SouthEast'): 3,
    ('SouthEast', 'SouthWest'): 4, ('SouthWest', 'WestSouth'): 5, ('WestSouth', 'WestNorth'): 6, ('WestNorth', 'NorthWest'): 7
}

# 'CameraName': (
    # (Axis with visable negative and positive side, Location of positive side relative to origin and new origin),
    # (Axis with one side, direction visable), Direction of the visable side),
    # North->1; East->2; South->3; West->4
# )


def get_pairs(camera_names=tuple(CAMERAS.keys())):
    num = len(camera_names)
    pairs = []
    for i in range(num):
        cam1 = camera_names[i]
        cam2 = camera_names[i+1] if i != num-1 else camera_names[0]
        pair = (cam1, cam2)
        pairs.append(pair)
        
    return pairs

def pair_cycler(idx, pairs=get_pairs()):
    from itertools import cycle

    circular_linked_list_iterator = cycle(pairs)
    for n in range(idx):
        pair = next(circular_linked_list_iterator)

    return pair
