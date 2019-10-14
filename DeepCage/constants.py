

PAIRS = 8

CAMERAS = {
        'NorthWest': (('x-axis', 'close'), ('y-axis', 'positive'), 1),  'NorthEast': (('x-axis', 'close'), ('y-axis', 'positive'), 1),
        'EastNorth': (('y-axis', 'far'),  ('x-axis', 'positive'), 2),   'EastSouth': (('y-axis', 'far'),  ('x-axis', 'positive'), 2),
        'SouthEast': (('x-axis', 'far'), ('y-axis', 'negative'), 3),    'SouthWest': (('x-axis', 'far'), ('y-axis', 'negative'), 3),
        'WestSouth': (('y-axis', 'close'),  ('x-axis', 'negative'), 4), 'WestNorth': (('y-axis', 'close'),  ('x-axis', 'negative'), 4)
}

# 'CameraName': (
    # (Axis with visable negative and positive side, Location of positive side relative to origin and new origin),
    # (Axis with one side, direction visable), Direction of the visable side),
    # North->1; East->2; South->3; West->4
# )