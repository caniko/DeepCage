import matplotlib.pyplot as plt
from pathlib import Path

from deepcage.plugins.open_field import (
    OE_basis_label, OE_map_experiment,
    OE_visualize_workflow,
    OE_plot_3d_trajectories,
)


ROOT = Path('H:/Can_cage/DeepCageProjects/Projects')
project = ROOT / 'PenTrack'
dc_config = str(project / 'PenTrack-Can-2019-12-04' / 'no_glass' / 'config.yaml')

# OE_basis_label(dc_config, name_pos=2)
# coords = OE_map_experiment(dc_config)
OE_visualize_workflow(dc_config)
res = OE_plot_3d_trajectories(dc_config)
# plt.show()
