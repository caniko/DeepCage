from deepcage.compute.test import visualize_workflow, visualize_triangulation, visualize_basis_vectors
import os

root = os.path.realpath('H:/Can_cage/DeepCage_DLC_files/DCage_1')
config_path = os.path.join(root, 'DeepCage_MROS_V1-Can-2019-10-30/config.yaml')

visualize_workflow(config_path)
visualize_triangulation(config_path)
visualize_basis_vectors(config_path)
