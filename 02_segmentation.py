# %%

import open3d as o3d
import numpy as np
import pandas as pd

from utils import get_colors_for_labels, get_pc_obj, get_pc_obj_from_df, get_points_df

# %%

points_df = pd.read_csv('data/area_points.csv')
pcd_od3 = get_pc_obj_from_df(points_df)
o3d.visualization.draw_geometries([pcd_od3])

# %% perform image segmentation

epsilon = 2
min_cluster_points = 100

db_labels = np.array(pcd_od3.cluster_dbscan(eps=epsilon, min_points=min_cluster_points))
max_labels = db_labels.max()
max_labels


# %% visualize

label_colors = get_colors_for_labels(db_labels)
xyz = points_df.loc[:, ['x','y','z']].values

pcd_od3 = get_pc_obj(xyz, label_colors)
o3d.visualization.draw_geometries([pcd_od3])

# %% visualize a single class

select_idx = 5

segment = pcd_od3.select_by_index(np.where(db_labels == select_idx)[0])

n_points = len(segment.points)
red_color = np.array([[1, 0, 0]] * n_points)
segment.colors = o3d.utility.Vector3dVector(red_color)

non_segment = pcd_od3.select_by_index(np.where(db_labels == select_idx)[0], 
                                    invert=True)
n_points = len(non_segment.points)
gray_color = np.array([[0.2, 0.2, 0.2]] * n_points)
non_segment.colors = o3d.utility.Vector3dVector(gray_color)

o3d.visualization.draw_geometries([segment, non_segment])

