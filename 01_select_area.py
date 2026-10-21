# %%
import open3d as o3d
import laspy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils import get_colors_for_labels, get_pc_obj, get_pc_obj_from_df, get_points_df

# %% Notes

# Point data was classified as:
#
# 1 Unclassified;
# 2 Bare-earth and low grass;
# 3 Low vegetation (height <2m);
# 4 High vegetation (height >2m);
# 5 Water;
# 6 Buildings;
# 7 Other; and
# 8 Noise (noise points, blunders, outliners, etc)


# %%

# Open the LAS file
las_file = laspy.read("data/492000_5457000.las")

# get the CRS
# crs = las_file.vlrs[2].string

# %% get data as DataFrame

points_df = get_points_df(las_file)
ss_points_df = points_df.sample(10_000, replace=False, random_state=1234)
points_df.head(5)

# %% visualize with colors according to labels

label_colors = get_colors_for_labels(points_df['label'].values)
xyz = points_df.loc[:, ['x','y','z']].values

pcd_od3 = get_pc_obj(xyz, label_colors)
o3d.visualization.draw_geometries([pcd_od3])

# %% visualize true colors

pcd_od3 = get_pc_obj_from_df(points_df)
o3d.visualization.draw_geometries([pcd_od3])

# %%

nn_distances = pcd_od3.compute_nearest_neighbor_distance()
avg_nn_distance = np.mean(nn_distances)
avg_nn_distance

# %% interactively select a point

ss_pcd_od3 = get_pc_obj_from_df(ss_points_df)

# Create a visualizer object
vis = o3d.visualization.VisualizerWithEditing()
vis.create_window()

# Add the point cloud to the visualizer
vis.add_geometry(ss_pcd_od3)

# Run the visualizer (you can select points by clicking on them)
print("Select a point by clicking, then press 'q' to quit.")
vis.run()  # User clicks to select points, coordinates will be saved.

# Close the visualizer window
vis.destroy_window()


# %%

selected_points = vis.get_picked_points()
selected_points

# %% select a vicinity of a given point

# hard-code selected point
selected_idx = 3141

points = points_df.loc[:, ['x','y','z']].values
selected_point = ss_pcd_od3.points[selected_idx]

# Define a radius for vicinity (100 meters)
radius = 100.0

# Compute distances from the selected point to all other points
distances = np.linalg.norm(points - selected_point, axis=1)

# Find the indices of all points within the radius
points_within_radius_indices = np.where(distances <= radius)[0]

# Get the coordinates of points within the 100-meter radius
area_points_df = points_df.loc[points_within_radius_indices]

area_pcd_od3 = get_pc_obj_from_df(area_points_df)
o3d.visualization.draw_geometries([area_pcd_od3])

# %%

area_points_df.to_csv('data/area_points.csv', index=False)

