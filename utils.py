import pandas as pd
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def get_points_df(las_file):
    points_df = pd.DataFrame(np.array(las_file.x), columns=['x'])
    points_df['y'] = np.array(las_file.y)
    points_df['z'] = np.array(las_file.z)
    points_df['r'] = np.array(las_file.red) / 65535.0
    points_df['b'] = np.array(las_file.blue) / 65535.0
    points_df['g'] = np.array(las_file.green) / 65535.0
    points_df['label'] = np.array(las_file.classification)
    
    # center points
    points_df['x'] = points_df['x'] - points_df['x'].mean()
    points_df['y'] = points_df['y'] - points_df['y'].mean()
    points_df['z'] = points_df['z'] - points_df['z'].mean()
    
    return points_df

def get_pc_obj(xyz, colors=None):
    
    pcd_od3 = o3d.geometry.PointCloud()
    pcd_od3.points = o3d.utility.Vector3dVector(xyz)
    
    if colors is not None:
        pcd_od3.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd_od3

def get_pc_obj_from_df(points_df):
    # using true colors
    
    xyz = points_df.loc[:, ['x','y','z']].values
    colors = points_df.loc[:, ['r', 'g', 'b']].values
    
    return get_pc_obj(xyz, colors)

def get_colors_for_labels(labels):

    cmap = plt.colormaps.get_cmap('tab10')
    colors = cmap(labels)
    colors = colors[:, :3]
    colors = np.clip(colors, 0, 1)
    
    return colors
