# %% feature computation / visualization
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import pyvista as pv
pv.set_jupyter_backend('trame')

# %%

def compute_eig_values_vectors(points):
    cov_matrix = np.cov(points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]

    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    return sorted_eigenvalues, sorted_eigenvectors


def pca_features(eig_values, eig_vectors):
    
    planarity = (eig_values[1] - eig_values[2])/eig_values[0]
    
    linearity = (eig_values[0] - eig_values[1])/eig_values[0]
    
    omnivariance = (eig_values[0] * eig_values[1] * eig_values[2]) ** (1/3)
    
    normal = eig_vectors[:, 2]
    
    verticality = 1 - np.abs(normal[2])
    
    return planarity, linearity, omnivariance, verticality, normal[0], normal[1], normal[2]

def dist_to_min_and_max(nn_points, this_point):
    d_high = np.max(nn_points[:, 2]) - this_point[2]
    d_low = this_point[2] - np.min(nn_points[:, 2])
    return d_low, d_high

# %%

points_df = pd.read_csv('data/area_points.csv')
points = points_df.loc[:, ['x', 'y', 'z']].values

# %%

point_cloud = pv.PolyData(points)

plotter = pv.Plotter()
plotter.add_mesh(point_cloud, point_size=2,
                 render_points_as_spheres=True,
                 scalars=points_df['z'].values,
                 show_scalar_bar=False
                 )
plotter.show()

# %%

tree = KDTree(points)
tree_2D = KDTree(points[:, 0:2])

# %% get 20 nearest neighbors

dists, indices = tree.query(points, k=20)
neighbors = points[indices]

# %%

idx_2D_radius = tree_2D.query_ball_point(points[:, 0:2], 1)

# %%

sel = 4
this_selection = points[idx_2D_radius[sel]]
np.max(this_selection, axis=0)

# %%

n_points = points.shape[0]

feature_list = ['planarity', 'linearity', 'omnivariance', 'verticality',
                'nx', 'ny', 'nz', 'd_low', 'd_high'
                ]

n_features = len(feature_list)

features_df = pd.DataFrame(np.ones((n_points, n_features)), columns=feature_list)
features_df[:] = np.nan
features_df.head(4)

# %%

for this_idx in np.arange(0, n_points):
    
    this_point = points[this_idx, :]
    
    # get neighboring points
    this_neighbors = neighbors[this_idx]
    this_2D_radius = points[idx_2D_radius[this_idx], :]

    eig_values, eig_vectors = compute_eig_values_vectors(this_neighbors)
    p, l, o, v, nx, ny, nz = pca_features(eig_values, eig_vectors)
    
    d_low = np.nan
    d_high = np.nan
    
    if len(this_2D_radius) > 2:
        d_low, d_high = dist_to_min_and_max(this_2D_radius, this_point)
        
    # store in dataframe
    features_df.loc[this_idx, :] = [p, l, o, v, nx, ny, nz, d_low, d_high]
    
# %%

features_df

# %%

point_cloud = pv.PolyData(points)

plotter = pv.Plotter()
plotter.add_mesh(point_cloud, point_size=2,
                 render_points_as_spheres=True,
                 scalars=features_df['d_high'],
                 show_scalar_bar=False
                 )
plotter.show()

# %%

