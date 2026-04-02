"""
Obstacle-Aware Clustering
=========================

A modified k-Means algorithm that respects geographic obstacles
by incorporating arc-length parameterization and loop-aware distances.

Quick Start
-----------
    from obstacle_clustering import ObstacleKMeans, EllipseBoundary

    boundary = EllipseBoundary(sigma_x=3.0, sigma_y=0.6, h=6.0, k=0.0)
    model = ObstacleKMeans(k=3, boundary=boundary, alpha=1.0, beta=1.0)
    model.fit(X, t_data=t_values)

Modules
-------
boundary      : Obstacle boundary representation and parameterization
distance      : Distance metrics (Euclidean, loop-aware, weighted composite)
clustering    : Obstacle-aware k-Means algorithm
optimization  : Hyperparameter tuning via simulated annealing
visualization : Plotting utilities and interactive maps
"""

__version__ = "0.1.0"

# Convenient top-level imports
from .boundary import Boundary, EllipseBoundary, SplineBoundary
from .distance import loop_aware_distance, weighted_distance
from .clustering import ObstacleKMeans
from .optimization import optimize_weights, objective_surface, attribute_separation
from .visualization import plot_clusters, plot_comparison, create_folium_map