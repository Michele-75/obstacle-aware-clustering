"""
Obstacle-Aware Clustering
=========================

A modified k-Means algorithm that respects geographic obstacles
by incorporating arc-length parameterization and loop-aware distances.

Modules
-------
boundary    : Obstacle boundary representation and parameterization
distance    : Distance metrics (Euclidean, loop-aware, weighted composite)
clustering  : Obstacle-aware k-Means algorithm
optimization: Hyperparameter tuning via simulated annealing
visualization: Plotting utilities and interactive maps
"""

__version__ = "0.1.0"
