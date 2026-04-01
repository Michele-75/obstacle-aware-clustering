# Obstacle-Aware Clustering for Geographic Data

**Clustering around real-world barriers using arc-length parameterization and modified k-Means**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

Standard clustering algorithms like k-Means assume that straight-line (Euclidean) distance reflects true proximity. In geographic settings, this assumption breaks down when obstacles — lakes, rivers, mountain ranges — separate nearby points. Two locations on opposite shores of a lake may appear close in Euclidean space but are far apart by any realistic travel path.

This project develops an **obstacle-aware extension of k-Means** that addresses this problem by:

1. **Parameterizing the obstacle boundary** as a smooth closed curve (analytically for simple shapes, via cubic splines for real-world geometries)
2. **Projecting each data point** onto the boundary and computing a normalized arc-length parameter $s \in [0, 1]$
3. **Defining a loop-aware distance** that respects the circular topology: $d_s(s_1, s_2) = \min(|s_1 - s_2|,\; 1 - |s_1 - s_2|)$
4. **Combining spatial and attribute distances** in a weighted metric that simultaneously optimizes geographic coherence and attribute separation

The method remains fully compatible with the k-Means framework, making it computationally efficient while capturing obstacle geometry that pure Euclidean methods miss.

## Case Studies

### 1. Synthetic Ellipse (Proof of Concept)
A controlled experiment with generated data around an elliptical obstacle. Demonstrates that standard k-Means incorrectly clusters across the barrier, while the arc-length–augmented version produces geographically coherent clusters.

### 2. Lake Tahoe Hotels
Real-world application clustering hotels around Lake Tahoe. The lake boundary is extracted programmatically and parameterized with cubic splines. Clusters jointly optimize spatial compactness and separation in hotel attributes (rating, price).

### 3. Lake Tahoe Environmental Data
Application to environmental monitoring data around Lake Tahoe, demonstrating the method's relevance to scientific research contexts such as water quality monitoring and ecological assessment.

## Repository Structure

```
obstacle-aware-clustering/
│
├── README.md                          # This file
├── environment.yml                    # Conda environment specification
├── LICENSE                            # MIT License
│
├── notebooks/
│   ├── 01_toy_problem_ellipse.ipynb   # Synthetic ellipse case study
│   ├── 02_lake_boundary.ipynb         # Lake Tahoe boundary extraction & parameterization
│   ├── 03_clustering_hotels.ipynb     # Hotel clustering around Lake Tahoe
│   ├── 04_clustering_environment.ipynb# Environmental data clustering
│   └── 05_results_comparison.ipynb    # Cross-case-study results and interactive maps
│
├── src/
│   └── obstacle_clustering/
│       ├── __init__.py                # Package init
│       ├── boundary.py                # Boundary parameterization (ellipse + spline)
│       ├── distance.py                # Distance metrics (Euclidean, loop-aware, weighted)
│       ├── clustering.py              # Obstacle-aware k-Means algorithm
│       ├── optimization.py            # Hyperparameter tuning (simulated annealing)
│       └── visualization.py           # Plotting and interactive map utilities
│
├── data/
│   ├── raw/                           # Original data files (not modified)
│   ├── processed/                     # Cleaned and normalized datasets
│   └── boundaries/                    # Obstacle boundary coordinates
│
├── figures/                           # Saved plots and maps
│
└── docs/
    └── report.pdf                     # Project writeup
```

## Getting Started

### Prerequisites

- Python 3.10+
- Conda (recommended) or pip
- ArcGIS Online account (optional — needed only for ArcGIS-based boundary extraction)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/obstacle-aware-clustering.git
cd obstacle-aware-clustering

# Create and activate the conda environment
conda env create -f environment.yml
conda activate obstacle-clustering

# Install the local package in development mode
pip install -e .
```

### Running the Notebooks

Launch Jupyter and run the notebooks in order:

```bash
jupyter lab
```

1. **`01_toy_problem_ellipse.ipynb`** — Start here to understand the method
2. **`02_lake_boundary.ipynb`** — Extract and parameterize the Lake Tahoe boundary
3. **`03_clustering_hotels.ipynb`** — Cluster hotels around the lake
4. **`04_clustering_environment.ipynb`** — Environmental data application
5. **`05_results_comparison.ipynb`** — Compare results and view interactive maps

## Method Summary

### Distance Metric

Each data point is represented as $(x, y, s, a_1, a_2, \ldots)$ where $(x,y)$ are geographic coordinates, $s$ is the normalized arc-length position, and $a_i$ are attribute features. The weighted distance is:

$$d^2(\mathbf{x}_i, \mathbf{c}_j) = \alpha^2 \| (x_i, y_i) - (c_{jx}, c_{jy}) \|^2 + \beta^2 \, d_s(s_i, s_{cj})^2 + \gamma^2 \sum_m (a_{im} - a_{cjm})^2$$

### Centroid Update

- $(x, y)$ and attribute centroids are updated by standard averaging
- The $s$ centroid is recomputed by projecting back onto the boundary curve, respecting the circular topology

### Hyperparameter Optimization

Weights $(\alpha, \beta, \gamma)$ are tuned via simulated annealing to minimize a composite objective:

$$J = \bar{\rho} + (1 - \sigma_a)$$

where $\bar{\rho}$ measures within-cluster geographic distortion and $\sigma_a$ measures the fraction of statistically significant attribute differences between clusters.

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `numpy`, `scipy` | Numerical computation, optimization, spline fitting |
| `scikit-learn` | Preprocessing, baseline k-Means comparison |
| `matplotlib`, `seaborn` | Static visualizations |
| `folium` | Interactive maps |
| `arcgis` | Boundary extraction from ArcGIS Living Atlas |
| `osmnx` / `shapely` | Alternative boundary extraction (OpenStreetMap) |
| `geopandas` | Geospatial data handling |
| `jupyter` | Interactive notebooks |

## Author

**Michele Perry**
M.S. Applied Mathematics

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- Supervised by [Professor Name], [University Name]
- Lake Tahoe boundary data from [ArcGIS Living Atlas / OpenStreetMap]
- Hotel data collected via Google Maps Places API
