"""
Visualization utilities for obstacle-aware clustering.

Provides functions for:
  - Plotting clustered data with obstacle boundary
  - Objective function surface / contour plots
  - Interactive Folium maps for geographic data
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot_clusters(X_original, labels, boundary, k=3, centroids_original=None,
                  title=None, ax=None, figsize=(10, 8)):
    """
    Plot clustered data points in original (x, y) coordinates with
    the obstacle boundary overlaid.

    Parameters
    ----------
    X_original : ndarray of shape (n_samples, 2)
        Original (unscaled) x, y coordinates for plotting.
    labels : ndarray of shape (n_samples,)
        Cluster assignments.
    boundary : Boundary
        The obstacle boundary object (for drawing the outline).
    k : int
        Number of clusters.
    centroids_original : ndarray of shape (k, 2), optional
        Centroid positions in original coordinates.
    title : str, optional
        Plot title.
    ax : matplotlib Axes, optional
        Axes to plot on. Created if None.
    figsize : tuple
        Figure size (only used if ax is None).

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    colors = cm.get_cmap('tab10', k)

    # Plot each cluster
    for i in range(k):
        mask = labels == i
        ax.scatter(
            X_original[mask, 0], X_original[mask, 1],
            c=[colors(i)], label=f'Cluster {i + 1}',
            s=60, edgecolors='white', linewidth=0.5, zorder=3
        )

    # Plot centroids
    if centroids_original is not None:
        ax.scatter(
            centroids_original[:, 0], centroids_original[:, 1],
            c='red', marker='X', s=200, edgecolors='black',
            linewidth=1.5, zorder=4, label='Centroids'
        )

    # Plot boundary
    if boundary is not None:
        boundary_pts = boundary.sample_boundary(n_points=500)
        ax.plot(
            boundary_pts[:, 0], boundary_pts[:, 1],
            color='black', linestyle='--', linewidth=1.5,
            label='Obstacle boundary', zorder=2
        )

    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    return ax


def plot_objective_surface(alpha_grid, beta_grid, obj_grid,
                           optimal_point=None, title=None,
                           ax=None, figsize=(10, 8)):
    """
    Plot a contour map of the objective function over (alpha, beta).

    Parameters
    ----------
    alpha_grid, beta_grid, obj_grid : ndarray
        Output from optimization.objective_surface().
    optimal_point : tuple of (alpha, beta), optional
        Mark the optimal point on the plot.
    title : str, optional
        Plot title.
    ax : matplotlib Axes, optional
    figsize : tuple

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    contour = ax.contourf(
        alpha_grid, beta_grid, obj_grid,
        levels=50, cmap='viridis'
    )
    plt.colorbar(contour, ax=ax, label='Objective Value  J = ρ̄ + (1 − σₐ)')

    if optimal_point is not None:
        ax.scatter(
            optimal_point[0], optimal_point[1],
            c='red', marker='*', s=300, edgecolors='white',
            linewidth=1.5, zorder=5, label='Optimum'
        )
        ax.legend(fontsize=12)

    ax.set_xlabel('α (geographic weight)', fontsize=12)
    ax.set_ylabel('β (arc-length weight)', fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)

    return ax


def plot_comparison(X_original, labels_standard, labels_obstacle,
                    boundary, title_standard="Standard k-Means",
                    title_obstacle="Obstacle-Aware k-Means",
                    figsize=(16, 7)):
    """
    Side-by-side comparison of standard vs. obstacle-aware clustering.

    Parameters
    ----------
    X_original : ndarray of shape (n_samples, 2)
        Original (x, y) coordinates.
    labels_standard : ndarray of shape (n_samples,)
        Labels from standard k-Means.
    labels_obstacle : ndarray of shape (n_samples,)
        Labels from obstacle-aware k-Means.
    boundary : Boundary
        Obstacle boundary.
    title_standard, title_obstacle : str
        Subplot titles.
    figsize : tuple

    Returns
    -------
    fig, axes
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    plot_clusters(X_original, labels_standard, boundary,
                  title=title_standard, ax=axes[0])
    plot_clusters(X_original, labels_obstacle, boundary,
                  title=title_obstacle, ax=axes[1])

    plt.tight_layout()
    return fig, axes


def create_folium_map(latitudes, longitudes, labels, k=3,
                      boundary_coords=None, popup_text=None,
                      tile_style='OpenStreetMap'):
    """
    Create an interactive Folium map showing clustered geographic points.

    Parameters
    ----------
    latitudes : array-like
        Latitude values.
    longitudes : array-like
        Longitude values.
    labels : array-like
        Cluster assignments.
    k : int
        Number of clusters.
    boundary_coords : ndarray of shape (n, 2), optional
        Boundary coordinates as (lat, lon) pairs for drawing the outline.
    popup_text : list of str, optional
        Popup text for each marker.
    tile_style : str
        Folium tile style.

    Returns
    -------
    folium.Map
    """
    import folium

    # Cluster colors
    cluster_colors = ['red', 'blue', 'green', 'purple', 'orange',
                      'darkred', 'darkblue', 'darkgreen', 'cadetblue', 'pink']

    # Center the map
    center_lat = np.mean(latitudes)
    center_lon = np.mean(longitudes)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11,
                   tiles=tile_style)

    # Add markers
    for i, (lat, lon, label) in enumerate(zip(latitudes, longitudes, labels)):
        color = cluster_colors[int(label) % len(cluster_colors)]
        popup = popup_text[i] if popup_text else f"Cluster {int(label) + 1}"
        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=popup
        ).add_to(m)

    # Draw boundary if provided
    if boundary_coords is not None:
        boundary_latlon = [(row[0], row[1]) for row in boundary_coords]
        folium.PolyLine(
            locations=boundary_latlon,
            color='black',
            weight=2,
            dash_array='5, 10'
        ).add_to(m)

    return m