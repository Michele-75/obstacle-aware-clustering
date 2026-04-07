"""
Distance metrics for obstacle-aware clustering.

This module defines:
  - loop_aware_distance: circular distance on s in [0, 1]
  - weighted_distance:   composite metric combining geography, arc-length,
                         and attributes with tunable weights (alpha, beta, gamma)
"""

import numpy as np


def loop_aware_distance(s1, s2):
    """
    Compute the circular distance between two arc-length values.

    Because s in [0, 1] represents position on a closed loop, the distance
    must account for wrap-around. For example, s=0.05 and s=0.95 are
    actually close (distance = 0.10), not far apart (0.90).

    Parameters
    ----------
    s1, s2 : float or ndarray
        Normalized arc-length values in [0, 1].

    Returns
    -------
    float or ndarray
        The shortest distance around the loop.
    """
    diff = np.abs(s1 - s2)
    return np.minimum(diff, 1.0 - diff)


def weighted_distance(point, centroid, alpha=1.0, beta=1.0, gamma=1.0,
                      n_geo=2, n_arc=1, n_attr=0):
    """
    Compute the weighted composite distance between a data point and a centroid.

    The distance combines three domains:
      d^2 = alpha^2 * ||geo||^2 + beta^2 * d_s^2 + gamma^2 * ||attr||^2

    where:
      - geo:  Euclidean distance in (x, y)
      - d_s:  loop-aware distance in s
      - attr: Euclidean distance in attribute features

    The data vector layout is assumed to be:
      [x, y, s, attr_1, attr_2, ...]
       ^^^^  ^  ^^^^^^^^^^^^^^^^^^
       geo  arc      attributes

    Parameters
    ----------
    point : ndarray of shape (d,)
        The data point vector.
    centroid : ndarray of shape (d,)
        The cluster centroid vector.
    alpha : float
        Weight for geographic (x, y) distance.
    beta : float
        Weight for arc-length distance.
    gamma : float
        Weight for attribute distance.
    n_geo : int
        Number of geographic dimensions (default 2 for x, y).
    n_arc : int
        Number of arc-length dimensions (default 1 for s).
    n_attr : int
        Number of attribute dimensions (inferred from vector length if 0).

    Returns
    -------
    float
        The weighted composite distance.
    """
    n_features = len(point)

    # Infer attribute count from vector length
    if n_attr == 0:
        n_attr = n_features - n_geo - n_arc

    # Geographic distance (Euclidean in x, y)
    geo_dist = np.linalg.norm(point[:n_geo] - centroid[:n_geo]) * alpha

    # Arc-length distance (loop-aware)
    arc_dist = 0.0
    if n_arc > 0 and n_features > n_geo:
        s_point = point[n_geo]
        s_centroid = centroid[n_geo]
        arc_dist = loop_aware_distance(s_point, s_centroid) * beta

    # Attribute distance (Euclidean)
    attr_dist = 0.0
    attr_start = n_geo + n_arc
    if n_attr > 0 and n_features > attr_start:
        attr_dist = np.linalg.norm(
            point[attr_start:attr_start + n_attr] -
            centroid[attr_start:attr_start + n_attr]
        ) * gamma

    return np.sqrt(geo_dist**2 + arc_dist**2 + attr_dist**2)


def pairwise_weighted_distances(X, centroids, alpha=1.0, beta=1.0, gamma=1.0,
                                 n_geo=2, n_arc=1, n_attr=0):
    """
    Compute distances from each data point to each centroid.

    Parameters
    ----------
    X : ndarray of shape (n_samples, d)
        Data matrix.
    centroids : ndarray of shape (k, d)
        Centroid matrix.
    alpha, beta, gamma : float
        Domain weights.
    n_geo, n_arc, n_attr : int
        Feature layout (see weighted_distance).

    Returns
    -------
    distances : ndarray of shape (n_samples, k)
        Distance from each point to each centroid.
    """
    n = X.shape[0]
    k = centroids.shape[0]
    distances = np.zeros((n, k))

    for i in range(n):
        for j in range(k):
            distances[i, j] = weighted_distance(
                X[i], centroids[j],
                alpha=alpha, beta=beta, gamma=gamma,
                n_geo=n_geo, n_arc=n_arc, n_attr=n_attr
            )

    return distances