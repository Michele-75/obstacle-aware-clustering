"""
Hyperparameter optimization for obstacle-aware clustering.

This module provides tools for tuning the weight parameters
(alpha, beta, gamma) using simulated annealing. The objective
function balances two goals:
  - Geographic coherence (rho_bar): clusters should be spatially tight
  - Attribute separation (sigma_a): clusters should differ meaningfully
    in their attribute distributions

The composite objective is:
    J = rho_bar + (1 - sigma_a)
"""

import numpy as np
from scipy.optimize import dual_annealing
from scipy.stats import kruskal
from itertools import combinations

from .clustering import ObstacleKMeans


def attribute_separation(X, labels, k, attr_indices):
    """
    Measure the fraction of statistically significant pairwise
    attribute differences between clusters.

    Uses the Kruskal-Wallis test for each attribute, followed by
    pairwise comparisons. sigma_a in [0, 1] where 1 means all
    pairwise differences are significant.

    Parameters
    ----------
    X : ndarray of shape (n_samples, d)
        Data matrix.
    labels : ndarray of shape (n_samples,)
        Cluster assignments.
    k : int
        Number of clusters.
    attr_indices : list of int
        Column indices of attribute features in X.

    Returns
    -------
    sigma_a : float
        Fraction of significant pairwise attribute differences.
    details : dict
        Per-attribute significance results.
    """
    if len(attr_indices) == 0:
        return 0.0, {}

    n_pairs = k * (k - 1) // 2
    total_tests = n_pairs * len(attr_indices)
    significant_count = 0
    details = {}

    for attr_idx in attr_indices:
        # Group attribute values by cluster
        groups = [X[labels == i, attr_idx] for i in range(k)]
        groups = [g for g in groups if len(g) > 0]

        attr_detail = {'kruskal_p': None, 'pairwise_significant': []}

        if len(groups) < 2:
            details[attr_idx] = attr_detail
            continue

        # Kruskal-Wallis test (non-parametric ANOVA)
        try:
            stat, p_value = kruskal(*groups)
            attr_detail['kruskal_p'] = p_value

            if p_value < 0.05:
                # Pairwise comparisons
                for (a, b) in combinations(range(len(groups)), 2):
                    if len(groups[a]) > 0 and len(groups[b]) > 0:
                        try:
                            _, p_pair = kruskal(groups[a], groups[b])
                            if p_pair < 0.05:
                                significant_count += 1
                                attr_detail['pairwise_significant'].append((a, b))
                        except ValueError:
                            pass
        except ValueError:
            pass

        details[attr_idx] = attr_detail

    sigma_a = significant_count / total_tests if total_tests > 0 else 0.0
    return sigma_a, details


def objective_function(params, X, t_data, boundary, k=3, n_attr=0,
                       attr_indices=None, random_state=None):
    """
    Compute the composite objective J = rho_bar + (1 - sigma_a).

    Parameters
    ----------
    params : tuple of float
        (alpha, beta, gamma) weight values to evaluate.
    X : ndarray of shape (n_samples, d)
        Data matrix.
    t_data : ndarray of shape (n_samples,)
        Boundary parameter values.
    boundary : Boundary
        The obstacle boundary object.
    k : int
        Number of clusters.
    n_attr : int
        Number of attribute features.
    attr_indices : list of int
        Column indices of attribute features.
    random_state : int or None
        Random seed for reproducibility.

    Returns
    -------
    float
        The objective value (lower is better).
    """
    alpha, beta, gamma = params

    model = ObstacleKMeans(
        k=k, boundary=boundary,
        alpha=alpha, beta=beta, gamma=gamma,
        random_state=random_state, n_attr=n_attr
    )
    model.fit(X, t_data=t_data)

    rho_bar = model.rho_bar_

    # Compute attribute separation
    if attr_indices is not None and len(attr_indices) > 0:
        sigma_a, _ = attribute_separation(X, model.labels_, k, attr_indices)
    else:
        sigma_a = 0.0

    return rho_bar + (1.0 - sigma_a)


def optimize_weights(X, t_data, boundary, k=3, n_attr=0, attr_indices=None,
                     bounds=None, random_state=None, maxiter=1000):
    """
    Find optimal (alpha, beta, gamma) via simulated annealing.

    Parameters
    ----------
    X : ndarray of shape (n_samples, d)
        Data matrix.
    t_data : ndarray of shape (n_samples,)
        Boundary parameter values.
    boundary : Boundary
        The obstacle boundary object.
    k : int
        Number of clusters (default 3).
    n_attr : int
        Number of attribute features.
    attr_indices : list of int
        Column indices of attribute features in X.
    bounds : list of tuple, optional
        Search bounds for (alpha, beta, gamma). Default [(0, 2)] * 3.
    random_state : int or None
        Random seed.
    maxiter : int
        Maximum iterations for simulated annealing.

    Returns
    -------
    result : dict
        Dictionary with keys:
          - 'alpha', 'beta', 'gamma': optimal weights
          - 'objective': optimal objective value
          - 'rho_bar': geographic coherence at optimum
          - 'sigma_a': attribute separation at optimum
          - 'model': the fitted ObstacleKMeans at optimal weights
    """
    if bounds is None:
        bounds = [(0, 2), (0, 2), (0, 2)]

    if attr_indices is None:
        attr_indices = []

    result = dual_annealing(
        objective_function,
        bounds=bounds,
        args=(X, t_data, boundary, k, n_attr, attr_indices, random_state),
        seed=random_state,
        maxiter=maxiter
    )

    opt_alpha, opt_beta, opt_gamma = result.x

    # Refit with optimal weights
    model = ObstacleKMeans(
        k=k, boundary=boundary,
        alpha=opt_alpha, beta=opt_beta, gamma=opt_gamma,
        random_state=random_state, n_attr=n_attr
    )
    model.fit(X, t_data=t_data)

    sigma_a, attr_details = attribute_separation(
        X, model.labels_, k, attr_indices
    )

    return {
        'alpha': opt_alpha,
        'beta': opt_beta,
        'gamma': opt_gamma,
        'objective': result.fun,
        'rho_bar': model.rho_bar_,
        'sigma_a': sigma_a,
        'attr_details': attr_details,
        'model': model,
    }


def objective_surface(X, t_data, boundary, alpha_range, beta_range,
                      gamma=1.0, k=3, n_attr=0, attr_indices=None,
                      random_state=None):
    """
    Evaluate the objective function over a grid of (alpha, beta) values.

    Useful for generating contour plots of the optimization landscape.

    Parameters
    ----------
    X, t_data, boundary, k, n_attr, attr_indices, random_state :
        See objective_function.
    alpha_range : array-like
        Alpha values to evaluate.
    beta_range : array-like
        Beta values to evaluate.
    gamma : float
        Fixed gamma value for the surface.

    Returns
    -------
    alpha_grid : ndarray
        2D grid of alpha values.
    beta_grid : ndarray
        2D grid of beta values.
    obj_grid : ndarray
        2D grid of objective values.
    """
    alpha_grid, beta_grid = np.meshgrid(alpha_range, beta_range)
    obj_grid = np.zeros_like(alpha_grid)

    for i in range(alpha_grid.shape[0]):
        for j in range(alpha_grid.shape[1]):
            a = alpha_grid[i, j]
            b = beta_grid[i, j]
            obj_grid[i, j] = objective_function(
                (a, b, gamma), X, t_data, boundary,
                k=k, n_attr=n_attr, attr_indices=attr_indices,
                random_state=random_state
            )

    return alpha_grid, beta_grid, obj_grid