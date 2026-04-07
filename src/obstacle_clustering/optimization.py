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
from scipy.stats import kruskal, rankdata
from itertools import combinations

from .clustering import ObstacleKMeans


def _dunns_test(groups):
    """
    Perform Dunn's post-hoc test for pairwise comparisons after Kruskal-Wallis.

    Uses the standard Dunn's z-test approach with Bonferroni correction:
      1. Rank all observations across all groups together
      2. For each pair of groups, compute the z-statistic based on
         the difference in mean ranks
      3. Apply Bonferroni correction to the p-values

    Parameters
    ----------
    groups : list of ndarray
        Attribute values for each cluster.

    Returns
    -------
    significant_pairs : list of tuple
        Pairs (i, j) where the difference is significant after correction.
    n_tests : int
        Total number of pairwise tests performed.
    """
    from scipy.stats import norm

    k = len(groups)
    n_tests = k * (k - 1) // 2

    # Rank all values together
    all_values = np.concatenate(groups)
    N = len(all_values)
    ranks = rankdata(all_values)

    # Split ranks back into groups
    group_ranks = []
    start = 0
    for g in groups:
        group_ranks.append(ranks[start:start + len(g)])
        start += len(g)

    # Compute mean rank for each group
    mean_ranks = [np.mean(r) for r in group_ranks]
    group_sizes = [len(g) for g in groups]

    # Tied ranks correction factor
    # C = 1 - sum(t^3 - t) / (N^3 - N) where t = number of ties at each rank
    _, tie_counts = np.unique(ranks, return_counts=True)
    tie_correction = 1.0 - np.sum(tie_counts**3 - tie_counts) / (N**3 - N)

    significant_pairs = []

    for (a, b) in combinations(range(k), 2):
        # Z-statistic for the difference in mean ranks
        diff = abs(mean_ranks[a] - mean_ranks[b])
        variance = (N * (N + 1) / 12.0) * (1.0 / group_sizes[a] + 1.0 / group_sizes[b])

        # Apply tie correction
        if tie_correction > 0:
            variance *= tie_correction

        if variance <= 0:
            continue

        z = diff / np.sqrt(variance)

        # Two-tailed p-value
        p_value = 2.0 * (1.0 - norm.cdf(abs(z)))

        # Bonferroni correction: multiply p-value by number of tests
        p_corrected = min(p_value * n_tests, 1.0)

        if p_corrected < 0.05:
            significant_pairs.append((a, b))

    return significant_pairs, n_tests


def attribute_separation(X, labels, k, attr_indices):
    """
    Measure the fraction of statistically significant pairwise
    attribute differences between clusters.

    Uses the Kruskal-Wallis omnibus test for each attribute, followed
    by Dunn's post-hoc test with Bonferroni correction for pairwise
    comparisons. sigma_a in [0, 1] where 1 means all pairwise
    differences are significant.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
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

        # Kruskal-Wallis omnibus test (non-parametric ANOVA)
        try:
            stat, p_value = kruskal(*groups)
            attr_detail['kruskal_p'] = p_value

            if p_value < 0.05:
                # Dunn's post-hoc test with Bonferroni correction
                sig_pairs, _ = _dunns_test(groups)
                significant_count += len(sig_pairs)
                attr_detail['pairwise_significant'] = sig_pairs

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
    X : ndarray of shape (n_samples, n_features)
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
    X : ndarray of shape (n_samples, n_features)
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