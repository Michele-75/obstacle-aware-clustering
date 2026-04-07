"""
Obstacle-aware k-Means clustering algorithm.

This module implements a modified k-Means that incorporates:
  - A loop-aware arc-length distance for obstacle boundaries
  - Weighted composite distances across geographic, arc-length,
    and attribute domains
  - Boundary-aware centroid updates for the s parameter

The algorithm follows the standard k-Means alternating structure
(assign → update → check convergence), but with custom distance
and centroid computations.
"""

import numpy as np
from .distance import weighted_distance, pairwise_weighted_distances, loop_aware_distance
from .boundary import Boundary


class ObstacleKMeans:
    """
    Obstacle-aware k-Means clustering.

    Parameters
    ----------
    k : int
        Number of clusters.
    boundary : Boundary
        An obstacle boundary object (EllipseBoundary or SplineBoundary).
    alpha : float
        Weight for geographic (x, y) distance.
    beta : float
        Weight for arc-length (s) distance.
    gamma : float
        Weight for attribute distance.
    max_iter : int
        Maximum number of iterations (default 100).
    tol : float
        Convergence tolerance on centroid movement (default 1e-4).
    random_state : int or None
        Random seed for reproducibility.
    n_attr : int
        Number of attribute features (default 0, inferred from data).

    Attributes (n_features = number of features per data point)
    ----------
    centroids_ : ndarray of shape (k, n_features)
        Final cluster centroids after fitting.
    labels_ : ndarray of shape (n_samples,)
        Cluster assignment for each data point.
    t_data_ : ndarray of shape (n_samples,)
        Parameter values (t) for each data point's boundary projection.
    n_iter_ : int
        Number of iterations run before convergence.
    rho_bar_ : float
        Mean within-cluster distortion (geographic coherence measure).
    """

    def __init__(self, k=3, boundary=None, alpha=1.0, beta=1.0, gamma=1.0,
                 max_iter=100, tol=1e-4, random_state=None, n_attr=0):
        self.k = k
        self.boundary = boundary
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.n_attr = n_attr

        # Fitted attributes
        self.centroids_ = None
        self.labels_ = None
        self.t_data_ = None
        self.n_iter_ = 0
        self.rho_bar_ = None

    def fit(self, X, t_data=None):
        """
        Run obstacle-aware k-Means on the data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data matrix. Expected layout: [x, y, s, attr_1, attr_2, ...]
        t_data : ndarray of shape (n_samples,), optional
            Parameter values from boundary projection. Required for
            boundary-aware centroid updates.

        Returns
        -------
        self
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape
        self.t_data_ = t_data

        # Infer attribute count
        n_attr = self.n_attr if self.n_attr > 0 else max(0, n_features - 3)

        # Step 1: Initialize centroids with k-means++
        centroids = self._init_centroids_pp(X, n_attr)

        # Step 2: Iterate assign → update
        for iteration in range(self.max_iter):
            # Assign each point to nearest centroid
            labels = self._assign(X, centroids, n_attr)

            # Update centroids
            new_centroids = self._update_centroids(X, labels, centroids, n_attr)

            # Check convergence
            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids

            if shift < self.tol:
                self.n_iter_ = iteration + 1
                break
        else:
            self.n_iter_ = self.max_iter

        self.centroids_ = centroids
        self.labels_ = self._assign(X, centroids, n_attr)

        # Compute final coherence measure
        self.rho_bar_ = self._compute_rho_bar(X, n_attr)

        return self

    def predict(self, X):
        """Assign new data points to the nearest cluster."""
        n_attr = self.n_attr if self.n_attr > 0 else max(0, X.shape[1] - 3)
        return self._assign(X, self.centroids_, n_attr)

    def _init_centroids_pp(self, X, n_attr):
        """
        Initialize centroids using k-means++ strategy.

        The first centroid is chosen randomly, and subsequent centroids
        are selected with probability proportional to squared distance
        from the nearest existing centroid.
        """
        n_samples, n_features = X.shape
        centroids = np.zeros((self.k, n_features))

        # First centroid: random
        centroids[0] = X[np.random.randint(n_samples)]

        for i in range(1, self.k):
            # Compute distance from each point to nearest existing centroid
            D = np.zeros(n_samples)
            for j in range(n_samples):
                dists = [
                    weighted_distance(
                        X[j], centroids[m],
                        alpha=self.alpha, beta=self.beta, gamma=self.gamma,
                        n_attr=n_attr
                    )
                    for m in range(i)
                ]
                D[j] = min(dists)

            # Select next centroid with probability proportional to D^2
            probs = D**2 / np.sum(D**2)
            cumprobs = np.cumsum(probs)
            r = np.random.rand()
            next_idx = np.argmax(cumprobs >= r)
            centroids[i] = X[next_idx]

        return centroids

    def _assign(self, X, centroids, n_attr):
        """Assign each point to the nearest centroid."""
        distances = pairwise_weighted_distances(
            X, centroids,
            alpha=self.alpha, beta=self.beta, gamma=self.gamma,
            n_attr=n_attr
        )
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels, old_centroids, n_attr):
        """
        Update centroid positions.

        - (x, y) and attribute components: standard mean
        - s component: boundary-aware projection via the Boundary object
        """
        n_features = X.shape[1]
        new_centroids = np.zeros_like(old_centroids)

        for i in range(self.k):
            members = np.where(labels == i)[0]

            if len(members) == 0:
                # Keep old centroid if cluster is empty
                new_centroids[i] = old_centroids[i]
                continue

            cluster_data = X[members]

            # Update x, y by simple mean
            new_centroids[i, :2] = np.mean(cluster_data[:, :2], axis=0)

            # Update s via boundary-aware projection
            if n_features > 2 and self.boundary is not None and self.t_data_ is not None:
                t_members = self.t_data_[members]
                _, s_avg = self.boundary.project_centroid(t_members)
                new_centroids[i, 2] = s_avg

            # Update attribute features by simple mean
            attr_start = 3
            if n_features > attr_start:
                new_centroids[i, attr_start:] = np.mean(
                    cluster_data[:, attr_start:], axis=0
                )

        return new_centroids

    def _compute_rho_bar(self, X, n_attr):
        """
        Compute mean within-cluster distortion (rho_bar).

        This measures geographic coherence: how tightly clustered
        points are around their centroids in the (x, y, s) space.

        Returns
        -------
        float
            Average distortion across all clusters.
        """
        rho_total = 0.0

        for i in range(self.k):
            members = np.where(self.labels_ == i)[0]
            if len(members) == 0:
                continue

            x_vals = X[members, 0]
            y_vals = X[members, 1]

            # Geographic component
            geo_distortion = np.mean(
                (x_vals - self.centroids_[i, 0])**2 +
                (y_vals - self.centroids_[i, 1])**2
            )

            # Arc-length component (if present)
            arc_distortion = 0.0
            if X.shape[1] > 2:
                s_vals = X[members, 2]
                s_centroid = self.centroids_[i, 2]
                s_circular = loop_aware_distance(s_vals, s_centroid)
                arc_distortion = np.mean(s_circular**2)

            rho_total += geo_distortion + arc_distortion

        return rho_total / self.k

    def get_cluster_summary(self, X, feature_names=None):
        """
        Generate a summary of each cluster's composition.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data matrix used for fitting.
        feature_names : list of str, optional
            Names for each feature column.

        Returns
        -------
        list of dict
            One dictionary per cluster with size, centroid, and
            per-feature mean and std.
        """
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        summaries = []
        for i in range(self.k):
            members = np.where(self.labels_ == i)[0]
            cluster_data = X[members]

            summary = {
                'cluster': i,
                'size': len(members),
                'centroid': self.centroids_[i].tolist(),
                'features': {}
            }
            for j, name in enumerate(feature_names):
                summary['features'][name] = {
                    'mean': float(np.mean(cluster_data[:, j])),
                    'std': float(np.std(cluster_data[:, j]))
                }
            summaries.append(summary)

        return summaries