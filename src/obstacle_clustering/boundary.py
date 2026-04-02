"""
Boundary representations for obstacle-aware clustering.

This module provides classes for representing obstacle boundaries as
parameterized closed curves. Each boundary supports:
  - Evaluating points on the curve: (x(t), y(t))
  - Computing derivatives: (dx/dt, dy/dt)
  - Projecting external points onto the nearest boundary location
  - Computing normalized arc-length parameter s in [0, 1]

Two implementations are provided:
  - EllipseBoundary: Analytical ellipse (for the synthetic toy problem)
  - SplineBoundary:  Cubic spline fit to coordinate data (for real-world lakes)
"""

import numpy as np
from abc import ABC, abstractmethod
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, fsolve


class Boundary(ABC):
    """
    Abstract base class for a closed-curve obstacle boundary.

    Every boundary is parameterized by t in [t_min, t_max], where the
    endpoints correspond to the same physical location (closed loop).
    The normalized arc-length parameter s in [0, 1] records a point's
    fractional position along the total perimeter.
    """

    @abstractmethod
    def evaluate(self, t):
        """Return (x, y) coordinates on the boundary at parameter t."""
        pass

    @abstractmethod
    def derivative(self, t):
        """Return (dx/dt, dy/dt) at parameter t."""
        pass

    @abstractmethod
    def t_range(self):
        """Return (t_min, t_max) for the parameterization domain."""
        pass

    def arc_length_integrand(self, t):
        """Compute ds/dt = sqrt((dx/dt)^2 + (dy/dt)^2) at parameter t."""
        dx, dy = self.derivative(t)
        return np.sqrt(dx**2 + dy**2)

    def total_arc_length(self):
        """Compute the total perimeter of the boundary by numerical integration."""
        t_min, t_max = self.t_range()
        length, _ = quad(self.arc_length_integrand, t_min, t_max, limit=1000)
        return length

    def arc_length_at(self, t):
        """Compute the arc length from t_min to a given parameter value t."""
        t_min, _ = self.t_range()
        s, _ = quad(self.arc_length_integrand, t_min, t, limit=1000)
        return s

    def normalized_arc_length(self, t):
        """
        Compute s in [0, 1], the normalized arc-length position.

        s = (arc length from t_min to t) / (total perimeter)
        """
        L = self.total_arc_length()
        return self.arc_length_at(t) / L

    def project_point(self, x_p, y_p):
        """
        Find the closest point on the boundary to an external point (x_p, y_p).

        Returns
        -------
        t_closest : float
            The parameter value of the closest boundary point.
        s_closest : float
            The normalized arc-length at that point.
        """
        # Initial guess: evaluate boundary at many points, pick closest
        t_min, t_max = self.t_range()
        t_samples = np.linspace(t_min, t_max, 1000)
        xy_samples = np.array([self.evaluate(t) for t in t_samples])
        distances = np.sqrt((xy_samples[:, 0] - x_p)**2 +
                            (xy_samples[:, 1] - y_p)**2)
        t_guess = t_samples[np.argmin(distances)]

        # Refine using the orthogonality condition:
        # (point - curve(t)) dot curve'(t) = 0
        def orthogonality(t):
            x_t, y_t = self.evaluate(t)
            dx_dt, dy_dt = self.derivative(t)
            return (x_p - x_t) * dx_dt + (y_p - y_t) * dy_dt

        t_closest = fsolve(orthogonality, t_guess, full_output=False)[0]

        # Clamp to valid range
        t_closest = np.clip(t_closest, t_min, t_max)

        # Compute normalized arc length
        L = self.total_arc_length()
        s_closest = self.arc_length_at(t_closest) / L

        return t_closest, s_closest

    def project_centroid(self, t_values):
        """
        Compute a boundary-aware centroid for a set of parameter values.

        Instead of averaging s directly (which fails for circular data),
        this method:
          1. Maps each t to its (x, y) on the boundary
          2. Averages those (x, y) positions
          3. Projects the average back onto the boundary

        Parameters
        ----------
        t_values : array-like
            Parameter values for cluster members.

        Returns
        -------
        t_avg : float
            Parameter value of the projected centroid.
        s_avg : float
            Normalized arc-length of the projected centroid.
        """
        xy_on_boundary = np.array([self.evaluate(t) for t in t_values])
        x_avg = np.mean(xy_on_boundary[:, 0])
        y_avg = np.mean(xy_on_boundary[:, 1])

        t_avg, s_avg = self.project_point(x_avg, y_avg)
        return t_avg, s_avg

    def sample_boundary(self, n_points=500):
        """
        Generate evenly spaced points along the boundary for plotting.

        Returns
        -------
        xy : ndarray of shape (n_points, 2)
            Coordinates of boundary sample points.
        """
        t_min, t_max = self.t_range()
        t_vals = np.linspace(t_min, t_max, n_points)
        return np.array([self.evaluate(t) for t in t_vals])


class EllipseBoundary(Boundary):
    """
    Elliptical obstacle boundary derived from a Gaussian level set.

    The boundary is the level set of a 2D Gaussian at threshold c:
        A * exp(-((x-h)^2 / (2*sigma_x^2) + (y-k)^2 / (2*sigma_y^2))) = c

    This yields the parametric ellipse:
        x(t) = sigma_x * sqrt(-2*log(c/A)) * cos(t) + h
        y(t) = sigma_y * sqrt(-2*log(c/A)) * sin(t) + k

    for t in [0, 2*pi].

    Parameters
    ----------
    sigma_x : float
        Horizontal scale parameter.
    sigma_y : float
        Vertical scale parameter.
    h, k : float
        Center coordinates of the ellipse.
    c : float
        Level-set threshold (default 0.01).
    A : float
        Gaussian amplitude (default 1.0).
    """

    def __init__(self, sigma_x=3.0, sigma_y=0.6, h=6.0, k=0.0,
                 c=0.01, A=1.0):
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.h = h
        self.k = k
        self.c = c
        self.A = A
        self.sqrt_term = np.sqrt(-2 * np.log(c / A))

        # Cache total arc length (computed once)
        self._total_length = None

    def evaluate(self, t):
        x = self.sigma_x * self.sqrt_term * np.cos(t) + self.h
        y = self.sigma_y * self.sqrt_term * np.sin(t) + self.k
        return np.array([x, y])

    def derivative(self, t):
        dx = -self.sigma_x * self.sqrt_term * np.sin(t)
        dy = self.sigma_y * self.sqrt_term * np.cos(t)
        return np.array([dx, dy])

    def t_range(self):
        return (0.0, 2 * np.pi)

    def total_arc_length(self):
        if self._total_length is None:
            self._total_length = super().total_arc_length()
        return self._total_length

    def project_point(self, x_p, y_p):
        """Optimized projection for the ellipse using atan2 initial guess."""
        delta_x = (x_p - self.h) / (self.sigma_x * self.sqrt_term)
        delta_y = (y_p - self.k) / (self.sigma_y * self.sqrt_term)
        t_guess = np.arctan2(delta_y, delta_x) % (2 * np.pi)

        def orthogonality(t):
            x_t, y_t = self.evaluate(t)
            dx_dt, dy_dt = self.derivative(t)
            return (x_p - x_t) * dx_dt + (y_p - y_t) * dy_dt

        t_closest = fsolve(orthogonality, t_guess, full_output=False)[0]
        t_closest = t_closest % (2 * np.pi)

        L = self.total_arc_length()
        s_closest = self.arc_length_at(t_closest) / L
        return t_closest, s_closest

    def project_centroid(self, t_values):
        """Optimized centroid projection using minimize_scalar."""
        xy_on_boundary = np.array([self.evaluate(t) for t in t_values])
        x_avg = np.mean(xy_on_boundary[:, 0])
        y_avg = np.mean(xy_on_boundary[:, 1])

        delta_x = (x_avg - self.h) / (self.sigma_x * self.sqrt_term)
        delta_y = (y_avg - self.k) / (self.sigma_y * self.sqrt_term)
        t_guess = np.arctan2(delta_y, delta_x) % (2 * np.pi)

        result = minimize_scalar(
            lambda t: np.linalg.norm(
                self.evaluate(t) - np.array([x_avg, y_avg])
            ),
            bounds=(t_guess - np.pi / 4, t_guess + np.pi / 4),
            method='bounded'
        )
        t_avg = result.x
        L = self.total_arc_length()
        s_avg = self.arc_length_at(t_avg) / L
        return t_avg, s_avg


class SplineBoundary(Boundary):
    """
    Boundary represented by cubic splines fitted to coordinate data.

    Used for real-world obstacles (e.g., Lake Tahoe) where the boundary
    is known only as a sequence of (x, y) points.

    Parameters
    ----------
    x_coords : array-like
        x-coordinates (e.g., longitude) of boundary points, ordered
        sequentially around the perimeter.
    y_coords : array-like
        y-coordinates (e.g., latitude) of boundary points.
    """

    def __init__(self, x_coords, y_coords):
        from scipy.interpolate import CubicSpline

        self.x_coords = np.asarray(x_coords)
        self.y_coords = np.asarray(y_coords)

        # Create parameter t in [0, 1] based on point ordering
        t = np.linspace(0, 1, len(self.x_coords))

        # Fit cubic splines to x(t) and y(t)
        self.spline_x = CubicSpline(t, self.x_coords)
        self.spline_y = CubicSpline(t, self.y_coords)

        # Cache total arc length
        self._total_length = None

    def evaluate(self, t):
        return np.array([float(self.spline_x(t)), float(self.spline_y(t))])

    def derivative(self, t):
        return np.array([float(self.spline_x(t, 1)), float(self.spline_y(t, 1))])

    def t_range(self):
        return (0.0, 1.0)

    def total_arc_length(self):
        if self._total_length is None:
            self._total_length = super().total_arc_length()
        return self._total_length

    @classmethod
    def from_file(cls, filepath):
        """
        Load boundary coordinates from a text file.

        Supports two formats:
          - Two columns (space-separated, no header): latitude longitude
          - CSV with header: x, y columns

        Parameters
        ----------
        filepath : str or Path
            Path to the coordinate file.

        Returns
        -------
        SplineBoundary
            A new boundary instance fitted to the coordinates.
        """
        try:
            coords = np.loadtxt(filepath)
            x, y = coords[:, 1], coords[:, 0]  # lon=x, lat=y
        except (ValueError, IndexError):
            coords = np.loadtxt(filepath, delimiter=',', skiprows=1)
            x, y = coords[:, 0], coords[:, 1]

        return cls(x, y)