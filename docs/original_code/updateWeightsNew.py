# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 15:01:45 2024

@author: mpp24
"""

# updateWeightsNew

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

# Function to calculate x(t) and y(t) for the ellipse
def parametric_ellipse(t, sigma_x, sigma_y, sqrt_term, h, k):
    x_t = sigma_x * sqrt_term * np.cos(t) + h
    y_t = sigma_y * sqrt_term * np.sin(t) + k
    return x_t, y_t

# Arc length differential ds/dt
def ds_dt(t, sigma_x, sigma_y, sqrt_term):
    return np.sqrt((-sigma_x * sqrt_term * np.sin(t))**2 + (sigma_y * sqrt_term * np.cos(t))**2)

# Compute the arc length from t=0 to t=t for each point
def arc_length(t, sigma_x, sigma_y, sqrt_term):
    s, _ = quad(ds_dt, 0, t, args=(sigma_x, sigma_y, sqrt_term), limit=1000)
    return s

# Function to estimate initial t guess based on the position (x_avg, y_avg)
def initial_guess_for_t(x_avg, y_avg, sigma_x, sigma_y, h, k, sqrt_term):
    delta_x = (x_avg - h) / (sigma_x * sqrt_term)
    delta_y = (y_avg - k) / (sigma_y * sqrt_term)
    t_guess = np.arctan2(delta_y, delta_x)
    return t_guess % (2 * np.pi)

def updateWeightsNew(k, XData, IndexSet, alpha, beta, gamma, t_data, c, A, sigma_x, sigma_y, h, k_param, L=38.2575, final_iteration=False):
    c_new = np.zeros((k, XData.shape[1]))
    sqrt_term = np.sqrt(-2 * np.log(c / A))
    rho_j_total = 0

    include_s = XData.shape[1] > 2
    include_h = XData.shape[1] > 3
    include_r = XData.shape[1] > 4

    for i in range(k):
        ClusterIndices = np.where(IndexSet == i)[0]
        NumVecsInCluster = len(ClusterIndices)

        if NumVecsInCluster > 0:
            # Always update x, y centroids
            c_new[i, 0:2] = np.mean(XData[ClusterIndices, 0:2], axis=0)
            
            if include_s:
                c_new[i, 2] = np.mean(XData[ClusterIndices, 2])
            if include_h:
                c_new[i, 3] = np.mean(XData[ClusterIndices, 3])
            if include_r:
                c_new[i, 4] = np.mean(XData[ClusterIndices, 4])

            # Handle s attribute (arc length calculations) if included
            if include_s:
                t_values = t_data[ClusterIndices]
                x_t_vals, y_t_vals = [], []
                for t in t_values:
                    x_t, y_t = parametric_ellipse(t, sigma_x, sigma_y, sqrt_term, h, k_param)
                    x_t_vals.append(x_t)
                    y_t_vals.append(y_t)

                x_avg = np.mean(x_t_vals)
                y_avg = np.mean(y_t_vals)
                t_initial_guess = initial_guess_for_t(x_avg, y_avg, sigma_x, sigma_y, h, k_param, sqrt_term)

                result = minimize_scalar(
                    lambda t: np.linalg.norm([parametric_ellipse(t, sigma_x, sigma_y, sqrt_term, h, k_param)[0] - x_avg,
                                              parametric_ellipse(t, sigma_x, sigma_y, sqrt_term, h, k_param)[1] - y_avg]),
                    bounds=(t_initial_guess - np.pi / 4, t_initial_guess + np.pi / 4),
                    method='bounded'
                )
            t_avg = result.x
            avg_s = arc_length(t_avg, sigma_x, sigma_y, sqrt_term) / L

            # Store the updated s value in the centroid
            if XData.shape[1] > 2:
                c_new[i, 2] = avg_s  # Store updated s value

            # Update coherence measure for x, y, s, and optionally h, r
            if final_iteration:
                x_vals = XData[ClusterIndices, 0]
                y_vals = XData[ClusterIndices, 1]
                s_vals = XData[ClusterIndices, 2]
                s_circular_distances = np.minimum(np.abs(s_vals - avg_s), 1 - np.abs(s_vals - avg_s))
                
                
                rho_j = np.mean( ((x_vals - c_new[i, 0]) ** 2 + 
                                (y_vals - c_new[i, 1]) ** 2) + 
                                  (s_circular_distances ** 2))

                rho_j_total += rho_j

    if final_iteration:
        # Sum the coherence measure across all clusters
        mean_rho_bar = rho_j_total / k
        return mean_rho_bar

    return c_new
