# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 20:15:50 2024

@author: mpp24
"""

# Generate geographic and attribute data

import numpy as np
import matplotlib.pyplot as plt

def generate_geo_data(k=3, n=7, cluster_params=None, housing_params=None, favorability_params=None, seed=23):
    """
    Generates k clusters of geographic data points, each sampled from a specified uniform distribution
    with added Gaussian noise, and additional attributes like housing price and favorability rating.

    Parameters:
        k (int): Number of clusters.
        n (int): Number of points in each cluster.
        cluster_params (list of dicts): List of dictionaries with parameters for each cluster's distribution.
                                        Each dictionary should contain:
                                        - 'x_range' (tuple): Range for x values (min, max).
                                        - 'y_range' (tuple): Range for y values (min, max).
                                        - 'x_std' (float): Standard deviation for x noise.
                                        - 'y_std' (float): Standard deviation for y noise.
        housing_params (list of dicts): List of dictionaries with mean housing prices for each cluster.
                                        Each dictionary should contain:
                                        - 'mean_price' (float): Mean housing price for the cluster.
                                        - 'std_price' (float): Standard deviation for housing price.
        favorability_params (list of dicts): List of dictionaries with mean favorability ratings for each cluster.
                                        Each dictionary should contain:
                                        - 'mean_rating' (float): Mean favorability rating for the cluster.
                                        - 'std_rating' (float): Standard deviation for favorability rating.
    Returns:
        np.array: Generated data points with shape (k * n, 4) for x, y, h, and r.
    """
    np.random.seed(23)  # Set random seed for reproducibility
    
    if cluster_params is None:
        # Default cluster parameters if none provided
        cluster_params = [
            {'x_range': (7, 11), 'y_range': (-4, -2), 'x_std': 2, 'y_std': 0.2},  # Bottom right cluster
            {'x_range': (0, 2.5), 'y_range': (-3, -2), 'x_std': 2, 'y_std': 0.2},  # Bottom left cluster
            {'x_range': (6, 10), 'y_range': (3, 6), 'x_std': 2, 'y_std': 0.2}   # Top cluster
            #{'x_range': (16.5, 17.5), 'y_range': (-1, 1), 'x_std': 0.6, 'y_std': 1}  # Right cluster
        ]
    
    if housing_params is None:
        # Default housing price parameters if none provided
        housing_params = [
            {'mean_price': 500000, 'std_price': 50000},  # Cluster 1
            {'mean_price': 200000, 'std_price': 50000},  # Cluster 2
            {'mean_price': 350000, 'std_price': 50000} # Cluster 3
            #{'mean_price': 300000, 'std_price': 50000}   # Cluster 4
        ]

    if favorability_params is None:
        # Default favorability rating parameters if none provided
        favorability_params = [
            {'mean_rating': 0.3, 'std_rating': 0.1},  # Cluster 1
            {'mean_rating': 0.9, 'std_rating': 0.1},  # Cluster 2
            {'mean_rating': 0.6, 'std_rating': 0.1}  # Cluster 3
            #{'mean_rating': 0.6, 'std_rating': 0.1}   # Cluster 4
        ]

    data = []
    actual_means = {'h': [], 'r': []}  # To store the actual means of h and r for each cluster

    for i in range(k):
        params = cluster_params[i]
        housing = housing_params[i]
        favorability = favorability_params[i]
        
        # Generate x and y values with uniform distribution and Gaussian noise
        x_vals = np.random.uniform(params['x_range'][0], params['x_range'][1], n)
        y_vals = np.random.uniform(params['y_range'][0], params['y_range'][1], n)
        x_vals += np.random.normal(0, params['x_std'], n)
        y_vals += np.random.normal(0, params['y_std'], n)
        
        # Generate h values (housing prices) with Gaussian distribution
        h_vals = np.random.normal(housing['mean_price'], housing['std_price'], n)
        
        # Generate r values (favorability ratings) with Gaussian distribution
        r_vals = np.random.normal(favorability['mean_rating'], favorability['std_rating'], n)
        
        # Combine x, y, h, and r values and add to the data list
        cluster_points = np.vstack((x_vals, y_vals, h_vals, r_vals)).T
        data.append(cluster_points)

        # Calculate and store the actual means of h and r for this cluster
        actual_means['h'].append(np.mean(h_vals))
        actual_means['r'].append(np.mean(r_vals))

    # Combine all clusters into a single array
    data = np.vstack(data)

    # Print the actual means of h and r for each cluster
    for i in range(k):
        print(f"Cluster {i + 1}: Actual Mean h = {actual_means['h'][i]:.2f}, Actual Mean r = {actual_means['r'][i]:.2f}")

    return data

def save_geo_data(data, filename_xy="xy_data.txt", filename_xyh="xyh_data.txt", filename_xyhr="xyhr_data.txt"):
    """
    Saves the generated geographic data points to text files.

    Parameters:
        data (np.array): Array of data points with shape (k * n, 4) for x, y, h, and r.
        filename_xy (str): Name of the file to save x and y data.
        filename_xyh (str): Name of the file to save x, y, and h data.
        filename_xyhr (str): Name of the file to save x, y, h, and r data.
    """
    # Save x, y coordinates to xy_data.txt
    np.savetxt(filename_xy, data[:, :2], delimiter=',', fmt='%.4f', header='x,y', comments='')
    print(f"x, y coordinates saved to {filename_xy}")

    # Save x, y, h coordinates to xyh_data.txt
    np.savetxt(filename_xyh, data[:, :3], delimiter=',', fmt='%.4f', header='x,y,h', comments='')
    print(f"x, y, h coordinates saved to {filename_xyh}")

    # Save x, y, h, r coordinates to xyhr_data.txt
    np.savetxt(filename_xyhr, data, delimiter=',', fmt='%.4f', header='x,y,h,r', comments='')
    print(f"x, y, h, r coordinates saved to {filename_xyhr}")

# Example usage
if __name__ == "__main__":
    k = 3  # Number of clusters
    n = 7  # Number of points per cluster
    
    # Generate geographic data with the default parameters
    data = generate_geo_data(k=k, n=n)
    
    # Save the generated data to three files
    save_geo_data(data, filename_xy="xy_data.txt", filename_xyh="xyh_data.txt", filename_xyhr="xyhr_data.txt")
