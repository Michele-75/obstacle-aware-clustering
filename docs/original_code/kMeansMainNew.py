# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:57:23 2024

@author: mpp24
"""

# kMeansMainNew
# kmeansmain.py script for clustering. Takes in file xysth_data.txt and performs custom clustering algorithm
# requires reassignDataVectors.py and updateWeights.py

import numpy as np
import matplotlib.pyplot as plt
from reassignDataVectorsNew import reassignDataVectorsNew
from updateWeightsNew import updateWeightsNew

def kMeansMainNew(attributes_to_include=[0, 1, 2,4,5], alpha=1.0, beta=1.0, gamma=1.0, plot_clusters=False):
    np.random.seed(35)  # Set random seed for reproducibility

    # Load data from 'xysthr_data.txt'
    filename = 'xysthr_data.txt'
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    
    # Select specified attributes
    XData = data[:, attributes_to_include]
    t_data = data[:, 3]  # t values for arc length calculations

    k = 3  # Number of clusters
    L = 38.2575  # Total arc length of ellipse obstacle

    # Dynamically set weights for attributes
    weights = np.ones(len(attributes_to_include))
    weights[:2] *= alpha  # Apply alpha to x, y
    if len(attributes_to_include) > 2:
        weights[2] *= beta  # Apply beta to s
    if len(attributes_to_include) > 3:
        weights[3:] *= gamma  # Apply gamma to h and r

    print("Weights:", weights)

    # Load original x, y coordinates from 'xyhr_data.txt' (or another source of original data)
    original_data_filename = 'xyhr_data.txt'
    original_data = np.loadtxt(original_data_filename, delimiter=',', skiprows=1)  # Original data, including x, y
    original_xy = original_data[:, :2]  # Extract only x and y
    print("Original XY shape:", original_xy.shape)
      

    # Constants for the parametric curve
    c_param = 0.01
    A = 1.0
    sigma_x = 3.0
    sigma_y = 0.6
    h = 6.0
    k_param = 0.0

    # k-means++ initialization for centroids
    numDataPoints = XData.shape[0]
    c = np.zeros((k, XData.shape[1]))
    c[0, :] = XData[np.random.randint(numDataPoints), :] #Randomly choose first centroid

    for i in range(1, k):
        D = np.zeros(numDataPoints)
        for j in range(numDataPoints):
            point = XData[j, :]
            distances = np.zeros(i)
            for m in range(i):
                distances[m] = np.linalg.norm((point - c[m, :]) * weights, 2)
            D[j] = np.min(distances)

        probs = D**2 / np.sum(D**2)
        cumprobs = np.cumsum(probs)
        r = np.random.rand()
        next_centroid = np.argmax(cumprobs >= r)
        c[i, :] = XData[next_centroid, :]

    # Iterative k-means clustering
    maxIterations = 100
    for iter in range(maxIterations):
        IndexSet = reassignDataVectorsNew(k, XData, c, L, alpha, beta, gamma)
        print("IndexSet shape:", IndexSet.shape)
        c_new = updateWeightsNew(k, XData, IndexSet, alpha, beta, gamma, t_data, c_param, A, sigma_x, sigma_y, h, k_param, L)
        
        if np.linalg.norm(c_new - c) < 1e-4:
            print(f"Converged after {iter + 1} iterations.")
            break
        c = c_new

    # Final update to calculate rho_bar for each cluster
    print("\nCalculating final rho_bar values:")
    rho_bar_total = updateWeightsNew(k, XData, IndexSet, alpha, beta, gamma, t_data, c_param, A, sigma_x, sigma_y, h, k_param, L, final_iteration=True)

    # Plot clusters if specified
    if plot_clusters:
        plt.figure()
        colors = plt.cm.get_cmap('viridis', k)  # Generate k distinct colors for the clusters
        transformed_centroids = []  # To store centroids in original coordinates
        
        for i in range(k):
            clusterPoints = original_xy[np.where(IndexSet == i)[0], :]  # Use original x, y coordinates
            plt.scatter(clusterPoints[:, 0], clusterPoints[:, 1], label=f'Cluster {i+1}')
        
            # Compute centroid in terms of original x, y coordinates
            centroid_x = np.mean(clusterPoints[:, 0])
            centroid_y = np.mean(clusterPoints[:, 1])
            transformed_centroids.append((centroid_x, centroid_y))
            plt.scatter(centroid_x, centroid_y, color='red', marker='x', s=200)  # Plot transformed centroid

        # Load and plot the ellipse
        ellipse_data = np.loadtxt('ellipse_xy.txt', delimiter=',', skiprows=1)
        plt.plot(ellipse_data[:, 0], ellipse_data[:, 1], color='black', linestyle='--')
        
        plt.xlabel('X (Original)')
        plt.ylabel('Y (Original)')
        plt.title(f'Clustering with alpha={alpha}, beta={beta}, gamma={gamma}', fontsize=18)
        plt.legend(loc='best')
        plt.grid(True)
        plt.axis('equal')
        plt.show()

    return rho_bar_total

if __name__ == "__main__":
   
    # # Option 1: Test a range of (alpha, beta) values and plot the results as a contour plot
    # alpha_values = [0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6]  # Define a range of alpha values
    # beta_values = [0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6]  # Define a range of beta values
    # results = []

    # # Try each alpha-beta combination and store the results
    # for alpha in alpha_values:
    #     for beta in beta_values:
    #         print(f"\nRunning clustering for alpha = {alpha}, beta = {beta}")
    #         J_alpha_beta = kMeansMainNew(attributes_to_include=[0, 1, 2, 4], alpha=alpha, beta=beta)
    #         results.append((alpha, beta, J_alpha_beta))
    #         print(f"Total rho_bar (J(alpha, beta)) for alpha={alpha}, beta={beta}: {J_alpha_beta}")

    # # Plot contour of J(alpha, beta)
    # alphas, betas, J_values = zip(*results)
    # plt.figure(figsize=(10, 6))
    # plt.tricontourf(alphas, betas, J_values, levels=15, cmap="viridis")
    # plt.colorbar(label=r'Distance from centroids $\bar{\rho}$')
    # plt.xlabel(r'$\alpha$ value')
    # plt.ylabel(r'$\beta$ value')
    # plt.title(r'Finding optimal $\alpha$ and $\beta$')
    # plt.grid(True)
    # plt.show()
    
    
    
    
    # # Option 2: Define a fine mesh of alpha and beta values to test
    # alpha_values = np.linspace(1, 2.0, 30)  # 50 evenly spaced values between 0 and 2.0 for alpha
    # beta_values = np.linspace(1, 2.0, 30)   # 50 evenly spaced values between 0 and 2.0 for beta
    # results = np.zeros((len(alpha_values), len(beta_values)))  # Initialize a 2D array to store results

    # # Test each combination of alpha and beta
    # for i, alpha in enumerate(alpha_values):
    #     for j, beta in enumerate(beta_values):
    #         print(f"\nRunning clustering for alpha = {alpha:.2f}, beta = {beta:.2f}")
    #         rho_bar = kMeansMainNew(attributes_to_include=[0, 1, 2,4,5], alpha=alpha, beta=beta)
    #         results[i, j] = rho_bar  # Store the coherence measure
    #         print(f"Total rho_bar for alpha={alpha:.2f}, beta={beta:.2f}: {rho_bar}")

    # # Plot the results as a heatmap
    # plt.figure(figsize=(10, 8))
    # plt.imshow(results, origin='lower', aspect='auto', extent=[1, 2.0, 1, 2.0], cmap='viridis')
    # plt.colorbar(label=r'Coherence Measure $\bar{\rho}_s$', orientation='vertical')
    # plt.xlabel(r'$\beta$ value', fontsize=16)
    # plt.ylabel(r'$\alpha$ value', fontsize=16)
    # plt.title(r'Coherence Measure $\bar{\rho}_s$ for $\alpha$ and $\beta$', fontsize=18)
    # plt.xticks(np.linspace(1, 2.0, 10))  # Adjust tick spacing for clarity
    # plt.yticks(np.linspace(1, 2.0, 10))
    # plt.grid(False)
    # plt.show()
    
    
    

    # Option 3: Run clustering with a specific (alpha, beta, gamma) and plot the clusters
    specific_alpha = 0
    specific_beta = 0.21052631578947367
    specific_gamma = 0.10526315789473684
    print(f"\nRunning clustering for specific alpha = {specific_alpha}, beta = {specific_beta}, gamma = {specific_gamma}")
    J_specific = kMeansMainNew(attributes_to_include=[0, 1, 2, 4, 5], alpha=specific_alpha, beta=specific_beta, gamma=specific_gamma, plot_clusters=True)
    print(f"Total rho_bar (J(alpha, beta)) for alpha={specific_alpha}, beta={specific_beta}, gamma={specific_gamma}: {J_specific}")
