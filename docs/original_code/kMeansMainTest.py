# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 12:54:30 2024

@author: mpp24
"""

from scipy.optimize import dual_annealing
import numpy as np
import matplotlib.pyplot as plt
from reassignDataVectorsNew2 import reassignDataVectorsNew2
from updateWeightsNew2 import updateWeightsNew2
from matplotlib.colors import ListedColormap



def kMeansMainTest(attributes_to_include=[0, 1, 2, 4, 5], alpha=1.0, beta=1.0, plot_clusters=False):
    np.random.seed(36)  # Set random seed for reproducibility

    # Load data and preprocess
    filename = 'xysthr_data.txt'
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    XData = data[:, attributes_to_include]
    t_data = data[:, 3]  # Arc length parameter

    k = 3  # Number of clusters
    L = 38.2575  # Total arc length of the ellipse

    # Initialize weights
    weights = np.ones(len(attributes_to_include))
    weights[:2] *= alpha  # Apply alpha to x, y
    if len(attributes_to_include) > 2:
        weights[2] *= beta  # Apply beta to s

    # Load original x, y coordinates
    original_data_filename = 'xyhr_data.txt'
    original_data = np.loadtxt(original_data_filename, delimiter=',', skiprows=1)
    original_xy = original_data[:, :2]

    # Constants for the parametric curve
    c_param = 0.01
    A = 1.0
    sigma_x = 3.0
    sigma_y = 0.6
    h = 6.0
    k_param = 0.0



    # Centroid initialization
    def initialize_clusters(randomize=True):
        np.random.seed(36)
        numDataPoints = XData.shape[0]
        c = np.zeros((k, XData.shape[1]))
        c[0, :] = XData[np.random.randint(numDataPoints), :]
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
        return c

    # Fixed centroids for contour plot
    c_fixed = initialize_clusters(randomize=False)

    # Debugging: Compare centroids and objective values for specific alpha and beta
    specific_alpha = 0.10
    specific_beta = 1

    IndexSet_specific = reassignDataVectorsNew2(k, XData, c_fixed, L, specific_alpha, specific_beta)
    rho_bar_specific, sigma_a_specific, sig_diff_h, sig_diff_r = updateWeightsNew2(
        k, XData, IndexSet_specific, specific_alpha, specific_beta, t_data, c_param, A, sigma_x, sigma_y, h, k_param, L, final_iteration=True
    )
    specific_objective_value = rho_bar_specific + (1 - sigma_a_specific)

    # Debugging output
    print("\nDebugging Specific Alpha/Beta Calculation")
    print(f"Specified alpha: {specific_alpha}, Specified beta: {specific_beta}")
    print(f"rho_bar: {rho_bar_specific}, 1 - sigma_a: {1 - sigma_a_specific}")
    print(f"Objective value: {specific_objective_value}")
    print(f"Fixed Centroids (c_fixed): {c_fixed}")
    print(f"IndexSet Specific: {IndexSet_specific}")

    # Define objective function for optimization
    def objective(params):
        a, b = params
        IndexSet_opt = reassignDataVectorsNew2(k, XData, c_fixed, L, a, b)
        rho_bar, sigma_a, _, _ = updateWeightsNew2(k, XData, IndexSet_opt, a, b, t_data, c_param, A, sigma_x, sigma_y, h, k_param, L, final_iteration=True)
        return rho_bar + (1 - sigma_a)

    # Validate contour plot at specific alpha and beta
    def debug_objective(params):
        a, b = params
        IndexSet_opt = reassignDataVectorsNew2(k, XData, c_fixed, L, a, b)
        rho_bar, sigma_a, _, _ = updateWeightsNew2(k, XData, IndexSet_opt, a, b, t_data, c_param, A, sigma_x, sigma_y, h, k_param, L, final_iteration=True)
        objective_value = rho_bar + (1 - sigma_a)
        print(f"Debug Contour: Alpha={a}, Beta={b}, rho_bar={rho_bar}, 1-sigma_a={1-sigma_a}, Objective={objective_value}")
        return objective_value

    specific_contour_value = debug_objective([specific_alpha, specific_beta])
    print(f"Contour Plot Value for alpha={specific_alpha}, beta={specific_beta}: {specific_contour_value}")

    # Generate objective surface for contour plot
    alpha_values = np.linspace(0, 2, 50)
    beta_values = np.linspace(0, 2, 50)
    obj_values = np.zeros((len(alpha_values), len(beta_values)))

    for i, a in enumerate(alpha_values):
        for j, b in enumerate(beta_values):
            obj_values[i, j] = objective([a, b])

    # Plot the contour
    plt.figure()
    plt.contourf(alpha_values, beta_values, obj_values.T, levels=50, cmap='viridis')
    plt.colorbar(label='Objective Value')
    plt.xlabel('Alpha')
    plt.ylabel('Beta')
    plt.title('Objective Surface')

    # Mark the specific alpha and beta on the plot
    plt.scatter(specific_alpha, specific_beta, color='red', label=f'Point (Alpha={specific_alpha}, Beta={specific_beta})')
    plt.text(specific_alpha, specific_beta, f"{specific_objective_value:.4f}", color='red')
    plt.legend()
    plt.show()

    # Simulated Annealing Optimization
    bounds = [(0, 2), (0, 2)]
    result = dual_annealing(objective, bounds, seed=44)
    opt_alpha, opt_beta = result.x

    # Final output for optimal parameters
    IndexSet_opt = reassignDataVectorsNew2(k, XData, c_fixed, L, opt_alpha, opt_beta)
    rho_bar, sigma_a, sig_diff_h, sig_diff_r = updateWeightsNew2(
        k, XData, IndexSet_opt, opt_alpha, opt_beta, t_data, c_param, A, sigma_x, sigma_y, h, k_param, L, final_iteration=True
    )
    print(f"\nOptimal weights found via Simulated Annealing: alpha={opt_alpha}, beta={opt_beta}")
    print(f"Best objective value: {result.fun}")
    print(f"rho_bar: {rho_bar}, 1 - sigma_a: {1 - sigma_a}")
    print(f"Significant differences in h: {sig_diff_h}, r: {sig_diff_r}")

    # Return results
    return opt_alpha, opt_beta, rho_bar_specific, sigma_a_specific


if __name__ == "__main__":
    # Run clustering and plot results
    alpha, beta, rho_bar, sigma_a = kMeansMainTest(attributes_to_include=[0, 1, 2, 4, 5], plot_clusters=True) 