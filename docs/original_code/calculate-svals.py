# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 01:38:55 2024
Code uploads .txt file with format x coord,y coord in rows. Then numerically 
calculates the arc length parameter t for each point and outputs new data vectors
with x coord, y coord, t value.



@author: mpp24
"""

# This script inputs file xyhr_data.txt, computes the closest point on the parametrized curve to data points, calculates the
# angle from center to that point (t) and then finds associated arc length (s). Then, scales x, y, and s using MinMax 
# Scaler and saves vectors (x, y, s, t, h, r) to a file titled xysthr_data.txt.

import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Function to estimate initial t guess based on the position of (x_p, y_p)
def initial_guess_for_t(x_p, y_p, sigma_x, sigma_y, h, k, sqrt_term):
    delta_x = (x_p - h) / (sigma_x * sqrt_term)
    delta_y = (y_p - k) / (sigma_y * sqrt_term)
    t_guess = np.arctan2(delta_y, delta_x)
    return t_guess % (2 * np.pi)

# Function to find the closest point on the obstacle
def closest_point(t, x_p, y_p, c, A, sigma_x, sigma_y, h, k):
    sqrt_term = np.sqrt(-2 * np.log(c / A))
    x_t = sigma_x * sqrt_term * np.cos(t) + h
    y_t = sigma_y * sqrt_term * np.sin(t) + k
    dx_dt = -sigma_x * sqrt_term * np.sin(t)
    dy_dt = sigma_y * sqrt_term * np.cos(t)
    equation = (x_p - x_t) * dx_dt + (y_p - y_t) * dy_dt
    return equation

# Function to solve for t numerically
def find_closest_point(x_p, y_p, c, A, sigma_x, sigma_y, h, k, initial_guess):
    t_closest = fsolve(closest_point, initial_guess, args=(x_p, y_p, c, A, sigma_x, sigma_y, h, k))
    return t_closest[0]

# Arc length differential ds/dt
def ds_dt(t, sigma_x, sigma_y, sqrt_term):
    return np.sqrt((-sigma_x * sqrt_term * np.sin(t))**2 + (sigma_y * sqrt_term * np.cos(t))**2)

# Compute the arc length from t=0 to t=t for each point
def arc_length(t, sigma_x, sigma_y, sqrt_term):
    s, _ = quad(ds_dt, 0, t, args=(sigma_x, sigma_y, sqrt_term), limit=1000)
    return s

# Function to process the data points, compute t and s values
def process_data_points(file_path, c, A, sigma_x, sigma_y, h, k, L=38.2575):
    # Load the data points from the file
    data_points = np.loadtxt(file_path, delimiter=',', skiprows=1)

    # Initialize lists to store the t, s values and data vectors
    t_vec = []
    s_vec = []
    data_vectors = []

    # Compute sqrt_term once
    sqrt_term = np.sqrt(-2 * np.log(c / A))

    for x_p, y_p in data_points[:, :2]:
        # Compute the initial guess for t
        initial_guess = initial_guess_for_t(x_p, y_p, sigma_x, sigma_y, h, k, sqrt_term)
        # Refine t using fsolve
        t_closest = find_closest_point(x_p, y_p, c, A, sigma_x, sigma_y, h, k, initial_guess)
        t_closest = t_closest % (2 * np.pi)
        # Compute arc length s(t)
        s_closest = arc_length(t_closest, sigma_x, sigma_y, sqrt_term)
        # Store t, s, and original coordinates
        t_vec.append(t_closest)
        s_vec.append(s_closest / L)  # Scale s by dividing by L
        data_vectors.append([x_p, y_p, s_closest])

    # Convert to numpy arrays
    t_vec = np.array(t_vec).reshape(-1, 1)
    s_vec = np.array(s_vec).reshape(-1, 1)
    data_vectors = np.array(data_vectors)

    # Apply Min-Max scaling only to x and y (columns 0 and 1)
    scaler = MinMaxScaler()
    x_y_scaled = scaler.fit_transform(data_vectors[:, :2])  # Only scale x and y

    # Combine the scaled x, y with the scaled s (s/L) and the original t values
    data_vectors_scaled_with_t = np.hstack([x_y_scaled, s_vec, t_vec])  # Combine x, y scaled, s/L, and t

    return data_vectors_scaled_with_t, scaler


# Function to normalize h values and attach r values to create the xysthr_data vector
def normalize_and_combine_h_and_r(data_vectors_scaled_with_t, scaler, file_path="xyhr_data.txt", output_file="xysthr_data.txt"):
    # Load xyhr_data.txt and extract h and r values
    xyhr_data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    h_values = xyhr_data[:, 2].reshape(-1, 1)  # Extract h column and reshape
    r_values = xyhr_data[:, 3].reshape(-1, 1)  # Extract r column and reshape

    # Normalize h values
    h_scaler = MinMaxScaler()
    h_scaled = h_scaler.fit_transform(h_values)

    # Combine x_scaled, y_scaled, s_scaled, t, h_scaled, and r into final data vector
    xysthr_data = np.hstack([data_vectors_scaled_with_t, h_scaled, r_values])

    # Save the result to xysthr_data.txt
    np.savetxt(output_file, xysthr_data, delimiter=',', fmt='%.4f', header='x_scaled,y_scaled,s_scaled,t,h_scaled,r', comments='')
    print(f"Data with normalized h values and original r values saved to {output_file}")

    # Print the final combined vector
    print("\nCombined (x_scaled, y_scaled, s_scaled, t, h_scaled, r) data:")
    for row in xysthr_data:
        print(", ".join(f"{val:.4f}" for val in row))
        

# Function to plot data points in XY plane with labels for scaled s
def plot_xy_with_s_labels(data_vectors_scaled_with_t):
    plt.figure(figsize=(10, 8))

    # Extract x, y, and s values for plotting
    x_vals = data_vectors_scaled_with_t[:, 0]
    y_vals = data_vectors_scaled_with_t[:, 1]
    s_vals = data_vectors_scaled_with_t[:, 2]

    # Plot data points and label with scaled s values
    plt.scatter(x_vals, y_vals, color='blue', label='Data Points')
    for i, (x, y, s) in enumerate(zip(x_vals, y_vals, s_vals)):
        plt.text(x, y, f's={s:.2f}', fontsize=9, ha='right', color='red')

    plt.xlabel('X (scaled)')
    plt.ylabel('Y (scaled)')
    plt.title('Data Points in XY Plane with Scaled S Labels')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.show()

# Main function to process the data and save to file
if __name__ == "__main__":
    # User-defined constants for the parametric curve
    c = 0.01
    A = 1.0
    sigma_x = 3.0
    sigma_y = 0.6
    h = 6.0
    k = 0.0

    # Process the data points to get scaled data vectors (including t values) and the scaler
    data_vectors_scaled_with_t, scaler = process_data_points("xyhr_data.txt", c, A, sigma_x, sigma_y, h, k)

    # Normalize h values and attach r values from xyhr_data.txt and combine with the scaled data
    normalize_and_combine_h_and_r(data_vectors_scaled_with_t, scaler, file_path="xyhr_data.txt", output_file="xysthr_data.txt")

    # Plot the data points in XY plane with s labels
    plot_xy_with_s_labels(data_vectors_scaled_with_t)
    
       
    # Scale the ellipse outline using the same scaler
    # scale_ellipse_outline(scaler, file_path="ellipse_xy.txt", output_file="ellipse_scaled_xy.txt")


    # Save the first two columns of ellipse_scaled (x and y) to a file
    #np.savetxt("ellipse_scaled_xy.txt", ellipse_scaled[:, :2], delimiter=',', fmt='%f', header='x_scaled,y_scaled', comments='')
    #print(f"Scaled ellipse points saved to 'ellipse_scaled_xy.txt'")
    
    #total_length = total_arc_length(sigma_x, sigma_y, c, A)
    #print(f"Total arc length of the ellipse: {total_length:.4f}")
