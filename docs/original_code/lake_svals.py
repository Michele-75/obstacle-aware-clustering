# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 02:20:10 2025

@author: mpp24
"""
#lake-svals.py
#Parametrize lake, get s values, normalize all factors, save as xysrp_data.txt

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from scipy.interpolate import splprep, splev
from scipy.interpolate import UnivariateSpline
from scipy.optimize import fsolve
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Step 1: Parametrize the lake from latlong_manual.txt
def load_lake_coordinates(file_path):
    """Load lake coordinates from the given file."""
    coords = np.loadtxt(file_path)
    x, y = coords[:, 1], coords[:, 0]  # Longitude (x) and Latitude (y)
    return x, y



def parameterize_lake(x, y):
    """
    Parametrize the lake boundary using cubic splines.
    The x and y coordinates are treated as functions of a common 't' parameter.
    """
    # Create a parameter t based on the index of the data points
    t = np.linspace(0, 1, len(x))
    
    # Fit cubic splines to x(t) and y(t)
    spline_x = CubicSpline(t, x)
    spline_y = CubicSpline(t, y)
    
    # Generate points for the fitted splines
    t_vals = np.linspace(0, 1, 1000)
    
    return spline_x, spline_y, t_vals

def lake_length(spline_x, spline_y):
    """Calculate the total arc length of the lake."""
    ds_dt = lambda t: np.sqrt(spline_x(t, 1)**2 + spline_y(t, 1)**2)
    
    # # **Insert the plotting code here:**
    # t_vals = np.linspace(0, 1, 1000)
    # ds_dt_vals = ds_dt(t_vals)
  
    # import matplotlib.pyplot as plt
    # plt.figure
    # plt.plot(t_vals, ds_dt_vals)
    # plt.xlabel("t")
    # plt.ylabel("ds/dt")
    # plt.title("Plot of ds/dt over t")
    # plt.grid(True)
    # print("Displaying ds/dt plot (Figure 1)")
    # plt.show()
  
    length, _ = quad(ds_dt, 0, 1, limit=500)
    print(f"length={length}")
    return length

# Step 2: Process the hotel data
def load_hotel_data(file_path):
    """Load hotel data from the given file."""
    # Load the data assuming space-separated values
    data = np.loadtxt(file_path, delimiter=' ', skiprows=1)  # Skip header row
    
    # Extract the columns
    latitudes = data[:, 0]
    longitudes = data[:, 1]
    ratings = data[:, 2]
    prices = data[:, 3]
    
    return latitudes, longitudes, ratings, prices

def initial_guess(x_p, y_p, t, spline_x, spline_y):
    """Provide an initial guess for the parameter t."""
    distances = np.sqrt((spline_x(t) - x_p)**2 + (spline_y(t) - y_p)**2)
    return t[np.argmin(distances)]

def closest_point_on_curve(t, x_p, y_p, spline_x, spline_y):
    """Equation to find the closest point on the curve."""
    dx_dt = spline_x(t, 1)
    dy_dt = spline_y(t, 1)
    x_t = spline_x(t)
    y_t = spline_y(t)
    return (x_p - x_t) * dx_dt + (y_p - y_t) * dy_dt

def solve_for_t(x_p, y_p, spline_x, spline_y, t_guess):
    """Solve for t numerically."""
    t_closest = fsolve(closest_point_on_curve, t_guess, args=(x_p, y_p, spline_x, spline_y))
    return t_closest[0]

def calculate_arc_length(t, spline_x, spline_y):
    """Calculate arc length from t=0 to t."""
    ds_dt = lambda t_val: np.sqrt(spline_x(t_val, 1)**2 + spline_y(t_val, 1)**2)
    arc_length, _ = quad(ds_dt, 0, t, limit=500)
    return arc_length

# Step 3: Normalize and save the data
def normalize_and_save(x, y, s, r, p, output_file):
    """Normalize data and save to a file."""
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x.reshape(-1, 1))
    y_scaled = scaler.fit_transform(y.reshape(-1, 1))
    s_scaled = s.reshape(-1, 1)  # No need to scale further since s is already normalized
    r_scaled = scaler.fit_transform(r.reshape(-1, 1))
    p_scaled = scaler.fit_transform(p.reshape(-1, 1))
    combined_data = np.hstack((x_scaled, y_scaled, s_scaled, r_scaled, p_scaled))
    np.savetxt(output_file, combined_data, delimiter=',', header='x_scaled,y_scaled,s_scaled,r_scaled,p_scaled', fmt='%.4f')

# def visualize_data(spline_x, spline_y, x, y, s):
#     """Visualize the lake and hotel points."""
#     t_vals = np.linspace(0, 1, 500)
    
#     lake_x = spline_x(t_vals)
#     lake_y = spline_y(t_vals)
    
#     plt.figure(figsize=(12, 8))
#     plt.plot(lake_x, lake_y, label='Lake Boundary', color='blue')
#     plt.scatter(x, y, color='red', label='Hotels')
    
#     # Increased font size for the labels
#     for i, (x_pt, y_pt, s_val) in enumerate(zip(x, y, s)):
#         plt.text(x_pt, y_pt, f'{s_val:.2f}', fontsize=12, color='green')  # Adjust fontsize here
    
#     plt.xlabel('Longitude')
#     plt.ylabel('Latitude')
#     plt.title('Hotels Around Lake Tahoe with Arc Length S')
#     plt.legend()
#     plt.grid()
#     plt.show()

def visualize_data(spline_x, spline_y, x, y, s):
    """Visualize the lake and hotel points with enhanced formatting."""
    t_vals = np.linspace(0, 1, 500)

    lake_x = spline_x(t_vals)
    lake_y = spline_y(t_vals)

    plt.figure(figsize=(12, 8))
    plt.plot(lake_x, lake_y, label='Lake Boundary', color='blue', linewidth=2)
    plt.scatter(x, y, color='red', label='Hotels')

    # Label each hotel point with normalized s value
    for i, (x_pt, y_pt, s_val) in enumerate(zip(x, y, s)):
        plt.text(x_pt, y_pt, f's={s_val:.2f}', fontsize=14, color='green', ha='right')

    # Enhanced axis and title formatting
    plt.xlabel('Longitude', fontsize=16)
    plt.ylabel('Latitude', fontsize=16)
    plt.title('Hotels Around Lake Tahoe with Arc Length S', fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Enhanced grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.7)
    plt.minorticks_on()
    plt.tick_params(axis='both', which='major', length=8, width=1)
    plt.tick_params(axis='both', which='minor', length=4, width=0.5)

    # Enhanced legend
    plt.legend(fontsize=14)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

# Main script
if __name__ == "__main__":
    # Load and parametrize the lake
    lake_x, lake_y = load_lake_coordinates("latlong_manual.txt")
    spline_x, spline_y, t = parameterize_lake(lake_x, lake_y)
    L = lake_length(spline_x, spline_y)

    # Load hotel data
    hotel_lat, hotel_lon, hotel_ratings, hotel_prices = load_hotel_data("hotels_formatted.txt")

    # Process each hotel
    s_values = []
    for x_p, y_p in zip(hotel_lon, hotel_lat):
        t_guess = initial_guess(x_p, y_p,t, spline_x, spline_y)
        t_closest = solve_for_t(x_p, y_p, spline_x, spline_y, t_guess)
        s_closest = calculate_arc_length(t_closest, spline_x, spline_y) / L  # Normalize by lake length
        s_values.append(s_closest)

    s_values = np.array(s_values)

    # Normalize and save data
    normalize_and_save(hotel_lon, hotel_lat, s_values, hotel_ratings, hotel_prices, "xysrp_data.txt")

    # Visualize
    visualize_data(spline_x, spline_y, hotel_lon, hotel_lat, s_values)
