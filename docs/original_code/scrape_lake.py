# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 16:14:52 2024

@author: mpp24
"""

#scrape_lake.py

#Before using install required Python library: pip install -U googlemaps
import googlemaps
import numpy as np

# Initialize Google Maps API client
API_KEY = "AIzaSyAQ5rqasFIBXikvmoWocnXvDXNyDTStHdM"  # Current API key associted with mpperry2@ncsu.edu
gmaps = googlemaps.Client(key=API_KEY)

def read_data(file_name):
    """
    Reads latitude and longitude points from a text file.
    Format: Latitude Longitude (no commas, whitespace-separated)
    """
    with open(file_name, "r") as file:
        points = [tuple(map(float, line.split())) for line in file]
    return points

def collect_hotels(lat_long_list, radius=5000, output_file="hotels_data.txt"):
    """
    Collect hotel data near the manually selected points.
    Parameters:
        lat_long_list (list): List of (latitude, longitude) tuples.
        radius (int): Search radius in meters.
        output_file (str): File to save hotel data.
    """
    hotels = set()
    
    # Keywords to filter out undesired results
    undesired_keywords = ["campground", "camp", "park", "housing", "hostel", "estates",
                          "recreation", "management", "airbnb", "townhome", "area", 
                          "mobile","rv", "trail", "rental", "US"]
    
    
    # Minimum number of user ratings to include a property
    min_user_ratings = 50
    
    
    # Function to check if a name contains undesired keywords
    def is_desired_property(name):
        # Check for any undesired keyword in the name (case-insensitive)
        name_lower = name.lower()
        return not any(keyword.lower() in name_lower for keyword in undesired_keywords)
    for lat, lng in lat_long_list:
        # Query hotels near each point
        try:
            results = gmaps.places_nearby(location=(lat, lng), radius=radius, type="lodging")
            for result in results["results"]:
                name = result.get("name", "N/A")
                user_ratings = result.get("user_ratings_total", 0)  # Default to 0 if not provided

                if is_desired_property(name) and user_ratings >= min_user_ratings:  # Apply both filters
                    location = result["geometry"]["location"]
                    hotels.add((
                        name,
                        location["lat"],
                        location["lng"],
                        result.get("rating", "N/A"),
                        user_ratings
                    ))
        except Exception as e:
            print(f"Error querying location ({lat}, {lng}): {e}")

    # Save data to file
    with open(output_file, "w", encoding="utf-8") as file:
        file.write("Name\tLatitude\tLongitude\tRating\tUser_Ratings\n")
        for hotel in hotels:
            file.write("\t".join(map(str, hotel)) + "\n")

if __name__ == "__main__":
    # File containing manually chosen points
    input_file = "latlong_manual.txt"
    
    # Read points
    print("Reading manually chosen points...")
    points = read_data(input_file)
    
    # Collect hotel data
    hotels_file = "hotels_lake_tahoe1.txt"
    print(f"Collecting hotel data within radius of manually selected points and saving to {hotels_file}...")
    collect_hotels(points, radius=5000, output_file=hotels_file)

    print("Process complete!")
    
    
    
    
    
    
    
    
    
    
    
    
    
