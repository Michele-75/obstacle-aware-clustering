# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 15:01:05 2024

@author: mpp24
"""

# reassignDataVectorsNew

import numpy as np

def reassignDataVectorsNew(k, XData, c, L, alpha, beta, gamma):
    n = XData.shape[0]  # Number of data points
    closestCluster = np.zeros(n, dtype=int)
    
    # Check which attributes are included
    include_s = XData.shape[1] > 2
    include_h = XData.shape[1] > 3
    include_r = XData.shape[1] > 4

    for d in range(n):
        xD = XData[d, :]  # Current data vector
        sqDistMin = float('inf')  # Set initial minimum distance to a large value
        
        for i in range(k):
            # Apply weights and calculate distances
            geoDist = np.linalg.norm((c[i, 0:2] - xD[0:2]) * alpha, 2)  # Always includes x, y
            
            sDist = 0
            if include_s:
                sDist = min(abs(float(c[i, 2] - xD[2])), L - abs(float(c[i, 2] - xD[2]))) * beta
            
            hDist = 0
            if include_h:
                hDist = abs(float(c[i, 3] - xD[3])) * gamma
            
            rDist = 0
            if include_r:
                rDist = abs(float(c[i, 4] - xD[4])) * gamma
            
            sqDist = np.sqrt(geoDist**2 + sDist**2 + hDist**2 + rDist**2)
            
            if sqDist < sqDistMin:
                closestCluster[d] = i
                sqDistMin = sqDist

    return closestCluster