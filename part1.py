# Imports
import numpy as np
from sklearn.cluster import KMeans

# Reading Input
with open("inferences.txt", "r") as file: # inferences.txt is the 'with outliers.txt' file and used as an example
    file.readline() # Skips first line because it isn't needed
    coords = []
    for line in file:
        lat, lon = map(float, line.split()) # Converts coordinates from strings to floats
        coords.append([lat, lon])
    coords = np.array(coords) # Converts coords to NumPy array for fast clustering

# Creating Clustering Model
kmeans = KMeans(n_clusters=5, random_state=0, n_init='auto').fit(coords)

# Finding/Sorting Centroids
centroids = kmeans.cluster_centers_ 
latitudes = centroids[:,0] # Takes first column of all rows
centroids = centroids[latitudes.argsort()]  # Sorts by ascending latitude

# Output
for c in centroids:
    print(round(c[0], 5), round(c[1], 5)) # Rounds to 5 decimal places