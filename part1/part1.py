# Imports
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor

for fileName in ['no_outliers.txt','with_outliers.txt']: # Example files used for input; feel free to change
    # Reading Input
    with open(fileName, "r") as file:
        file.readline() # Skips first line because it isn't needed
        coords = []
        for line in file:
            lat, lon = map(float, line.split()) # Converts coordinates from strings to floats
            coords.append([lat, lon])
        coords = np.array(coords) # Converts coords to NumPy array for fast clustering

    # Filtering Outliers
    lof = LocalOutlierFactor(n_neighbors=5, contamination='auto')
    prediction = lof.fit_predict(coords)
    coords = coords[prediction == 1] # Filters out coordinates that are predicted to be outliers

    # Creating Clustering Model
    kmeans = KMeans(n_clusters=5, random_state=0, n_init='auto').fit(coords)

    # Finding/Sorting Centroids
    centroids = kmeans.cluster_centers_ 
    latitudes = centroids[:,0] # Takes first column of all rows
    centroids = centroids[latitudes.argsort()]  # Sorts by ascending latitude

    # Output
    for c in centroids:
        print(round(c[0], 5), round(c[1], 5)) # Rounds to 5 decimal places
    print() # New line to create space for future outputs