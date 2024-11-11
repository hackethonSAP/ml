# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs

# Step 1: Create a sample dataset
data, _ = make_blobs(n_samples=10, centers=3, random_state=42)

# Step 2: Perform hierarchical clustering
Z = linkage(data, method='ward')  # 'ward' minimizes variance within clusters

# Step 3: Plot the dendrogram
plt.figure(figsize=(8, 4))
dendrogram(Z)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.show()


# %%
# Hierarchical clustering is a method of clustering that builds a hierarchy of clusters. Unlike methods like K-Means, where you specify a fixed number of clusters, hierarchical clustering starts with each data point as a separate cluster and merges them step-by-step until only one cluster remains. This method is commonly visualized with a dendrogram, showing the merging process.

# Here’s how to implement hierarchical clustering in Python using the scipy and sklearn libraries.
# Steps for Implementation

#     Import Necessary Libraries.
#     Load or Generate a Dataset.
#     Standardize the Data (recommended for distance-based algorithms like hierarchical clustering).
#     Perform Hierarchical Clustering using scipy to create a dendrogram and sklearn for the clustering itself.
#     Plot the Dendrogram for visualization.
#     Assign Clusters to data points.


