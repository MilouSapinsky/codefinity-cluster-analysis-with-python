import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# Generating synthetic dataset with an anisotropic shape
data, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=42)
# Defining the transformation matrix
transformation_matrix = np.array([[0.6, -0.6], [0.4, 0.8]])
# Applying transformation
data = np.dot(data, transformation_matrix)
# Standardizing the dataset
data = StandardScaler().fit_transform(data)

# Initialize the model and predict cluster labels
dbscan = DBSCAN(eps=0.3, min_samples=6)
labels = dbscan.fit_predict(data)

unique_labels = set(labels)

# Plotting the clustered data
plt.figure(figsize=(8, 5))

clusters = []
for label in unique_labels:
    # Extract the points belonging to each cluster
    cluster_points = data[labels == label]
    clusters.append(cluster_points)
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {label}")

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("DBSCAN Clustering on Anisotropic Data")
plt.legend()
plt.show()