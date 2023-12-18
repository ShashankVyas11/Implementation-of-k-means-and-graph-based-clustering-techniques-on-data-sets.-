# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Generate a synthetic dataset with three clusters
X, y = make_blobs(n_samples=300, centers=3, random_state=42, cluster_std=1.0)

# Standardize the features (important for k-means)
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Visualize the synthetic dataset
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k', s=50)
plt.title('Original Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_std)

# Visualize the k-means clustering result
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', edgecolors='k', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# Evaluate the silhouette score for k-means clustering
silhouette_kmeans = silhouette_score(X_std, kmeans_labels)
print(f'Silhouette Score (K-Means): {silhouette_kmeans}')

# Spectral Clustering
spectral = SpectralClustering(n_clusters=3, random_state=42, affinity='nearest_neighbors')
spectral_labels = spectral.fit_predict(X_std)

# Visualize the spectral clustering result
plt.scatter(X[:, 0], X[:, 1], c=spectral_labels, cmap='viridis', edgecolors='k', s=50)
plt.title('Spectral Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Evaluate the silhouette score for spectral clustering
silhouette_spectral = silhouette_score(X_std, spectral_labels)
print(f'Silhouette Score (Spectral Clustering): {silhouette_spectral}')
