import numpy as np
import cupy as cp

def gpu_kmeans(data, n_clusters, max_iters=100, tol=1e-4):
    data = cp.array(data)
    centroids = cp.random.choice(data, size=n_clusters, replace=False)
    
    for _ in range(max_iters):
        # Assign each point to the nearest centroid
        distances = cp.linalg.norm(data[:, cp.newaxis] - centroids, axis=2)
        labels = cp.argmin(distances, axis=1)
        
        new_centroids = cp.array([data[labels == i].mean(axis=0) for i in range(n_clusters)])
        
        if cp.all(cp.abs(new_centroids - centroids) < tol):
            break
        
        centroids = new_centroids
    
    return labels.get(), centroids.get()

# Generate random data points
np.random.seed(0)
data = np.random.rand(1000, 2)

# Perform k-means clustering on GPU
labels, centroids = gpu_kmeans(data, n_clusters=3)

print("Cluster Labels:", labels)
print("Cluster Centroids:", centroids)
