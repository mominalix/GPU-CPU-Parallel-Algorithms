# GPU-Accelerated k-means Clustering

This repository contains a Python code implementation of the k-means clustering algorithm using GPU acceleration. K-means is a popular unsupervised machine learning algorithm used for clustering data points into distinct groups. By leveraging GPU acceleration, this implementation offers faster execution, particularly for larger datasets.

## Prerequisites

Before you begin, ensure you have the following requirements:

- Python 3.6+
- `cupy` library (for GPU acceleration)

You can install the `cupy` library using the following command:

```bash
pip install cupy
```

## Usage

1. Open a terminal or command prompt and navigate to the repository directory.

2. Run the script `gpu_kmeans.py` using the following command:

```bash
python gpu_kmeans.py
```

## Functionality

The `gpu_kmeans.py` script performs k-means clustering on a randomly generated dataset. Here's what the script does:

1. Imports the necessary libraries: `numpy` for array manipulation and `cupy` for GPU-accelerated computations.
2. Defines the `gpu_kmeans` function which performs k-means clustering using GPU acceleration.
3. Generates random data points using `numpy`.
4. Calls the `gpu_kmeans` function to cluster the data into specified clusters.
5. Prints the resulting cluster labels and centroids.

The GPU-accelerated k-means algorithm aims to find the optimal centroids for the given data points by iteratively updating assignments and centroids until convergence. This implementation leverages the power of GPUs to significantly speed up the computation, making it suitable for handling large datasets efficiently.

