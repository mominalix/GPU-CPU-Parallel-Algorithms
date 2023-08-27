import numpy as np
import scipy.sparse as sp

# Define a sparse matrix in Compressed Sparse Column (CSC) format
# You can load a real sparse matrix or create a synthetic one
rows = np.array([0, 1, 2, 2, 3])
cols = np.array([0, 1, 2, 3, 3])
data = np.array([1, 2, 3, 4, 5])
matrix = sp.csc_matrix((data, (rows, cols)))

# Define a vector for multiplication
vector = np.array([1, 2, 3, 4])

# Perform sparse matrix-vector multiplication in parallel
def parallel_sparse_matrix_vector_multiplication(matrix, vector):
    result = np.zeros(matrix.shape[0])
    for i in range(matrix.shape[1]):
        # Parallelize this loop for improved performance
        for j in range(matrix.indptr[i], matrix.indptr[i + 1]):
            result[matrix.indices[j]] += matrix.data[j] * vector[i]
    return result

# Call the parallel multiplication function
result = parallel_sparse_matrix_vector_multiplication(matrix, vector)

# Print the result
print("Result of Parallel SpMV:", result)
