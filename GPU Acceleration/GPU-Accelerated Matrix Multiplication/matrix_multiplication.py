import numpy as np
import numba.cuda as cuda

# Generate random matrices
size = 128
matrix_a = np.random.rand(size, size)
matrix_b = np.random.rand(size, size)
result = np.zeros((size, size))

@cuda.jit
def matrix_multiply(a, b, result):
    i, j = cuda.grid(2)
    if i < result.shape[0] and j < result.shape[1]:
        temp = 0
        for k in range(a.shape[1]):
            temp += a[i, k] * b[k, j]
        result[i, j] = temp

# Set up grid and block dimensions
threadsperblock = (16, 16)
blockspergrid_x = (size + threadsperblock[0] - 1) // threadsperblock[0]
blockspergrid_y = (size + threadsperblock[1] - 1) // threadsperblock[1]
blockspergrid = (blockspergrid_x, blockspergrid_y)

# Transfer data to GPU
d_matrix_a = cuda.to_device(matrix_a)
d_matrix_b = cuda.to_device(matrix_b)
d_result = cuda.to_device(result)

# Perform matrix multiplication on GPU
matrix_multiply[blockspergrid, threadsperblock](d_matrix_a, d_matrix_b, d_result)

# Transfer result back to CPU
d_result.copy_to_host(result)

print("Matrix A:")
print(matrix_a)
print("\nMatrix B:")
print(matrix_b)
print("\nResult:")
print(result)
