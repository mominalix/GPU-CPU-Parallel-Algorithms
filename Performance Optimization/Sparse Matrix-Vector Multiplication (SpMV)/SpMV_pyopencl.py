import pyopencl as cl
import numpy as np
import scipy.sparse as sp

# Set up OpenCL context and command queue
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

# Define a sparse matrix in Compressed Sparse Column (CSC) format
# You can load a real sparse matrix or create a synthetic one
rows = np.array([0, 1, 2, 2, 3])
cols = np.array([0, 1, 2, 3, 3])
data = np.array([1, 2, 3, 4, 5])
matrix = sp.csc_matrix((data, (rows, cols)))

# Define a vector for multiplication
vector = np.array([1, 2, 3, 4])

# Create OpenCL buffers for matrix and vector
matrix_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=matrix.data)
row_indices_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=matrix.indptr)
column_indices_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=matrix.indices)
vector_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=vector)

# Create an OpenCL buffer for the result
result_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=matrix.shape[0] * np.dtype(np.float32).itemsize)

# Load and compile OpenCL kernel
kernel_code = """
__kernel void spmv_kernel(__global const float* matrix, __global const int* row_indices,
                          __global const int* column_indices, __global const float* vector,
                          __global float* result, int num_rows)
{
    int gid = get_global_id(0);
    if (gid < num_rows)
    {
        float sum = 0.0f;
        for (int j = row_indices[gid]; j < row_indices[gid + 1]; ++j)
            sum += matrix[j] * vector[column_indices[j]];
        result[gid] = sum;
    }
}
"""
program = cl.Program(context, kernel_code).build()

# Execute the kernel
result = np.zeros(matrix.shape[0], dtype=np.float32)
program.spmv_kernel(queue, result.shape, None, matrix_buf, row_indices_buf,
                    column_indices_buf, vector_buf, result_buf, np.int32(matrix.shape[0]))

# Read the result back from the OpenCL buffer into the 'result' NumPy array
cl.enqueue_copy(queue, result, result_buf).wait()

# Print the result
print("Result of SpMV using OpenCL:", result)
