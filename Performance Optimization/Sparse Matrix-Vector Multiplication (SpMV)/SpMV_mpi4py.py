from mpi4py import MPI
import numpy as np
import scipy.sparse as sp

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Define a sparse matrix in Compressed Sparse Column (CSC) format
rows = np.array([0, 1, 2, 2, 3])
cols = np.array([0, 1, 2, 3, 3])
data = np.array([1, 2, 3, 4, 5])
matrix = sp.csc_matrix((data, (rows, cols)))

# Define a vector for multiplication
vector = np.array([1, 2, 3, 4])

# Distribute rows among processes
local_rows = np.array_split(rows, size)[rank]
local_result = np.zeros(len(local_rows))

# Parallel sparse matrix-vector multiplication using MPI
for i in range(matrix.shape[1]):
    for j in range(matrix.indptr[i], matrix.indptr[i + 1]):
        if matrix.indices[j] in local_rows:
            local_result[np.where(local_rows == matrix.indices[j])] += matrix.data[j] * vector[i]

# Gather results from all processes
all_results = comm.gather(local_result, root=0)

# Process 0 combines and prints the final result
if rank == 0:
    result = np.zeros(matrix.shape[0])
    for res in all_results:
        result[local_rows] += res
    print("Result of Parallel SpMV using MPI:", result)
