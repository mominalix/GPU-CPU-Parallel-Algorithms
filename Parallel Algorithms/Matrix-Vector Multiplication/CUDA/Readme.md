## CUDA Matrix-Vector Multiplication

This repository contains a CUDA program that demonstrates matrix-vector multiplication using parallel processing on a GPU. This example serves as a starting point for learning CUDA programming and exploiting GPU parallelism.

### Prerequisites

Before you begin, ensure you have the following:

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- Compiler that supports CUDA (e.g., nvcc)

### Compilation

To compile the CUDA program, follow these steps:

1. Open a terminal window.

2. Navigate to the directory containing the code files.

3. Run the following command to compile the code using `nvcc`:

   ```
   nvcc parallel_matrix_vector_multiplication.cu -o matrix_vector_multiplication
   ```

### Running the Program

After successful compilation, run the executable to perform matrix-vector multiplication using CUDA:

1. Run the compiled executable:

   ```
   ./parallel_matrix_vector_multiplication
   ```

2. The program will execute and output the result of the matrix-vector multiplication.

### Code Explanation

The CUDA program consists of the following components:

- `parallel_matrix_vector_multiplication.cu`: The main source code file that implements matrix-vector multiplication using CUDA.

The code performs the following steps:

1. Initializes the matrix and vector on the host.

2. Allocates memory for the matrix, vector, and result on the GPU.

3. Transfers data from the host to the GPU.

4. Launches a kernel function that performs matrix-vector multiplication in parallel on the GPU.

5. Transfers the result back from the GPU to the host.

6. Prints the result of the matrix-vector multiplication.
