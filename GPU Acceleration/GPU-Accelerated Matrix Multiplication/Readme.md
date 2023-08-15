# GPU Accelerated Matrix Multiplication

This repository contains a simple Python code that demonstrates matrix multiplication using GPU acceleration with the `numba` library's CUDA support. The code showcases how to perform matrix multiplication efficiently on a GPU, taking advantage of parallel processing.

## Prerequisites

- Python (>= 3.6)
- `numba` library (install using `pip install numba`)

## How to Compile and Run

1. Make sure you have Python and the `numba` library installed.

2. Navigate to the project directory:

   ```
   cd gpu-matrix-multiplication
   ```

3. Run the code using Python:

   ```
   python matrix_multiplication.py
   ```

## Functionality

The code in `matrix_multiplication.py` performs the following steps:

1. Generates random matrices (`matrix_a` and `matrix_b`) of the specified size.
2. Defines a CUDA kernel `matrix_multiply` using `numba` that calculates matrix multiplication for a given pair of matrices.
3. Sets up grid and block dimensions to determine the number of CUDA threads and blocks.
4. Transfers the matrices and the result to and from the GPU using `numba`'s CUDA support.
5. Prints the original matrices and the resultant matrix after multiplication.

This code demonstrates the basics of GPU-accelerated matrix multiplication. It's a simple example to help you understand the concept and usage of GPU programming using the `numba` library.

