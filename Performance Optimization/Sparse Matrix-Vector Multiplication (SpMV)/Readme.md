# Sparse Matrix-Vector Multiplication (SpMV) Project

This project showcases three different implementations of the Sparse Matrix-Vector Multiplication (SpMV) algorithm using Python, MPI (using mpi4py library), and OpenCL (using pyopencl library). SpMV is a crucial operation in various scientific computing and data analysis tasks, and these implementations aim to demonstrate parallel processing techniques on different computing platforms.

## Prerequisites

Before running the code, make sure you have the following libraries installed:

- For the **Python** implementation: No additional libraries are required.

- For the **MPI** implementation: Install the mpi4py library with the following command:
  ```
  pip install mpi4py
  ```

- For the **OpenCL** implementation: Install the pyopencl library with the following command:
  ```
  pip install pyopencl
  ```

## File Descriptions

1. **SpMV.py**: This file contains a Python implementation of the sparse matrix-vector multiplication algorithm. It demonstrates the basic algorithm without parallelization.

2. **SpMV_mpi4py.py**: This file contains an MPI implementation of the sparse matrix-vector multiplication algorithm using the mpi4py library. It demonstrates parallelization across multiple processes.

3. **SpMV_pyopencl.py**: This file contains an OpenCL implementation of the sparse matrix-vector multiplication algorithm using the pyopencl library. It demonstrates parallelization on heterogeneous computing platforms.

## Compilation and Execution

1. **Python Implementation**:
   ```
   python SpMV.py
   ```

2. **MPI Implementation**:
   - Compile and run using mpiexec:
     ```
     mpiexec -n <number_of_processes> python SpMV_mpi4py.py
     ```

3. **OpenCL Implementation**:
   - Before running, ensure you have a compatible OpenCL device (e.g., GPU) available.
   - Run the OpenCL implementation:
     ```
     python SpMV_pyopencl.py
     ```

## Functionality

- **SpMV.py**: This implementation performs the sparse matrix-vector multiplication using the basic sequential algorithm. It's a reference implementation to understand the core algorithm.

- **SpMV_mpi4py.py**: This implementation uses MPI to parallelize the sparse matrix-vector multiplication across multiple processes. It showcases how to distribute the computation across a distributed environment.

- **SpMV_pyopencl.py**: This implementation demonstrates parallelization using OpenCL, utilizing the power of heterogeneous computing platforms. It showcases the potential performance gains achieved through parallel processing on compatible devices.

Feel free to explore, experiment, and contribute to enhance these implementations for various computing scenarios.

