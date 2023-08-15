# CUDA QuickSort Implementation

This repository contains a parallel implementation of the QuickSort algorithm using CUDA. QuickSort is a widely-used sorting algorithm that can be efficiently parallelized using the power of GPUs.

## Compilation

To compile the code, you'll need the NVIDIA CUDA Toolkit installed on your system. If you haven't already, you can download and install it from the [NVIDIA CUDA website](https://developer.nvidia.com/cuda-downloads).

Once you have the CUDA Toolkit installed, navigate to the repository directory and execute the following commands:

```bash
nvcc parallel_quick_sort.cu -o quicksort
```

This will compile the code and generate an executable named `quicksort`.

## Running the Code

After compilation, you can run the code by executing the generated executable:

```bash
./quicksort
```

The program will perform a parallel QuickSort on a randomly generated array of integers. It will then verify if the array has been sorted correctly.

## Understanding the Code

- The `parallel_quick_sort.cu` file contains the main CUDA QuickSort implementation.
- The `quickSort` kernel function is responsible for performing the parallel sorting.
- The `main` function initializes the array, allocates memory on the GPU, launches the `quickSort` kernel, transfers data back to the host, and verifies the sorting.

