# Parallel Merge Sort using CUDA

This project demonstrates a parallel merge sort implementation using CUDA, a parallel computing platform and API. It leverages the power of GPU parallelism to efficiently sort large arrays.

## Prerequisites

Before you begin, ensure you have the following installed:

- CUDA Toolkit
- C++ Compiler (e.g., g++)
- Make sure your system supports CUDA and has compatible GPU hardware.

## How to Compile

1. Open a terminal window.
2. Navigate to the project directory.
3. Compile the code using the following command:

   ```bash
   nvcc parallel_merge_sort.cu -o merge_sort
   ```

## How to Run

1. After compiling, run the executable:

   ```bash
   ./merge_sort
   ```

## Working of the Code

1. The main function initializes an array with random values.

2. The array is transferred from the host (CPU) to the device (GPU) using `cudaMemcpy`.

3. The `mergeSort` kernel is launched with the initial parameters for the entire array.

4. The `mergeSort` kernel recursively divides the array into smaller sub-arrays and sorts them using the `merge` kernel.

5. The `merge` kernel merges two sorted sub-arrays back into a sorted array.

6. The sorted array is transferred back from the device to the host.

7. Finally, the sorted array is printed to the console.
