#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

const int ARRAY_SIZE = 1024;
const int THREADS_PER_BLOCK = 256;

// CUDA kernel for merging two sorted arrays
__global__ void merge(int *input, int *temp, int left, int middle, int right) {
    int i = left + threadIdx.x + blockIdx.x * blockDim.x;
    int j = middle + threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (i < middle && (j >= right || input[i] <= input[j]))
        temp[tid] = input[i];
    else if (j < right)
        temp[tid] = input[j];
    
    __syncthreads();
    input[tid] = temp[tid];
}

// CUDA kernel for merge sort
__global__ void mergeSort(int *input, int *temp, int left, int right) {
    if (right - left < 2)
        return;

    int middle = (left + right) / 2;
    mergeSort<<<1, 1>>>(input, temp, left, middle);
    mergeSort<<<1, 1>>>(input, temp, middle, right);
    merge<<<(right - left + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(input, temp, left, middle, right);
}

int main() {
    int *hostArray, *deviceArray, *tempArray;
    int arrayBytes = ARRAY_SIZE * sizeof(int);
    
    hostArray = new int[ARRAY_SIZE];
    cudaMalloc((void **)&deviceArray, arrayBytes);
    cudaMalloc((void **)&tempArray, arrayBytes);
    
    // Initialize array with random values
    srand(static_cast<unsigned>(time(0)));
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        hostArray[i] = rand() % 1000;
    }
    
    cudaMemcpy(deviceArray, hostArray, arrayBytes, cudaMemcpyHostToDevice);
    
    mergeSort<<<1, 1>>>(deviceArray, tempArray, 0, ARRAY_SIZE);
    
    cudaMemcpy(hostArray, deviceArray, arrayBytes, cudaMemcpyDeviceToHost);
    
    // Print sorted array
    std::cout << "Sorted Array: ";
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        std::cout << hostArray[i] << " ";
    }
    std::cout << std::endl;
    
    delete[] hostArray;
    cudaFree(deviceArray);
    cudaFree(tempArray);
    
    return 0;
}
