#include <stdio.h>

// Define matrix dimensions
#define N 4
#define M 4

// Kernel for matrix-vector multiplication
__global__ void matrixVectorMul(float *matrix, float *vector, float *result) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N) {
        float sum = 0.0f;
        for (int col = 0; col < M; col++) {
            sum += matrix[row * M + col] * vector[col];
        }
        result[row] = sum;
    }
}

int main() {
    // Initialize matrix and vector
    float matrix[N * M];
    float vector[M];
    float result[N];
    
    for (int i = 0; i < N * M; i++) {
        matrix[i] = i + 1;  // Initialize matrix elements
    }
    
    for (int i = 0; i < M; i++) {
        vector[i] = i + 1;  // Initialize vector elements
    }

    // Declare device pointers
    float *d_matrix, *d_vector, *d_result;
    
    // Allocate memory on device
    cudaMalloc((void**)&d_matrix, N * M * sizeof(float));
    cudaMalloc((void**)&d_vector, M * sizeof(float));
    cudaMalloc((void**)&d_result, N * sizeof(float));
    
    // Transfer data from host to device
    cudaMemcpy(d_matrix, matrix, N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, vector, M * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    matrixVectorMul<<<blocksPerGrid, threadsPerBlock>>>(d_matrix, d_vector, d_result);
    
    // Transfer result from device to host
    cudaMemcpy(result, d_result, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    printf("Matrix-Vector Multiplication Result:\n");
    for (int i = 0; i < N; i++) {
        printf("%.2f ", result[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_matrix);
    cudaFree(d_vector);
    cudaFree(d_result);

    return 0;
}
