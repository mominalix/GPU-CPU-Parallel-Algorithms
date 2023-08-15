#include <stdio.h>

__global__ void parallelPrefixSum(int *input, int *output, int n) {
    extern __shared__ int temp[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n) {
        temp[threadIdx.x] = input[idx];
        __syncthreads();

        for (int stride = 1; stride < blockDim.x; stride *= 2) {
            int index = (threadIdx.x + 1) * 2 * stride - 1;
            if (index < blockDim.x) {
                temp[index] += temp[index - stride];
            }
            __syncthreads();
        }

        output[idx] = temp[threadIdx.x];
    }
}

int main() {
    int n = 8;
    int input[n] = {3, 1, 7, 0, 4, 1, 6, 3};
    int output[n];

    int *d_input, *d_output;
    cudaMalloc((void**)&d_input, n * sizeof(int));
    cudaMalloc((void**)&d_output, n * sizeof(int));
    
    cudaMemcpy(d_input, input, n * sizeof(int), cudaMemcpyHostToDevice);

    int block_size = 4;
    int grid_size = (n + block_size - 1) / block_size;

    parallelPrefixSum<<<grid_size, block_size, block_size * sizeof(int)>>>(d_input, d_output, n);

    cudaMemcpy(output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        printf("%d ", output[i]);
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
