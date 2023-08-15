#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// GPU kernel function for image convolution
__global__ void convolutionGPU(const float* image, const float* kernel, float* output,
                               int imageWidth, int kernelSize) {
    // Calculate the index of the current thread in the grid
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the thread index is within the image width
    if (idx < imageWidth) {
        float result = 0.0f;
        int halfKernelSize = kernelSize / 2;

        // Perform convolution for the current pixel
        for (int j = -halfKernelSize; j <= halfKernelSize; j++) {
            int imageIdx = idx + j;

            // Check boundaries to avoid accessing out-of-bounds memory
            if (imageIdx >= 0 && imageIdx < imageWidth) {
                result += image[imageIdx] * kernel[j + halfKernelSize];
            }
        }

        // Store the result of convolution in the output array
        output[idx] = result;
    }
}

int main() {
    int imageWidth = 1024;
    int imageSize = imageWidth * sizeof(float);
    int kernelSize = 5;

    // Allocate memory on the host for image, kernel, and output
    float* h_image = (float*)malloc(imageSize);
    float* h_kernel = (float*)malloc(kernelSize * sizeof(float));
    float* h_output = (float*)malloc(imageSize);

    // Initialize image and kernel data (not shown here)

    // Allocate memory on the GPU for image, kernel, and output
    float* d_image, *d_kernel, *d_output;
    cudaMalloc((void**)&d_image, imageSize);
    cudaMalloc((void**)&d_kernel, kernelSize * sizeof(float));
    cudaMalloc((void**)&d_output, imageSize);

    // Copy data from host to GPU
    cudaMemcpy(d_image, h_image, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernelSize * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate grid and block dimensions for kernel launch
    int threadsPerBlock = 256;
    int blocksPerGrid = (imageWidth + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the GPU kernel for convolution
    convolutionGPU<<<blocksPerGrid, threadsPerBlock>>>(d_image, d_kernel, d_output, imageWidth, kernelSize);

    // Copy convolution result back from GPU to host
    cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost);

    // Clean up: free memory on GPU and host
    cudaFree(d_image);
    cudaFree(d_kernel);
    cudaFree(d_output);
    free(h_image);
    free(h_kernel);
    free(h_output);

    return 0;
}
