# GPU-Accelerated Image Convolution

This repository contains a simple example of GPU-accelerated image convolution using CUDA. Image convolution is a fundamental operation in image processing and computer vision, and using GPUs can significantly speed up this process.

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed

## Compilation

1. Ensure you have the CUDA Toolkit installed on your system.
2. Open a terminal and navigate to the project directory.
3. Compile the code using the following command:

   ```
   nvcc image_convolution.cu -o image_convolution
   ```

## Usage

1. After compilation, run the executable:

   ```
   ./image_convolution
   ```

2. The program will perform image convolution using GPU acceleration and display the results.

## Functionality

The code demonstrates GPU-accelerated image convolution, a common operation in image processing. The convolution process involves sliding a small matrix (kernel) over an input image to generate an output image. This operation is used for various tasks such as blurring, edge detection, and feature extraction.

In this code, the GPU kernel function `convolutionGPU` performs convolution for each pixel of the image. The code uses CUDA to parallelize the computation across multiple threads on the GPU, leading to faster processing compared to traditional CPU-based convolution.
