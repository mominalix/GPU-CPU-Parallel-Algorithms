#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cuda_runtime.h>
#include <GL/glut.h> // OpenGL library

const int numParticles = 1000000;
const int threadsPerBlock = 256;

// Kernel function for updating particle positions on GPU
__global__ void updateParticles(float* positions, float* velocities, float deltaTime) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles) {
        // Update position using velocity and deltaTime
        positions[idx] += velocities[idx] * deltaTime;
    }
}

// OpenGL display function for particle visualization
void display() {
    glClear(GL_COLOR_BUFFER_BIT);
    
    // Draw particles as points
    glBegin(GL_POINTS);
    for (int i = 0; i < numParticles; ++i) {
        glVertex2f(positions[i], 0.5f); // Visualize particles along the horizontal axis
    }
    glEnd();
    
    glutSwapBuffers();
}

int main(int argc, char** argv) {
    // Initialize OpenGL
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(800, 600);
    glutCreateWindow("Particle Simulation Visualization");

    // Initialize particle positions and velocities
    float* positions;
    float* velocities;
    
    cudaMallocManaged(&positions, numParticles * sizeof(float));
    cudaMallocManaged(&velocities, numParticles * sizeof(float));
    
    for (int i = 0; i < numParticles; ++i) {
        positions[i] = static_cast<float>(rand()) / RAND_MAX;
        velocities[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;
    }

    // Simulation parameters
    float deltaTime = 0.01f;
    int numBlocks = (numParticles + threadsPerBlock - 1) / threadsPerBlock;

    // Run simulation for a number of steps
    for (int step = 0; step < 1000; ++step) {
        updateParticles<<<numBlocks, threadsPerBlock>>>(positions, velocities, deltaTime);
        cudaDeviceSynchronize(); // Wait for kernel to finish
        
        // Display particles using OpenGL
        glutDisplayFunc(display);
        glutMainLoopEvent();
    }
   

    // Clean up
    cudaFree(positions);
    cudaFree(velocities);

    return 0;
}
