# GPU-Accelerated Particle Simulation with OpenGL Visualization

This repository contains a CUDA-based particle simulation code that leverages GPU acceleration and utilizes OpenGL for visualization. The code simulates the movement of particles over time and provides a basic visualization of the particle positions.

## Compilation and Running

1. **Prerequisites**: Make sure you have CUDA and OpenGL installed on your system.

2. **Compilation**: Compile the code using the following command:

   ```bash
   nvcc -o particle_simulation particle_simulation.cu -lGL -lGLU -lglut
   ```

3. **Running**: Run the compiled executable:

   ```bash
   ./particle_simulation
   ```

## Functionality

- The code initializes a large number of particles with random positions and velocities.
- The simulation runs for a specified number of steps, updating particle positions using GPU-accelerated CUDA kernels.
- After each simulation step, the particle positions are visualized using OpenGL, providing a basic representation of particle movement.

