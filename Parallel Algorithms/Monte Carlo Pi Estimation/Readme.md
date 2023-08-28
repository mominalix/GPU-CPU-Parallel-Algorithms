# Parallelized Monte Carlo Pi Estimation

This repository contains a Python implementation of the Parallelized Monte Carlo Pi Estimation. This technique uses a parallel approach to estimate the value of π using random points inside a unit square.

## Functionality

The program simulates the process of randomly throwing points within a unit square and calculates the ratio of points that fall inside the unit circle to the total number of points. By multiplying this ratio by 4, the program estimates the value of π.

The simulation is parallelized using multiple threads to distribute the workload and improve computation speed.

## Compilation and Execution

1. Make sure you have Python 3.x installed on your system.

2. Clone this repository or download the `monte_carlo_pi.py` file.

3. Open a terminal/command prompt and navigate to the directory where the `monte_carlo_pi.py` file is located.

4. Run the program using the following command:

   ```
   python monte_carlo_pi.py
   ```

5. The program will execute, and you will see the estimated value of π printed to the console.

## Configuration

You can adjust the `num_threads` and `total_points` variables in the `monte_carlo_pi.py` file to control the number of threads and the total number of points used in the simulation. Experiment with different values to observe the effect on the estimation accuracy and computation time.

