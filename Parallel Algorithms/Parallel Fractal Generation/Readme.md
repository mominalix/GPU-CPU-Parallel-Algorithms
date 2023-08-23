# Parallelized Mandelbrot Fractal Generation

This repository contains a Python program that generates the Mandelbrot fractal using parallel processing techniques. The fractal is a mesmerizing visual representation of complex numbers and their behavior under iteration.

## Prerequisites

- Python 3.x
- `numpy` and `matplotlib` libraries (can be installed using `pip`)

## How to Use

1. **Install Dependencies**

   Ensure you have the required dependencies installed:

   ```bash
   pip install numpy matplotlib
   ```

2. **Run the Code**

   Run the main script using Python:

   ```bash
   python fractal_generation.py
   ```

3. **View the Generated Fractal**

   The generated Mandelbrot fractal will be displayed using `matplotlib`. The visual will showcase intricate patterns and complex structures.

## Functionality

The code in `fractal_generation.py` achieves the following:

- It generates the Mandelbrot fractal using parallel processing.
- The `mandelbrot_row` function calculates a row of the fractal matrix based on the provided parameters.
- The `generate_mandelbrot` function distributes the computation of rows across multiple processes using the `multiprocessing` library.
- The resulting fractal visualization demonstrates the fascinating and intricate patterns of the Mandelbrot set.

## Customization

You can customize the following parameters in the `fractal_generation.py` script:

- `width` and `height`: Dimensions of the generated fractal image.
- `xmin`, `xmax`, `ymin`, `ymax`: Range of complex coordinates to visualize.
- `max_iter`: Maximum number of iterations to determine fractal points.
- `num_processes`: Number of parallel processes to utilize.
