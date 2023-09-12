import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

# Function to compute a single row of the Mandelbrot fractal
def mandelbrot_row(args):
    # Unpack the arguments
    y, width, height, xmin, xmax, ymin, ymax, max_iter = args
    
    # Initialize an empty array to store the values for this row
    row = np.empty(width, dtype=int)
    
    # Loop through each pixel in the row
    for x in range(width):
        # Calculate the complex number corresponding to this pixel
        zx, zy = x * (xmax - xmin) / (width - 1) + xmin, y * (ymax - ymin) / (height - 1) + ymin
        c = zx + zy * 1j
        z = c
        
        # Iterate to check if the point belongs to the Mandelbrot set
        for i in range(max_iter):
            if abs(z) > 2.0:
                row[x] = i  # Store the number of iterations it took to escape
                break
            z = z * z + c
        else:
            row[x] = max_iter  # If it didn't escape within max_iter, set to max_iter
        
    return row

# Function to generate the Mandelbrot fractal using multiple processes
def generate_mandelbrot(width, height, xmin, xmax, ymin, ymax, max_iter, num_processes):
    # Create a pool of worker processes
    pool = multiprocessing.Pool(processes=num_processes)
    
    # Create a list of arguments for each row
    args_list = [(y, width, height, xmin, xmax, ymin, ymax, max_iter) for y in range(height)]
    
    # Use the pool to calculate Mandelbrot rows in parallel
    rows = pool.map(mandelbrot_row, args_list, chunksize=1)
    
    # Close the pool of worker processes
    pool.close()
    pool.join()

    # Combine the rows into the final Mandelbrot fractal
    fractal = np.vstack(rows)
    return fractal

if __name__ == "__main__":
    # Define parameters for the Mandelbrot fractal
    width, height = 800, 800
    xmin, xmax = -2.0, 1.0
    ymin, ymax = -1.5, 1.5
    max_iter = 300
    num_processes = 4  # Number of processes to use for parallel computation

    # Generate the Mandelbrot fractal
    fractal = generate_mandelbrot(width, height, xmin, xmax, ymin, ymax, max_iter, num_processes)

    # Display the Mandelbrot fractal using Matplotlib
    plt.imshow(fractal, cmap="hot", extent=(xmin, xmax, ymin, ymax))
    plt.colorbar()
    plt.title("Mandelbrot Fractal")
    plt.show()
