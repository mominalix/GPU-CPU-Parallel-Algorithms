import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

def mandelbrot_row(args):
    y, width, height, xmin, xmax, ymin, ymax, max_iter = args
    row = np.empty(width, dtype=int)
    for x in range(width):
        zx, zy = x * (xmax - xmin) / (width - 1) + xmin, y * (ymax - ymin) / (height - 1) + ymin
        c = zx + zy * 1j
        z = c
        for i in range(max_iter):
            if abs(z) > 2.0:
                row[x] = i
                break
            z = z * z + c
        else:
            row[x] = max_iter
    return row

def generate_mandelbrot(width, height, xmin, xmax, ymin, ymax, max_iter, num_processes):
    pool = multiprocessing.Pool(processes=num_processes)
    args_list = [(y, width, height, xmin, xmax, ymin, ymax, max_iter) for y in range(height)]
    rows = pool.map(mandelbrot_row, args_list, chunksize=1)
    pool.close()
    pool.join()

    fractal = np.vstack(rows)
    return fractal

if __name__ == "__main__":
    width, height = 800, 800
    xmin, xmax = -2.0, 1.0
    ymin, ymax = -1.5, 1.5
    max_iter = 300
    num_processes = 4

    fractal = generate_mandelbrot(width, height, xmin, xmax, ymin, ymax, max_iter, num_processes)

    plt.imshow(fractal, cmap="hot", extent=(xmin, xmax, ymin, ymax))
    plt.colorbar()
    plt.title("Mandelbrot Fractal")
    plt.show()
