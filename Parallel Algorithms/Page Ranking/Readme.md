# Parallelized Page Rank Algorithm

This repository contains a parallelized implementation of the Page Rank algorithm using Python and the `multiprocessing` library. Page Rank is a widely used algorithm to rank web pages in search engine results. The parallel implementation demonstrated here significantly speeds up the process of calculating page ranks for a large web graph.

## How to Use

### Prerequisites

- Python (>= 3.6)
- NumPy library

### Usage

1. Open the terminal/command prompt.

2. Run the Page Rank algorithm with the following command:

   ```bash
   python page_rank.py
   ```

   This command will execute the parallelized Page Rank algorithm on a randomly generated web graph.

### Configuration

You can configure the behavior of the Page Rank algorithm by modifying the parameters in the `page_rank.py` file:

- `num_pages`: The number of web pages in the graph.
- `damping_factor`: The damping factor used in the Page Rank formula.
- `iterations`: The number of iterations for Page Rank calculation.
- `num_processes`: The number of parallel processes to use for computation.

## Functionality

The parallelized Page Rank algorithm showcased in this repository follows these steps:

1. **Generate Web Graph**: A random web graph (adjacency matrix) is generated using the `generate_web_graph` function.

2. **Parallel Page Rank Calculation**: The `parallel_page_rank` function divides the web graph into subsets, distributes them across multiple processes, and calculates Page Ranks concurrently.

3. **Combining Results**: The results from different processes are combined to obtain the final page ranks.

4. **Print Page Ranks**: The calculated page ranks are printed to the console.

By utilizing parallel processing, this implementation achieves faster Page Rank computation for large web graphs, demonstrating the power of parallelization in high-performance computing.