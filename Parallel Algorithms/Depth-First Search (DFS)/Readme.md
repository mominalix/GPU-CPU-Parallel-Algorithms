# Parallel Graph Traversal using Depth-First Search (DFS)

This repository contains Python code for performing parallel graph traversal using the Depth-First Search (DFS) algorithm. The code utilizes the `multiprocessing` library to achieve concurrent exploration of graph nodes, resulting in faster traversal times.

## Prerequisites

Before running the code, ensure you have the following:

- Python (>= 3.6)
- `multiprocessing` library (comes pre-installed with Python)

## How to Run


Open the terminal and run the following command to execute the DFS traversal.

```bash
python parallel_dfs.py
```

## Functionality

The code performs a parallel depth-first search traversal on a sample graph represented as a dictionary of nodes and their neighbors. Each process explores a subset of nodes concurrently, enabling efficient exploration of the entire graph. The traversal starts from a specified node ('A' in this case) and explores its neighbors in parallel.

The `parallel_dfs` function initializes a stack for each process and maintains a set of visited nodes to avoid revisiting already explored nodes. Each process prints the node it is currently visiting along with its process name.

