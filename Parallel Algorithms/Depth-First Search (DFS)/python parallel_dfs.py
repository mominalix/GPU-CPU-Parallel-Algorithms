import multiprocessing

# Sample Graph
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': [],
    'F': []
}

def parallel_dfs(node):
    stack = [node]
    visited = set()

    while stack:
        current_node = stack.pop()
        if current_node not in visited:
            print(f"Visiting node {current_node} on process {multiprocessing.current_process().name}")
            visited.add(current_node)
            stack.extend(neighbor for neighbor in graph[current_node] if neighbor not in visited)

if __name__ == "__main__":
    start_node = 'A'
    processes = []

    for _ in range(multiprocessing.cpu_count()):
        process = multiprocessing.Process(target=parallel_dfs, args=(start_node,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    print("Traversal completed.")
