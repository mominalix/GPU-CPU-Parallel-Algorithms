import numpy as np
from multiprocessing import Pool

# Generate a random web graph (adjacency matrix)
def generate_web_graph(num_pages):
    return np.random.randint(2, size=(num_pages, num_pages))

# Perform Page Rank calculation on a subset of pages
def calculate_page_rank(page_indices, graph, damping_factor, iterations):
    num_pages = len(graph)
    ranks = np.ones(num_pages) / num_pages

    for _ in range(iterations):
        new_ranks = np.zeros(num_pages)
        for i in page_indices:
            num_links = np.sum(graph[:, i])
            if num_links == 0:
                new_ranks[i] += damping_factor * ranks[i] / num_pages
            else:
                new_ranks += damping_factor * ranks[i] / num_links
        ranks = new_ranks

    return ranks

def parallel_page_rank(graph, damping_factor, iterations, num_processes):
    num_pages = len(graph)
    chunk_size = num_pages // num_processes
    pool = Pool(processes=num_processes)

    page_indices = [list(range(i * chunk_size, (i + 1) * chunk_size)) for i in range(num_processes)]

    results = pool.starmap(calculate_page_rank, [(indices, graph, damping_factor, iterations) for indices in page_indices])
    pool.close()
    pool.join()

    page_ranks = np.concatenate(results)

    return page_ranks

if __name__ == '__main__':
    num_pages = 1000
    damping_factor = 0.85
    iterations = 20
    num_processes = 4

    web_graph = generate_web_graph(num_pages)
    page_ranks = parallel_page_rank(web_graph, damping_factor, iterations, num_processes)

    print("Page Ranks:", page_ranks)
