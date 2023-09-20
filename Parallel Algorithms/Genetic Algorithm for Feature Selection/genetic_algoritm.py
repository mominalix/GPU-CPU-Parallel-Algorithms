import numpy as np
import multiprocessing

# Define fitness function (replace with your actual fitness evaluation)
def fitness(individual):
    return np.sum(individual)  #Example: maximize the sum of selected features

# Genetic algorithm operations
def initialize_population(pop_size, num_features):
    return np.random.randint(2, size=(pop_size, num_features))

def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1) - 1)
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] = 1 - individual[i]
    return individual

def evaluate_population(population):
    with multiprocessing.Pool() as pool:
        fitness_scores = pool.map(fitness, population)
    return fitness_scores

# Parallel Genetic Algorithm
def parallel_genetic_algorithm(pop_size, num_features, generations, mutation_rate):
    population = initialize_population(pop_size, num_features)

    for generation in range(generations):
        fitness_scores = evaluate_population(population)

        # Normalize fitness scores
        normalized_fitness = fitness_scores / sum(fitness_scores)

        cumulative_probs = np.cumsum(normalized_fitness)

        selected_parents = []
        for _ in range(pop_size // 2):
            parent1_idx = np.searchsorted(cumulative_probs, np.random.rand())
            parent2_idx = np.searchsorted(cumulative_probs, np.random.rand())
            selected_parents.extend([parent1_idx, parent2_idx])

        parents = population[selected_parents]

        next_generation = []

        for i in range(0, len(parents), 2):
            child1, child2 = crossover(parents[i], parents[i + 1])
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            next_generation.extend([child1, child2])

        population = np.array(next_generation)

        best_idx = np.argmax(fitness_scores)
 
        print(f"Generation {generation+1}: Best Fitness = {fitness_scores[best_idx]}")

# Parameters
pop_size = 100
num_features = 20
generations = 50
mutation_rate = 0.05

# Run parallel genetic algorithm
if __name__ == "__main__":
    parallel_genetic_algorithm(pop_size, num_features, generations, mutation_rate)
