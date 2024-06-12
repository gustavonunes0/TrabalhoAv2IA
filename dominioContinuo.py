import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

p = 20 
A = 10 
population_size = 50
max_generations = 100
mutation_rate = 0.01 
num_executions = 100 

def rastrigin(x):
    return A * p + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def create_population(size, dim):
    return np.random.randint(2, size=(size, dim))

def calculate_fitness(population):
    fitness = np.zeros(len(population))
    for i, individual in enumerate(population):
        x = decode_individual(individual)
        fitness[i] = rastrigin(x) + 1  
    return fitness

def decode_individual(individual):
    scale = 20 / (2**p - 1) 
    return -10 + scale * np.dot(individual, 2**np.arange(p)[::-1])

def roulette_selection(population, fitness):
    fitness_sum = np.sum(fitness)
    probabilities = fitness / fitness_sum
    selected_indices = np.random.choice(len(population), size=len(population), p=probabilities)
    return population[selected_indices]

def one_point_crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1))
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

def mutation(individual, mutation_rate):
    mutated_individual = individual.copy()
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            mutated_individual[i] = 1 - mutated_individual[i] 
    return mutated_individual

def execute_genetic_algorithm():
    all_best_fitnesses = []
    all_min_fitnesses = []
    all_max_fitnesses = []
    all_mean_fitnesses = []
    all_std_fitnesses = []

    for execution in range(num_executions):
        population = create_population(population_size, p)
        best_solution = np.inf

        for generation in range(max_generations):
            fitness = calculate_fitness(population)
            best_fitness = np.min(fitness)
            all_best_fitnesses.append(best_fitness)

            if best_fitness < best_solution:
                best_solution = best_fitness

            if best_solution <= 1.0: 
                break

            selected = roulette_selection(population, fitness)
            offspring = []
            for i in range(0, len(selected), 2):
                child1, child2 = one_point_crossover(selected[i], selected[(i + 1) % len(selected)])
                child1 = mutation(child1, mutation_rate)
                child2 = mutation(child2, mutation_rate)
                offspring.append(child1)
                offspring.append(child2)

            population = np.array(offspring)

        all_min_fitnesses.append(np.min(all_best_fitnesses))
        all_max_fitnesses.append(np.max(all_best_fitnesses))
        all_mean_fitnesses.append(np.mean(all_best_fitnesses))
        all_std_fitnesses.append(np.std(all_best_fitnesses))

    results = {
        "Menor valor de aptidão": all_min_fitnesses,
        "Maior valor de aptidão": all_max_fitnesses,
        "Valor médio de aptidão": all_mean_fitnesses,
        "Desvio padrão de valor de aptidão": all_std_fitnesses
    }

    results_df = pd.DataFrame(results)
    return results_df

results_df = execute_genetic_algorithm()
print(results_df)
