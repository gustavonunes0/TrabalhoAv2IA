import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

population_size = 50
max_generations = 100
optimal_value = 10
num_elites = 5
num_executions = 100

def read_points_from_file(file_path):
    dataframe = pd.read_csv(file_path, header=None)
    return dataframe.values

def create_population(size, num_points):
    return [np.random.permutation(num_points) for _ in range(size)]

def calculate_fitness(chromosome, points):
    total_distance = 0
    for index in range(len(chromosome)):
        current_point = points[chromosome[index]]
        next_point = points[chromosome[(index + 1) % len(chromosome)]]
        distance = np.linalg.norm(current_point - next_point)
        total_distance += distance
    return total_distance

def tournament_selection(population, fitnesses, tournament_size=3):
    selected = []
    while len(selected) < len(population):
        competitors = np.random.choice(len(population), tournament_size, replace=False)
        winner = competitors[np.argmin(fitnesses[competitors])]
        selected.append(population[winner])
    return selected

def two_point_crossover(parent1, parent2):
    size = len(parent1)
    child1, child2 = parent1.copy(), parent2.copy()
    point1, point2 = sorted(np.random.choice(range(1, size - 1), 2, replace=False))
    child1[point1:point2], child2[point1:point2] = parent2[point1:point2], parent1[point1:point2]
    return child1, child2

def swap_mutation(chromosome, mutation_rate=0.05):
    chromosome = np.array(chromosome)
    for index in range(len(chromosome)):
        if np.random.rand() < mutation_rate:
            j = np.random.randint(len(chromosome))
            chromosome[index], chromosome[j] = chromosome[j], chromosome[index]
    return chromosome.tolist()

def apply_elitism(population, fitnesses, num_elites):
    elite_indices = np.argsort(fitnesses)[:num_elites]
    return [population[index] for index in elite_indices]

def plot_tsp_3d(points, chromosome):
    figure = plt.figure(figsize=(10, 8))
    ax = figure.add_subplot(111, projection='3d')
    x_coords, y_coords, z_coords = points[:, 0], points[:, 1], points[:, 2]
    ax.scatter(x_coords, y_coords, z_coords, color='blue', s=100, zorder=5)
    ax.scatter(x_coords[0], y_coords[0], z_coords[0], color='red', s=200, zorder=5, label='Origin')
    for index in range(len(chromosome) - 1):
        start_pos = points[chromosome[index]]
        end_pos = points[chromosome[index + 1]]
        ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], [start_pos[2], end_pos[2]], 'k-', zorder=1)
    ax.plot([points[chromosome[-1]][0], points[0][0]], [points[chromosome[-1]][1], points[0][1]], [points[chromosome[-1]][2], points[0][2]], 'k-', zorder=1)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.legend()
    ax.grid(True)
    plt.show()

def execute_genetic_algorithm(points):
    population = create_population(population_size, len(points))
    all_fitnesses = []
    best_solution = np.inf
    best_chromosome = None

    for generation in range(max_generations):
        fitnesses = np.array([calculate_fitness(individual, points) for individual in population])
        all_fitnesses.extend(fitnesses)
        if np.min(fitnesses) < best_solution:
            best_solution = np.min(fitnesses)
            best_chromosome = population[np.argmin(fitnesses)]
            print(f"Generation {generation}: Best fitness = {best_solution}")

        if best_solution <= optimal_value:
            print("Stopping condition reached.")
            break

        elites = apply_elitism(population, fitnesses, num_elites)
        selected = tournament_selection(population, fitnesses)
        offspring = [two_point_crossover(selected[i], selected[(i + 1) % len(selected)]) for i in range(0, len(selected), 2)]
        population = [swap_mutation(child) for pair in offspring for child in pair]
        population.extend(elites)

    if best_chromosome is not None:
        plot_tsp_3d(points, best_chromosome)
    else:
        print("No valid solution found.")

    min_fitness = np.min(all_fitnesses)
    max_fitness = np.max(all_fitnesses)
    mean_fitness = np.mean(all_fitnesses)
    std_fitness = np.std(all_fitnesses)

    print("Menor valor de aptidão:", min_fitness)
    print("Maior valor de aptidão:", max_fitness)
    print("Valor médio de aptidão:", mean_fitness)
    print("Desvio padrão de valor de aptidão:", std_fitness)

    return best_solution

file_path = 'C:\\Users\\av80076\\Documents\\IA\\CaixeiroSimples.csv'
points = read_points_from_file(file_path)

best_fitness = execute_genetic_algorithm(points)
print("Best fitness found:", best_fitness)
