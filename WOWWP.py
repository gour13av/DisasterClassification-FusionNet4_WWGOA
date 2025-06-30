#  Walrus Optimization Algorithm + Binary Waterwheel plant optimization
import numpy as np


def Wowwp(Objective_function, lb, ub, pop_size, prob_size, epochs):
    population = np.random.uniform(lb, ub, size=(pop_size, prob_size))
    best_solution = None
    best_fitness = float('inf')
    lb = np.array(lb)
    ub = np.array(ub)
    for i in range(1, epochs+1):
        K = 0
        for j in range(pop_size):
            # bounds function for population
            population[j, population[j] < lb] = lb[population[j] < lb]
            population[j, population[j] > ub] = ub[population[j] > ub]
            fitness = Objective_function(population[j])

            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = population[j]
            r3 = np.random.uniform(0, 2)
            f = np.random.uniform(-5, 5)
            k = (1 + ((2*(i)**2)/epochs) + f)
            w = r3*(k*best_solution + r3*population[j])  # Hybridization Part from Binary Waterwheel Plant Optimization
            I = np.random.randint(1, 2, prob_size)
            # p1, p2, p3 meant the phases
            p1_population = population[j] + np.random.rand() * (best_solution - I* population[j])
            p2_population = population[j] + np.random.rand() * (population[j] - population[K])
            lb_local = lb/i
            ub_local = ub/i
            p3_population = population[j] + (lb_local + (ub_local - np.random.rand()*lb_local)) + (k*w)
            K += 1
            if Objective_function(p1_population) < fitness:
                population[j] = p1_population
            elif Objective_function(p2_population) < fitness:
                population[j] = p2_population
            elif Objective_function(p3_population) < fitness:
                population[j] = p3_population
            else:
                population[j] = population[j]
    return best_solution



