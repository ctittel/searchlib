# https://en.wikipedia.org/wiki/Genetic_algorithm
from typing import List, Callable, Iterable, Union, Tuple
import random

SolutionType = object
FitnessType = object


cdef list elitist_ranking(solutions: List[SolutionType], fitnesses: List[FitnessType]):
    s = sorted(zip(solutions, fitnesses), key=lambda x: x[1])
    solutions, _ = zip(*s)
    return list(reversed(solutions))

cdef list fitness_proportionate_ranking(solutions: List[SolutionType], fitnesses: List[FitnessType]):
    s = sorted(zip(solutions, fitnesses), key=lambda x: x[1])
    solutions_sorted, _ = zip(*s) # worst first, best last
    solutions_sorted = list(solutions_sorted)
    float_fitnesses = [(x+1)/len(solutions) for x in range(len(solutions))]
    float_fitnesses = [x + random.random() for x in float_fitnesses] # add random factor
    return elitist_ranking(solutions_sorted, float_fitnesses)

def genetic_algorithm(initial_population: List[SolutionType],
                        fitness_fn: Callable[[SolutionType], FitnessType],
                        breeding_fn: Callable[[List[SolutionType]], SolutionType],
                        stopping_fn: Callable[[SolutionType, FitnessType, int], bool],
                        ranking_fn: Callable[[List[SolutionType], List[FitnessType]], Iterable[SolutionType]] = fitness_proportionate_ranking,
                        number_of_parents: int = 2,
                        parent_population_size: int = None,
                        include_best_fitness=False
                        ) -> Union[SolutionType, Tuple[SolutionType, FitnessType]]:
    """
    Inputs:
    - initial_population: a list with the initial solutions
    - fitness_fn(solution) -> fitness: Function that returns the fitness of a solution (higher fitness is better)
    - breeding_fn(list of parent solutions) -> new child solution: Calculates a "child" solution from a list of parents
        - number_of_parents parameter determines how many parent solutions are in the list
    - stopping_fn(current best solution, corresponding fitness, iteration number) -> True if optimization should be stopped now
    - ranking_fn(list of current solution population, list of the corresponding fitnesses) -> the same solutions ranked (the best or selected solutions come first). 
        - parent_population_size solutions are chosen from the list
    - number_of_parents: see breeding_fn
    - parent_population_size: See ranking_fn; if None (default): the rounded half length of the intial_population list
    - include_best_fitness: If True not the best solution is returned but a tuple (best solution, corresponding fitness)

    Returns:
    - Best solution or (if include_best_fitness is True) a tuple (best solution, corresponding fitness)
    """
    population_size = len(initial_population)

    if parent_population_size == None:
        parent_population_size = population_size // 2
        assert parent_population_size > number_of_parents

    best, best_fitness = _ga(initial_population, fitness_fn, breeding_fn, stopping_fn, ranking_fn, population_size, number_of_parents, parent_population_size)
    if include_best_fitness:
        return (best, best_fitness)
    else:
        return best

cdef _ga(population: list,
            fitness_fn,
            breeding_fn,
            stopping_fn,
            ranking_fn,
            population_size:int,
            number_of_parents: int,
            parent_population_size: int):
    best = population[0]
    best_fitness = fitness_fn(best)
    i = 0
    while not stopping_fn(best=best, fitness=best_fitness, i=i):
        fitnesses = [fitness_fn(x) for x in population]

        current_best, current_best_fitness = max(zip(population, fitnesses), key=lambda x: x[1]) 
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best = current_best

        ranked_pop = ranking_fn(population, fitnesses)
        # select top parents
        parents = [x for _, x in zip(range(parent_population_size), ranked_pop)]
        # create pairs of parents
        parent_pairs = [random.sample(parents, number_of_parents) for _ in range(population_size)]
        # create next population
        population = [breeding_fn(x) for x in parent_pairs]
        assert len(population) == population_size
        i += 1

    return (best, best_fitness)