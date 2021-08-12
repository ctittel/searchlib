import cython
from typing import Any, Callable, Iterable

# Tabu Search https://en.wikipedia.org/wiki/Tabu_search

SolutionType = Any
FitnessType = Any
def tabu(
    initial_solution: SolutionType,
    get_neighbors_fn: Callable[[SolutionType], Iterable[SolutionType]],
    get_fitness_fn: Callable[[SolutionType], FitnessType],
    stopping_fn: Callable[[SolutionType, FitnessType, int], bool],
    max_tabu_list_len: int = 100,
    include_best_fitness: bool = False
):
    """
    SolutionType can be anything that can be anything that can be handeled by the corresponding callback functions.
    FitnessType (the type returned by the fitness function) must support comparison with >. Greater fitness values are better

    Inputs:
    - initial_solution: State to begin search from
    - get_neighbors_fn: Function get_neighbors_fn(solution) -> Iterable(SolutionType): Returns other solutions in the neighborhood of the given solution 
    - get_fitness_fn: Function get_fitness_fn(solution) -> cost: Fitness of the given solution (a CostType)
    - stopping_fn: Function stopping_fn(best_solution, best_solution_fitness) -> bool: Called every iteration; if it returns true, the search will be stopped. Users can implement this as they want. The condition may for example use a time or iteration limit, threshold on fitness improvement etc.
    - max_tabu_list_len: How long the tabu list may get before entries are removed
    - include_total_cost: If true, returns a tuple (best solution, best solution's fitness); otherwise, only the best solution is returned
    """
    return _tabu(initial_solution, 
                get_neighbors_fn, 
                get_fitness_fn, 
                stopping_fn, 
                max_tabu_list_len, 
                include_best_fitness)

cdef object _tabu(initial_solution,
                    get_neighbors_fn,
                    get_fitness_fn,
                    stopping_fn,
                    max_tabu_list_len: int,
                    include_best_fitness: bool
):
    best = initial_solution
    best_fitness = get_fitness_fn(best)
    best_candidate = initial_solution
    
    tabu_list = [initial_solution]
    i: int = 0
    while not stopping_fn(best=best, fitness=best_fitness, i=i):
        neighbors = list(get_neighbors_fn(best_candidate))
        if len(neighbors) == 0:
            raise Exception(f"get_neighbors_fn({best_candidate}) returned no objects! Need neighbors")
        neighbors = [x for x in neighbors if x not in tabu_list]
        if len(neighbors) == 0:
            return best

        assert best_candidate != neighbors[0]
        best_candidate = neighbors.pop(0)
        best_candidate_fitness = get_fitness_fn(best_candidate)

        for candidate in neighbors:
            assert candidate != best_candidate
            candidate_fitness = get_fitness_fn(candidate)
            if candidate_fitness > best_candidate_fitness:
                best_candidate = candidate
                best_candidate_fitness = candidate_fitness

        if best_candidate_fitness > best_fitness:
            best = best_candidate
            best_fitness = best_candidate_fitness

        tabu_list.append(best_candidate)
        if len(tabu_list) > max_tabu_list_len:
            tabu_list.pop(0)
        i += 1

    if include_best_fitness:
        return (best, best_fitness)
    return best
