import cython
from typing import Any, Callable, Iterable

# Tabu Search https://en.wikipedia.org/wiki/Tabu_search

SolutionType = Any
FitnessType = Any
def tabu2(
    initial_solution: SolutionType,
    get_neighbors: Callable[[SolutionType], Iterable[SolutionType]],
    get_fitness: Callable[[SolutionType], FitnessType],
    stopping_condition: Callable[[SolutionType, FitnessType], bool],
    max_tabu_list_len: int = 100,
    include_best_fitness: bool = False
):
    """
    SolutionType can be anything that can be anything that can be handeled by the corresponding callback functions.
    FitnessType (the type returned by the fitness function) must support comparison with >. Greater fitness values are better

    Inputs:
    - initial_solution: State to begin search from
    - get_neighbors: Function get_neighbors(solution) -> Iterable(SolutionType): Returns other solutions in the neighborhood of the given solution 
    - get_fitness: Function get_fitness(solution) -> cost: Fitness of the given solution (a CostType)
    - stopping_condition: Function stopping_condition(best_solution, best_solution_fitness) -> bool: Called every iteration; if it returns true, the search will be stopped. Users can implement this as they want. The condition may for example use a time or iteration limit, threshold on fitness improvement etc.
    - max_tabu_list_len: How long the tabu list may get before entries are removed
    - include_total_cost: If true, returns a tuple (best solution, best solution's fitness); otherwise, only the best solution is returned
    """
    best = initial_solution
    best_fitness = get_fitness(best)
    best_candidate = initial_solution
    
    tabu_list = [initial_solution]

    # TODO: Have made some adaptions; check if they are good
    while not stopping_condition(best=best, fitness=best_fitness):
        neighbors = get_neighbors(best_candidate)
        if len(neighbors) == 0:
            raise Exception(f"get_neighbors({best_candidate}) returned no objects! Need neighbors")
        neighbors = [x for x in neighbors if x not in tabu_list]
        if len(neighbors) == 0:
            return best

        assert best_candidate != neighbors[0]
        best_candidate = neighbors.pop(0)
        best_candidate_fitness = get_fitness(best_candidate)

        for candidate in neighbors:
            assert candidate != best_candidate
            candidate_fitness = get_fitness(candidate)
            if candidate_fitness > best_candidate_fitness:
                best_candidate = candidate
                best_candidate_fitness = candidate_fitness

        if best_candidate_fitness > best_fitness:
            best = best_candidate
            best_fitness = best_candidate_fitness

        tabu_list.append(best_candidate)
        if len(tabu_list) > max_tabu_list_len:
            tabu_list.pop(0)

    if include_best_fitness:
        return (best, best_fitness)
    return best
