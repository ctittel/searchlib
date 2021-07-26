# Tabu Search https://en.wikipedia.org/wiki/Tabu_search
from typing import Any, Callable, Iterable

SolutionType = Any
FitnessType = Any
def tabu(
    initial_solution: SolutionType,
    get_neighbors: Callable[[SolutionType], Iterable[SolutionType]],
    get_fitness: Callable[[SolutionType], FitnessType],
    stopping_condition: Callable[[SolutionType, FitnessType], bool],
    max_tabu_list_len: int = 100
):
    """
    SolutionType can be anything that can be anything that can be handeled by the corresponding callback functions.
    FitnessType (the type returned by the fitness function) must support comparison (>, ==, etc.). Greater fitness values are better

    Inputs:
    - initial_solution: State to begin search from
    - get_neighbors: Function get_neighbors(solution) -> Iterable(SolutionType): Returns other solutions in the neighborhood of the given solution 
    - get_fitness: Function get_fitness(solution) -> cost: Fitness of the given solution (a CostType)
    - stopping_condition: Function stopping_condition(best_solution, best_solution_fitness) -> bool: Called every iteration; if it returns true, the search will be stopped. Users can implement this as they want. The condition may for example use a time or iteration limit, threshold on fitness improvement etc.
    - max_tabu_list_len: How long the tabu list may get before entries are removed
    """
    best = initial_solution
    best_candidate = initial_solution
    
    tabu_list = [initial_solution]

    while not stopping_condition(initial_solution, get_fitness(initial_solution)):
        neighbors = list(get_neighbors(best_candidate))
        if len(neighbors) == 0:
            raise Exception(f"get_neighbors({best_candidate}) returned no objects! Need neighbors")
        best_candidate = neighbors[0]
        for candidate in neighbors:
            if (candidate not in tabu_list) and (get_fitness(candidate) > get_fitness(best_candidate)):
                best_candidate = candidate
        if get_fitness(best_candidate) > get_fitness(best):
            best = best_candidate
        tabu_list.append(best_candidate)
        if len(tabu_list) > max_tabu_list_len:
            tabu_list = tabu_list[1:]
    return best
