# Ant-colony optimization: https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms

from typing import Callable, List, Tuple, Dict
import numpy as np

StateType = object
PathType = List[StateType]
CostType = object


def default_path_fitness_fn(path: list, attractivenesses: dict):
    return sum([attractivenesses[(x)] for x in zip(path[:-1], path[1:])])


def default_select_next_state_fn(path: list,
                                 next_states: list,
                                 phermone_levels: list,
                                 costs: list,
                                 alpha=0.5, beta=0.5):
    etas = np.array(costs) ** (-1)
    taus = np.array(phermone_levels)
    product = (taus**alpha) * (etas**beta)
    p = product / product.sum()
    return np.random.choice(next_states, p=p)

def default_phermone_delta_fn(path, cost, finished, Q=1.0):
    if finished:
        return (Q / cost)
    else:
        return -1 * (Q / cost)

def aco(initial_state: StateType,
        cost_fn: Callable[[StateType, StateType], CostType],
        stopping_fn: Callable[[StateType, float, int], bool],
        n_ants: int,
        phi: float, # phermone decay factor
        next_states_fn: Callable[[PathType], List[StateType]],
        is_complete_fn: Callable[[PathType], bool],
        select_next_state_fn: Callable[[PathType,           # path
                                        List[StateType],    # next_states
                                        List[float],        # phermone_levels
                                        List[CostType]],    # costs
                                       StateType] = default_select_next_state_fn,  # -> selected next state
        phermone_delta_fn: Callable[[PathType, CostType, bool]] = default_phermone_delta_fn,
        initial_cost: CostType = 0.0,
        include_best_cost=False):

    phermone_levels = {}
    default_phermone_level = 1.0
    costs = {}  # key: (prev_state, next_state)

    best = None
    best_cost = None
    i = 0
    while not stopping_fn(best=best, cost=best_cost, i=i):
        i += 1
        paths = []
        unfinished_paths = []

        for ant in range(n_ants):
            path = [initial_state]
            _total_cost = initial_cost

            while not is_complete_fn(path):
                current_state = path[-1]
                next_states = next_states_fn(path)
                if not len(next_states):  # Stuck
                    break
                for x in next_states:
                    if (current_state, x) not in costs:
                        costs[(current_state, x)] = cost_fn(current_state, x)

                next_states_costs = [costs[(current_state, x)]
                                     for x in next_states]
                next_states_phermone = [
                    phermone_levels[(current_state, x)] for x in next_states]
                next_state = select_next_state_fn(path,
                                                  next_states,
                                                  next_states_phermone,
                                                  next_states_costs
                                                  )
                path.append(next_state)
                _total_cost += costs[(current_state, next_state)]
            if is_complete_fn(path):
                paths.append((path, _total_cost))
            else:
                # Ant got stuck and starved :(
                unfinished_paths.append((path, _total_cost))

        #+ Calc phermone deltas
        deltas = {}
        for path, path_cost in paths:
            for pair in zip(path[:-1], path[1:]):
                deltas[pair] = deltas.get(pair, 0.0) + phermone_delta_fn(path, path_cost, True)
        for path, path_cost in unfinished_paths:
            for pair in zip(path[:-1], path[1:]):
                deltas[pair] = deltas.get(pair, 0.0) + phermone_delta_fn(path, path_cost, False)

        #+ Apply decay
        for x, phermone in phermone_levels.items():
            phermone_levels[x] = (1-phi)*phermone
        default_phermone_level *= (1-phi)

        #+ Apply delta
        for pair, delta in deltas.items():
            phermone_levels[pair] = phermone_levels.get(pair, default_phermone_level) + delta

        current_best, current_best_cost = min(paths, key=lambda x: x[1])
        if best is None or current_best_cost < best_cost:
            best = current_best
            best_cost = current_best_cost

    if include_best_cost:
        return (best, best_cost)
    else:
        return best
