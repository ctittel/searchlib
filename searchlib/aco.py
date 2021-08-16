# Ant-colony optimization: https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms

from typing import Callable, List, Tuple
import numpy as np

StateType = object
PathType = List[StateType]

def aco(initial_state: StateType,
        attractiveness_fn: Callable[[StateType, StateType], float],
        stopping_fn: Callable[[StateType, float, int], bool],
        alpha: float,
        beta: float,
        n_ants: int,
        phi: float,
        next_states_fn: Callable[[PathType], List[StateType]],
        is_complete_fn: Callable[[PathType], bool],
        include_best_attractiveness=False):

    phermone_levels = {}
    default_phermone_level = 1.0
    attractivenesses = {} # key: (prev_state, next_state) val: attractiveness_fn(prev_state, next_state)
    Q = 1.0

    best = None
    best_att = None
    i = 0
    while not stopping_fn(best=best, attractivenesses=best_att, i=i):
        i += 1
        paths = []
        paths_atts = []

        for ant in range(n_ants):
            path = [initial_state]
            total_att = 0.0

            while not is_complete_fn(path):
                current_state = path[-1]
                next_states = next_states_fn(path)
                for x in next_states: 
                    xx = (current_state, x)
                    if xx not in attractivenesses:
                        attractivenesses[xx] = attractiveness_fn(*xx)
                    
                etas = np.array([attractivenesses[(current_state, x)] for x in next_states])
                taus = np.array([phermone_levels.get((current_state, x),default_phermone_level) for x in next_states])
                
                product = (taus**alpha) * (etas**beta)
                p = product / product.sum()
                next_state = np.random.choice(next_states, p=p)
                path.append(next_state)
                total_att += attractivenesses[(current_state, next_state)]
            paths.append(path)
            paths_atts.append(total_att)

        best_i = np.argmax(paths_atts)
        if best is None or paths_atts[best_i] > best_att:
            best = paths[best_i]
            best_att = paths_atts[best_i]

        deltas = {}
        for path, att in zip(paths, paths_atts):
            for pair in zip(path[:-1], path[1:]):
                deltas[pair] = deltas.get(pair, 0.0) + Q * att
        for pair, delta in deltas.items():
            phermone_levels[pair] = (1-phi)*phermone_levels.get(pair, default_phermone_level) + delta
    
    if include_best_attractiveness:
        return (best, best_att)
    else:
        return best