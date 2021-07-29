# Simulated Annealing https://en.wikipedia.org/wiki/Simulated_annealing

from typing import Any, Callable
from random import random
import math

SolutionType = Any
EnergyType = Any

def kirkpatrick_acceptance_probability(E: float, E_new: float, T: float):
    if E_new < E:
        return 1.0
    else:
        return math.exp((E - E_new) / T)

def default_temperature_fn(k, k_max):
    return (1 - (k+1) / k_max)

def simulated_annealing(initial_solution: SolutionType,
                        k_max: int,
                        random_neighbor_fn: Callable[[SolutionType], SolutionType],
                        energy_fn: Callable[[SolutionType], EnergyType],
                        temperature_fn: Callable[[int, int], float] = default_temperature_fn,
                        acceptance_probability_fn: Callable[[EnergyType, EnergyType, float], float] = kirkpatrick_acceptance_probability,
                        random_fn = random):
    """
    Inputs:
    - initial_s: initial solution; any type
    - k_max: number of optimization steps; positive integer
    - random_neighbor_fn: function: prev_solution -> random solution in the neighboorhood of prev_solution
    - energy_fn: function: solution -> energy 
        - the energy can be of any type; when using the default acceptance_probability function the energy must be of type float 
        - the algorithm tries to minimize the energy
    - temperature_fn: function: k_current, k_max -> float [0.0, 1.0]
        - maps current step number to a float
        - larger step numbers (large k_current) should return smaller (closer to 0) temperatures
    - acceptance_probability_fn: function (energy of current solution, energy of best solution, current temperature) -> probability [0.0, 1.0]
        - checkout description of the algorithm to understand
        - by default uses Kirkpatrick et al.'s algorithm
    - random_fn: By default, returns uniformly random floats between [0.0, 1.0)

    If the used energy type is not a number but of a custom type, a custom implementation of the acceptance_probability_fn must be used
    """
    s = initial_solution
    E = energy_fn(s)

    for k in range(k_max):
        T = temperature_fn(k, k_max)
        s_new = random_neighbor_fn(s)
        E_new = energy_fn(s_new)
        if acceptance_probability_fn(E, E_new, T) >= random_fn():
            s = s_new
            E = E_new
    return s
