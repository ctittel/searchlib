import numpy as np
import random
import itertools

class TSPProblem:
    """
    Simple interface for test problems
    """
    initial_cost = None

    def __init__(self, max_steps, dims, num_nodes):
        self.max_steps = max_steps
        self.nodes = list(range(num_nodes))
        coords = [np.random.rand(dims) for _ in range(num_nodes)]
        self.edges = {}
        for a in self.nodes:
            for b in self.nodes:
                if a != b:
                    self.edges[(a,b)] = np.linalg.norm(coords[a] - coords[b])
        self.intial_state = tuple()
        self.initial_solution = tuple(self.nodes)
        self.nodes_set = set(self.nodes)

    def stopping_fn(self, i, **kwargs):
        return i >= self.max_steps

    # Going from one city to another
    def cost_fn(self, state: list, action: int):
        if len(state) > 0:
            return self.edges[(state[-1], action)]
        else:
            return 0.0

    def next_state_fn(self, state: tuple, action: int):
        assert action not in state
        return tuple(list(state) + [action])

    def get_actions(self, state):
        a = set(state)
        return [x for x in self.nodes if x not in a] # possible next states

    def fitness_fn(self, solution: list):
        return self.energy_fn(solution) * -1

    def energy_fn(self, solution: list):
        costs = [self.edges[(n1,n2)] for n1,n2 in zip(solution[:-1], solution[1:])]
        return sum(costs)

    def is_goal_fn(self, state: list):
        return self.nodes_set == set(state)

    lower_bound_heuristic = None

    def breeding_fn(self, parent_solutions):
        p1, p2 = parent_solutions
        new = [x if random.choice([True, False]) else None for x in p1]
        missing = [x for x in p2 if x not in set(new)]
        new = [x if x is not None else missing.pop(0) for x in new]
        assert len(missing) == 0
        return new

    def random_neighbor_fn(self, solution):
        a = list(solution)
        random.shuffle(a)
        return a

    def neighbors_fn(self, solution: list):
        # return all permuations where two nodes are switched
        solution = list(solution)
        if len(solution) == 2:
            yield tuple(solution)
            yield tuple(solution[::-1])
        else:
            for i in range(1, len(solution)):
                yield tuple([solution[i]] + solution[1:i] + [solution[0]] + solution[i+1:])
            for x in self.neighbors_fn(solution[1:]):
                yield tuple([solution[0]] + list(x))

if __name__ == "__main__":
    from searchlib import astar, genetic_algorithm, simulated_annealing, tabu
    problem = TSPProblem(10, 2, 5)

    result, cost = astar(problem.intial_state,0.0,problem.is_goal_fn, problem.get_actions, problem.next_state_fn, problem.cost_fn, include_total_cost=True)
    states, actions = zip(*result)
    print("Astar result cost = ",cost, " solution = ", states[-1])

    result, fitness = tabu(problem.initial_solution, problem.neighbors_fn, problem.fitness_fn, problem.stopping_fn, include_best_fitness=True)
    print("Tabu result cost = ",fitness*-1, " solution = ", result)

    population = [random.sample(problem.initial_solution, len(problem.initial_solution)) for _ in range(10)]
    result, fitness = genetic_algorithm(population, problem.fitness_fn, problem.breeding_fn, 2, problem.stopping_fn, include_best_fitness=True)
    print("GA result cost = ",fitness*-1, " solution = ", result)

    result, cost = simulated_annealing(problem.initial_solution, 10, problem.random_neighbor_fn, problem.energy_fn, include_best_energy=True)
    print("Simulated Annealing result cost = ",cost, " solution = ", result)

