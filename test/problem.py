class Problem:
    """
    Simple interface for test problems
    """
    intial_state = None
    initial_cost = None

    def __init__(self, max_steps):
        self.max_steps = max_steps

    def cost_fn(self, state, action):
        raise NotImplementedError()

    def fitness_fn(self, state):
        raise NotImplementedError()

    def energy_fn(self, state):
        return self.fitness_fn(state) * -1

    def lower_bound_heuristic(self, state):
        raise NotImplementedError()

    def stopping_fn(self, i, **kwargs):
        return i >= self.max_steps

    def breeding_fn(self, parents):
        raise NotImplementedError()
