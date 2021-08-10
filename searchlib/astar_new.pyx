import heapq
from functools import total_ordering
from typing import Callable, Iterable, Any


def reconstruct_path(came_from: dict, current_state, initial_state):
    total_path = [(current_state, None)]
    while current_state in came_from:
        x = came_from[current_state]
        total_path.append(x)
        current_state = x[0]
    assert current_state == initial_state
    return list(reversed(total_path))


@total_ordering
class Node:
    def __init__(self, state, cost, active=True):
        self.state = state
        self.cost = cost
        self.active = True

    def __eq__(self, other):
        return self.cost == other.cost

    def __gt__(self, other):
        return self.cost > other.cost

    def __hash__(self):
        return hash(self.state)

class MyMinHeap:
    def __init__(self):
        self.d = dict()
        self.l = list()
        heapq.heapify(self.l)

    def push(self, state, cost):
        if state in self.d:
            self.d[state].active=False
        node = Node(state, cost)
        self.d[state] = node
        heapq.heappush(self.l, node)

    def pop(self):
        while True:
            node = heapq.heappop(self.l)
            if node.active == True:
                return node.state

    def __iter__(self):
        while len(self.l) > 0:
            node = heapq.heappop(self.l)
            if node.active == True:
                yield node.state

StateType = Any
CostType = Any
ActionType = Any
def astar(initial_state: StateType,
          initial_cost: CostType,
          is_goal: Callable[[StateType], bool],
          get_actions: Callable[[StateType], Iterable[ActionType]],
          get_state: Callable[[StateType, ActionType], StateType],
          get_cost: Callable[[StateType, ActionType], CostType],
          get_heuristic: Callable[[StateType], CostType] = None,
          graph_search: bool = False,
          include_states: bool = True,
          include_total_cost: bool = False,
          ):
    """
    Actions, states and costs can be of any type
    States must always be of a hashable type
    The objects returned by the cost functions must be comparable (>, ==, etc.) and support addition with each other

    Parameters:
    - initial_state (StateType): the initial state
    - initial_cost (CostType): total cost of initial_state
    - is_goal: Function is_goal(state) -> bool returns True if state suffices goal conditions
    - get_actions: Function get_actions(state) -> Iterable(Actions): Given a state, return all actions that can be used to get a next state
    - get_state: Function get_state(state, action) -> new_state: Given a state and an action that is taken (the action will always be one that was returned by get_actions) to receive the next state
    - get_cost: Function get_cost(state, action) -> cost: The cost of taking action at state
    - get_heuristic: Function get_heuristic(state) -> cost: Minimum (lower bound) cost expected to reach a goal from state
    - include_states: 
        - If False, the function returns the list of actions which must be taken from initial_state to reach goal
        - If True, the function returns a list of (state, action) pairs where the last action is None 
    - include_total_cost: if True a tuple will be returned of (result, total_cost)

    Returns:
    - result: List of (state, action) pairs with the action taken at the state; last action is None
    - if include_total_cost is True: returns tuple (list, total_cost)
    """
    if get_heuristic != None:
        def f(g_cost, state: StateType):
            return g_cost + get_heuristic(state)
    else:
        def f(g_cost: CostType, state: StateType):
            return g_cost

    open_set = MyMinHeap()
    open_set.push(initial_state, f(initial_cost, initial_state))
    came_from = {}
    g_scores = {initial_state: initial_cost}

    while True:
        current_state = open_set.pop()

        g_current = g_scores[current_state]
        if is_goal(current_state):
            path = reconstruct_path(came_from, current_state, initial_state)
            if include_states == False:
                _states, actions = zip(*path)
                path = actions[:-1]
            if include_total_cost:
                return (path, g_current)
            else:
                return path
        for action in get_actions(current_state):
            g_cost = get_cost(current_state, action)
            next_state = get_state(current_state, action)
            g_score = g_current + g_cost
            if (next_state not in g_scores) or (g_score < g_scores[next_state]):
                came_from[next_state] = (current_state, action)
                g_scores[next_state] = g_score
                f_score = f(g_score, next_state)
                open_set.push(next_state, f_score)
