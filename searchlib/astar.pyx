from functools import total_ordering
from typing import Callable, Iterable, Any, List, Tuple
import cython

cdef reconstruct_path(came_from: dict, current_state: object, initial_state: object):
    total_path: List[Tuple] = [(current_state, None)]
    while current_state in came_from:
        x = came_from[current_state]
        total_path.append(x)
        current_state = x[0]
    assert current_state == initial_state
    return list(reversed(total_path))

cdef class Node:
    cdef public object state
    cdef public object cost
    cdef public bint active

    def __cinit__(self, state: object, cost: object, active: bool=True):
        self.state = state
        self.cost = cost
        self.active = True

    def __eq__(self, other) -> bool:
        return self.cost == other.cost

    def __gt__(self, other) -> bool:
        return self.cost > other.cost

    def __hash__(self):
        return hash(self.state)

cdef class MyMinHeap:
    cdef dict d
    cdef list l

    def __init__(self):
        self.l = []
        self.d = {}

    cdef cython.void push(self, state: object, cost: object):
        if state in self.d:
            self.d[state].active=False
        node = Node(state, cost)
        self.d[state] = node
        self.l.append(node)
        self.l = sorted(self.l)

    cdef object pop(self):
        while len(self.l):
            node = self.l.pop(0)
            if node.active == True:
                return node.state
        return None # empty

StateType = object
CostType = object
ActionType = object

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
    - initial_state: the initial state
    - initial_cost: total cost of initial_state
    - is_goal: Function is_goal(state) -> True if state suffices goal conditions
    - get_actions: Function get_actions(state) -> Iterable of all actions that can be taken at the given state to get to a next state
    - get_state: Function get_state(state, action) -> Returns the state that will be reached when taking the given action at the given state
    - get_cost: Function get_cost(state, action) -> The cost of taking action at state
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
        def f(g_cost, state):
            return g_cost + get_heuristic(state)
    else:
        def f(g_cost, state):
            return g_cost

    best, best_cost = _astar(initial_state, initial_cost, is_goal, get_actions, get_state, get_cost, f, graph_search, include_states)
    if include_total_cost:
        return (best, best_cost)
    else:
        return best

cdef _astar(initial_state,
          initial_cost,
          is_goal,
          get_actions,
          get_state,
          get_cost,
          f,
          graph_search: bool,
          include_states: bool,
          ):
    open_set: MyMinHeap = MyMinHeap()
    open_set.push(initial_state, f(initial_cost, initial_state))
    came_from: dict = {}
    g_scores: dict = {initial_state: initial_cost}

    while True:
        current_state = open_set.pop()
        if current_state == None:
            # No path found
            return (None, None)

        g_current: object = g_scores[current_state]
        if is_goal(current_state):
            path = reconstruct_path(came_from, current_state, initial_state)
            if include_states == False:
                _states, actions = zip(*path)
                path: list = actions[:-1]
            return (path, g_current)
        for action in get_actions(current_state):
            g_cost: object = get_cost(current_state, action)
            next_state: object = get_state(current_state, action)
            g_score: object = g_current + g_cost
            if (next_state not in g_scores) or (g_score < g_scores[next_state]):
                came_from[next_state] = (current_state, action)
                g_scores[next_state] = g_score
                f_score = f(g_score, next_state)
                open_set.push(next_state, f_score)
