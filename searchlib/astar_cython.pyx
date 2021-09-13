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


cpdef _astar(initial_state,
          initial_cost,
          is_goal,
          get_actions,
          get_state,
          get_cost,
          f,
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
