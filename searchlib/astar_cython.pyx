from functools import total_ordering
from typing import Callable, Iterable, Any, List, Tuple
import cython

cdef reconstruct_path(came_from: dict, current_state: object, initial_state: object):
    total_path: List[Tuple] = [(current_state, None)]
    states = set([current_state])
    while current_state in came_from:
        x = came_from[current_state]
        current_state = x[0]
        if current_state in states:
            raise Exception(f"Encountered a cycle while reconstructing path. State {current_state} is already in path {total_path}")
        total_path.append(x)
        states.add(current_state)
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
                del self.d[node.state]
                return node.state
        return None # empty
    
    def __len__(self):
        return len(self.l)
    
    def __contains__(self, state):
        return (state in self.d)


cpdef _astar(initial_state,
            initial_cost,
            is_goal,
            get_actions,
            get_state,
            get_cost,
            f,
            include_states: bool,
            graph_search: bool
            ):
    open_set: MyMinHeap = MyMinHeap()
    open_set.push(initial_state, f(initial_cost, initial_state))
    came_from: dict = {}
    g_scores: dict = {initial_state: initial_cost}
    closed_set = set()

    while True:
        current_state = open_set.pop()
        while current_state in closed_set:
            current_state = open_set.pop()

        if current_state is None:
            return (None, None)

        g_current: object = g_scores[current_state]
        if is_goal(current_state):
            path = reconstruct_path(came_from, current_state, initial_state)
            if include_states == False:
                _states, actions = zip(*path)
                path: list = actions[:-1]
            return (path, g_scores[current_state])
        for action in get_actions(current_state):
            next_state: object = get_state(current_state, action)

            g_score = get_cost(current_state, action)
            if g_scores[current_state] is not None:
                g_score += g_scores[current_state]

            f_score = f(g_score, next_state)
            if (next_state not in open_set) and (next_state not in closed_set) and (next_state != current_state):
                assert next_state != current_state, f"current_state={current_state} next_state={next_state} open_set={open_set.d.keys()}"
                g_scores[next_state] = g_score
                came_from[next_state] = (current_state, action)
                open_set.push(next_state, f_score)
            elif (g_scores[next_state] is not None) and (g_score < g_scores[next_state]):
                came_from[next_state] = (current_state, action)
                g_scores[next_state] = g_score
                if next_state in open_set:
                    open_set.push(next_state, f_score)

        if graph_search:
            closed_set.add(current_state)
