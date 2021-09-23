from .aco import aco
from .astar_cython import _astar
from typing import Callable, Iterable

StateType = object
CostType = object
ActionType = object


def astar(initial_state: StateType,
          is_goal: Callable[[StateType], bool],
          get_actions: Callable[[StateType], Iterable[ActionType]],
          get_state: Callable[[StateType, ActionType], StateType],
          get_cost: Callable[[StateType, ActionType], CostType],
          get_heuristic: Callable[[StateType], CostType] = None,
          initial_cost: CostType = None,
          include_states: bool = True,
          include_total_cost: bool = False,
          graph_search=False
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
    if get_heuristic is not None:
        def f(g_cost, state):
            h = get_heuristic(state)
            if g_cost is not None:
                return g_cost + h
            else:
                return h
    else:
        def f(g_cost, state):
            return g_cost

    best, best_cost = _astar(initial_state,
                             initial_cost,
                             is_goal,
                             get_actions,
                             get_state,
                             get_cost,
                             f,
                             include_states,
                             graph_search)
    if include_total_cost:
        return (best, best_cost)
    else:
        return best
