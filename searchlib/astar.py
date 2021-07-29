import py_search
from py_search.informed import best_first_search
from typing import Callable, Any, Iterable
# For now just a wrapper of py_search best_first


class _Problem(py_search.base.Problem):
    def __init__(self, initial_state, initial_cost, goal_test_fun, successors_fun, calc_heuristic_fun=None):
        super().__init__(initial=initial_state, initial_cost=initial_cost)
        self.goal_test_fun = goal_test_fun
        self.successors_fun = successors_fun
        if calc_heuristic_fun:
            self.node_value = lambda state_node: py_search.base.Problem.node_value(self, state_node) + calc_heuristic_fun(state_node.state)

    def goal_test(self, state_node, goal_node):
        return self.goal_test_fun(state_node)

    def successors(self, state_node):
        return self.successors_fun(state_node)

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
    def goal_test_fun(state_node: py_search.base.Node):
        return is_goal(state_node.state)

    def successors_fun(node: py_search.base.Node):
        for action in get_actions(node.state):
            next_cost = node.cost() + get_cost(node.state, action)
            next_state = get_state(node.state, action)
            yield py_search.base.Node(
                state=next_state,
                action=action,
                node_cost=next_cost,
                parent=node
            )

    problem = _Problem(initial_state, initial_cost, goal_test_fun, successors_fun, get_heuristic)

    solution_node = None
    for result in best_first_search(problem=problem, graph=graph_search, backward=False, forward=True):
        solution_node = result
        break

    if solution_node != None:
        states = []
        actions = []
        node = solution_node.state_node
        while node:
            states.append(node.state)
            actions.append(node.action)
            node = node.parent

        actions = actions[:-1] # remove last one (= first one) for some reason
        actions = reversed(actions)
        states = reversed(states)

        result = actions
        if include_states:
            actions += [None]
            result = list(zip(states, actions))

        if include_total_cost:
            return (result, solution_node.cost())
        else:
            return result
    else:
        return None
