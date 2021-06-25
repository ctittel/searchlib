import py_search
from py_search.informed import best_first_search
import typing as typ

# For now just a wrapper of py_search best_first

class _State:
    def __init__(self, state, accumulated_cost):
        self.wrapped_state = state
        self.accumulated_cost = accumulated_cost

    def __hash__(self):
        return hash((self.wrapped_state, self.accumulated_cost))


class _Problem(py_search.base.Problem):
    def __init__(self, initial_state, initial_cost, goal_test_fun, successors_fun, calc_cost_fun):
        super().__init__(initial=initial_state, initial_cost=initial_cost)
        self.goal_test = goal_test_fun
        self.successors = successors_fun
        self.calc_cost = calc_cost_fun


def astar(initial_state: "StateType",
          initial_cost: "CostType",
          is_goal: typ.Callable[["StateType"], bool],
          get_actions: typ.Callable[["StateType"], typ.Iterable["ActionType"]],
          get_state: typ.Callable[["StateType", "ActionType"], "StateType"],
          get_cost: typ.Callable[["StateType", "ActionType"], "CostType"],
          get_heuristic: typ.Callable[["StateType"], "CostType"] = None
          ):
    def goal_test_fun(node: py_search.base.Node):
        state = node.state
        assert isinstance(state, _State)
        return is_goal(state.state)

    def successors_fun(node: py_search.base.Node):
        state = node.state
        assert isinstance(state, _State)
        for action in get_actions(state.wrapped_state):
            next_wrapped_state = get_state(state.wrapped_state, action)
            next_cost = state.accumulated_cost + get_cost(state.wrapped_state, action)
            next_state = _State(next_wrapped_state, next_cost)
            yield py_search.base.Node(
                state=next_state,
                action=action,
                node_cost=next_cost,
                parent=node
            )

    calc_cost_fun = lambda state: state.accumulated_cost
    if get_heuristic:
        calc_cost_fun = lambda state: state.accumulated_cost + get_heuristic(state.wrapped_state)

    problem = _Problem(_State(initial_state, initial_cost), initial_cost, goal_test_fun, successors_fun, calc_cost_fun)
    
    solution_node = None
    for result in best_first_search(problem=problem, graph=False):
        solution_node = result
        break
    
    nodes = []
    if solution_node:
        node = solution_node.state_node
        while node:
            nodes.append(node)
            node = node.parent
        nodes = [(node.state, node.action) for node in reversed(nodes)]
        return nodes
    else:
        return None
