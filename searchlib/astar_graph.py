import py_search
from py_search.informed import best_first_search
import typing as typ

# For now just a wrapper of py_search best_first


class _Problem(py_search.base.Problem):
    def __init__(self, initial_state, initial_cost, goal_test_fun, successors_fun, calc_heuristic_fun=None):
        super().__init__(initial=initial_state, initial_cost=initial_cost)
        self.goal_test_fun = goal_test_fun
        self.successors_fun = successors_fun
        if calc_heuristic_fun:
            self.node_value = lambda state_node: py_search.base.Problem.node_value(state_node) + calc_heuristic_fun(state_node.state)

    def goal_test(self, state_node, goal_node):
        return self.goal_test_fun(state_node)

    def successors(self, state_node):
        return self.successors_fun(state_node)

    # def node_value(self, state_node):
    #     return self.calc_cost_fun(state_node.state)


def astar_graph(initial_state: "StateType",
          initial_cost: "CostType",
          is_goal: typ.Callable[["StateType"], bool],
          get_actions: typ.Callable[["StateType"], typ.Iterable["ActionType"]],
          get_state: typ.Callable[["StateType", "ActionType"], "StateType"],
          get_cost: typ.Callable[["StateType", "ActionType"], "CostType"],
          get_heuristic: typ.Callable[["StateType"], "CostType"] = None
          ):
    def goal_test_fun(state_node: py_search.base.Node):
        state = state_node.state
        if not state:
            return False
        return is_goal(state)

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
    for result in best_first_search(problem=problem, graph=True, backward=False, forward=True):
        solution_node = result
        break

    if solution_node:
        states = []
        actions = []
        node = solution_node.state_node
        while node:
            states.append(node.state)
            actions.append(node.action)
            node = node.parent

        actions = [None] + actions[:-1]

        return list(reversed(list(zip(states, actions))))
    else:
        return None
