from searchlib import astar
import random

states = ["a", "b1", "b2", "c1", "c2", "d1", "d2"]

actions_costs = {("a", "b1"): 1,
               ("a", "b2"): 100,
               ("b1", "c1"): 100,
               ("b1", "c2"): 3,
               ("b2", "c1"): 100,
               ("b2", "c2"): 100,
               ("c1", "d1"): 100,
               ("c1", "d2"): 100,
               ("c2", "d1"): 7,
               ("c2", "d2"): 6}

# cheapest: a -> b1 -> c2 -> d2
best_path = ["a", "b1", "c2", "d2"]

initial_state = "a"
initial_cost = 5


def is_goal(state):
    done = (state == "d1" or state == "d2")
    # print(f"is_goal({state})={done}")
    return done


def get_actions(state):
    actions = [x for x in actions_costs.keys() if state == x[0]]
    # print(f"get_actions({state})={actions}")
    return actions


def get_state(state, action):
    return action[1]


def get_cost(state, action):
    return actions_costs[action]


result = astar(initial_state=initial_state,
               initial_cost=initial_cost,
               is_goal=is_goal,
               get_actions=get_actions,
               get_state=get_state,
               get_cost=get_cost,
               get_heuristic=None)

res_states = []
for state, action in result:
    res_states.append(state)

print("Best path = ", best_path)
print("Found best path = ", res_states)
for first, second in zip(res_states, best_path):
    assert first == second, f"{first} != {second}"
