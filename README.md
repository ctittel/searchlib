# searchlib

This is an attempt at a library of search algorithm with a straightforward API and no unnecessary restrictions (e.g. in the types of state and cost).
The user should not be required to subclass any internal type of object. However, object types may be required to have certain properties (e.g. states are in some algorithms required to be hashable; cost objects can be of any type but are required to support comparison operators).
Callback function parameters are the way the algorithms get the things they need (e.g. neighbor solutions, cost, etc.).

Current search algorithms:
- A* (`from searchlib import astar`)
- Simulated Annealing (`from searchlib import simulated_annealing`)
- Tabu Search (`from searchlib import tabu`)

## Contribution

- Contributions welcome!
- One file per search algorithm
- Callback functions > forcing the user to subclass some internal data structures

## Installation
- Building: `python3 setup.py build_ext --inplace`
- `pip3 install git+git://github.com/ctittel/searchlib` (or `pip3 install .` if you have the repository locally and are `cd`'d into it)

## TODO
- Add more algorithms, e.g. GAs
- Add test cases and examples for all algorithms
- Optimization as much as possible (currently with Cython, can it be improved?) 

## References
- "Essentials of Metaheuristics" by Sean Luke (Link: http://cs.gmu.edu/%7Esean/book/metaheuristics/)
