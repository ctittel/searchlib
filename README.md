# searchlib

Currently offers:
- A* (`from searchlib import astar`)
- Simulated Annealing (`from searchlib import simulated_annealing`)
- Tabu Search (`from searchlib import tabu`)

## Principles:
- *pythonic* API
- search algorithms take objects & function parameters
- don't force the user to create child classes of anything, because it makes things more complicated from the user's perspective
- free to use any kind of types wherever possible (for the states, actions and even the costs); never force the user to use a certain type for something if the algorithm doesn't strictly require it
- in some algorithms some of these types are required to have certain properties, e.g. being hashable. Costs must always be comparable (with `==`, `>`, etc. - `total-ordering`)


## Installation
- Building: `python3 setup.py build_ext --inplace`
- `pip3 install git+git://github.com/ctittel/searchlib` (or `pip3 install .` if you downloaded the repository)

## TODO
- Add test cases and examples for all algorithms
- Figure out an easy way to make algorithms hardware accelerated / compiled 

## References
- "Essentials of Metaheuristics" by Sean Luke (Link: http://cs.gmu.edu/%7Esean/book/metaheuristics/)
