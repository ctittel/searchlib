# searchlib

## Principles:
- *pythonic* API
- search algorithms take objects & function parameters
- free to use any kind of types for all objects (for the states, actions and even the costs); never force the user to use a certain type for something or to create a child class of something
- in some algorithms some of these types are required to have certain properties, e.g. being hashable. Costs must always be comparable (with `==`, `>`, etc. - `total-ordering`)


## Installation
`pip3 install git+git://github.com/ctittel/searchlib`

## References
- "Essentials of Metaheuristics" by Sean Luke (Link: http://cs.gmu.edu/%7Esean/book/metaheuristics/)
