# searchlib

## Principles:
- Try to design API in a *pythonic* way
- The simplest way is probably to implement the search algorithms as functions that take normal and function parameters
- Don't require the user to create child-classes
    - Different search algorithms may require very different parameters, so having a base class with methods the user must override will eventually create problems because methods certain search algorithms require may be missing or methods may be interpreted in different ways depending on the search algorithm used
- When possible allow the user to use any type of cost type (not necessarily floats), with the only prerequisite being that all returned costs/heuristics can be compared with operands like `==` and `>` (responsibility of the user)

## Installation
`pip3 install git+git://github.com/ctittel/searchlib`

## References
- "Essentials of Metaheuristics" by Sean Luke (Link: http://cs.gmu.edu/%7Esean/book/metaheuristics/)
