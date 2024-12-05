# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import List, Dict

from ..search import Search
from ..optimizers import (
    ParallelTemperingOptimizer as _ParallelTemperingOptimizer,
)


class ParallelTemperingOptimizer(_ParallelTemperingOptimizer, Search):
    """
    A class implementing **parallel tempering** for the public API.
    Inheriting from the `Search`-class to get the `search`-method and from
    the `ParallelTemperingOptimizer`-backend to get the underlying algorithm.

    Parameters
    ----------
    search_space : dict[str, list]
        The search space to explore. A dictionary with parameter
        names as keys and a numpy array as values.
    initialize : dict[str, int]
        The method to generate initial positions. A dictionary with
        the following key literals and the corresponding value type:
        {"grid": int, "vertices": int, "random": int, "warm_start": list[dict]}
    constraints : list[callable]
        A list of constraints, where each constraint is a callable.
        The callable returns `True` or `False` dependend on the input parameters.
    random_state : None, int
        If None, create a new random state. If int, create a new random state
        seeded with the value.
    rand_rest_p : float
        The probability of a random iteration during the the search process.
    epsilon : float
        The step-size for the climbing.
    distribution : str
        The type of distribution to sample from.
    n_neighbours : int
        The number of neighbours to sample and evaluate before moving to the best
        of those neighbours.
    """

    def __init__(
        self,
        search_space: Dict[str, list],
        initialize: Dict[str, int] = {"grid": 4, "random": 2, "vertices": 4},
        constraints: List[Dict[str, callable]] = [],
        random_state: int = None,
        rand_rest_p: float = 0,
        nth_process: int = None,
        population: int = 5,
        n_iter_swap: int = 5,
    ):
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
            population=population,
            n_iter_swap=n_iter_swap,
        )