"""Test that optimum='minimum' actually minimizes the objective function.

This test exposes a bug where the adapter receives the raw objective
instead of the negated one, causing GFO to maximize even when
optimum='minimum' is set.
"""

import numpy as np

from gradient_free_optimizers import HillClimbingOptimizer

search_space = {"x0": np.arange(-5, 6, 1)}


def sphere(params):
    return params["x0"] ** 2


def test_optimum_minimum_finds_low_score():
    """With optimum='minimum', best_score should be near 0, not 25."""
    opt = HillClimbingOptimizer(search_space, random_state=0)
    opt.search(sphere, n_iter=30, optimum="minimum", verbosity=False)

    # Sphere minimum is 0 at x0=0. With 30 iterations on 11 discrete
    # points, any working minimizer should find a score below 5.
    # Bug: GFO maximizes instead, finding best_score=25 (the boundary).
    assert opt.best_score < 5, (
        f"optimum='minimum' should minimize, but best_score={opt.best_score} "
        f"with best_para={opt.best_para} (expected score near 0 at x0=0)"
    )
