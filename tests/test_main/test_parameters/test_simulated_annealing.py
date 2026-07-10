"""Tests for SimulatedAnnealingOptimizer temperature and annealing parameters."""
# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import math

import numpy as np
import pytest

from gradient_free_optimizers import SimulatedAnnealingOptimizer


def objective_function(para):
    score = -para["x1"] * para["x1"]
    return score


search_space = {
    "x1": np.arange(0, 10, 1),
}


n_iter = 1000


def test_start_temp_0():
    n_initialize = 1

    start_temp_0 = 0
    start_temp_1 = 0.1
    start_temp_10 = 1
    start_temp_100 = 100
    start_temp_inf = np.inf

    epsilon = 1 / np.inf

    opt = SimulatedAnnealingOptimizer(
        search_space,
        start_temp=start_temp_0,
        epsilon=epsilon,
        initialize={"random": n_initialize},
    )
    opt.search(objective_function, n_iter=n_iter)
    n_transitions_0 = opt.n_transitions

    opt = SimulatedAnnealingOptimizer(
        search_space,
        start_temp=start_temp_1,
        epsilon=epsilon,
        initialize={"random": n_initialize},
    )
    opt.search(objective_function, n_iter=n_iter)
    n_transitions_1 = opt.n_transitions

    opt = SimulatedAnnealingOptimizer(
        search_space,
        start_temp=start_temp_10,
        epsilon=epsilon,
        initialize={"random": n_initialize},
    )
    opt.search(objective_function, n_iter=n_iter)
    n_transitions_10 = opt.n_transitions

    opt = SimulatedAnnealingOptimizer(
        search_space,
        start_temp=start_temp_100,
        epsilon=epsilon,
        initialize={"random": n_initialize},
    )
    opt.search(objective_function, n_iter=n_iter)
    n_transitions_100 = opt.n_transitions

    opt = SimulatedAnnealingOptimizer(
        search_space,
        start_temp=start_temp_inf,
        epsilon=epsilon,
        initialize={"random": n_initialize},
    )
    opt.search(objective_function, n_iter=n_iter)
    n_transitions_inf = opt.n_transitions

    print("\n n_transitions_0", n_transitions_0)
    print("\n n_transitions_1", n_transitions_1)
    print("\n n_transitions_10", n_transitions_10)
    print("\n n_transitions_100", n_transitions_100)
    print("\n n_transitions_inf", n_transitions_inf)

    assert n_transitions_0 == start_temp_0
    assert (
        n_transitions_1
        == n_transitions_10
        == n_transitions_100
        == n_transitions_inf
        == n_iter - n_initialize
    )


def test_start_temp_1():
    n_initialize = 1

    start_temp_0 = 0
    start_temp_1 = 0.001
    start_temp_100 = 10000

    epsilon = 0.03

    opt = SimulatedAnnealingOptimizer(
        search_space,
        start_temp=start_temp_0,
        epsilon=epsilon,
        initialize={"random": n_initialize},
        random_state=42,
    )
    opt.search(objective_function, n_iter=n_iter)
    n_transitions_0 = opt.n_transitions

    opt = SimulatedAnnealingOptimizer(
        search_space,
        start_temp=start_temp_1,
        epsilon=epsilon,
        initialize={"random": n_initialize},
        random_state=100,
    )
    opt.search(objective_function, n_iter=n_iter)
    n_transitions_1 = opt.n_transitions

    opt = SimulatedAnnealingOptimizer(
        search_space,
        start_temp=start_temp_100,
        epsilon=epsilon,
        initialize={"random": n_initialize},
        random_state=100,
    )
    opt.search(objective_function, n_iter=n_iter)
    n_transitions_100 = opt.n_transitions

    print("\n n_transitions_0", n_transitions_0)
    print("\n n_transitions_1", n_transitions_1)
    print("\n n_transitions_100", n_transitions_100)

    assert n_transitions_0 == start_temp_0
    assert n_transitions_1 <= n_transitions_100


def test_annealing_rate_0():
    n_initialize = 1

    annealing_rate_0 = 0
    annealing_rate_1 = 0.1
    annealing_rate_100 = 0.99

    epsilon = 0.03

    opt = SimulatedAnnealingOptimizer(
        search_space,
        annealing_rate=annealing_rate_0,
        epsilon=epsilon,
        initialize={"random": n_initialize},
    )
    opt.search(objective_function, n_iter=n_iter)
    n_transitions_0 = opt.n_transitions

    opt = SimulatedAnnealingOptimizer(
        search_space,
        annealing_rate=annealing_rate_1,
        epsilon=epsilon,
        initialize={"random": n_initialize},
    )
    opt.search(objective_function, n_iter=n_iter)
    n_transitions_1 = opt.n_transitions

    opt = SimulatedAnnealingOptimizer(
        search_space,
        annealing_rate=annealing_rate_100,
        epsilon=epsilon,
        initialize={"random": n_initialize},
    )
    opt.search(objective_function, n_iter=n_iter)
    n_transitions_100 = opt.n_transitions

    print("\n n_transitions_0", n_transitions_0)
    print("\n n_transitions_1", n_transitions_1)
    print("\n n_transitions_100", n_transitions_100)

    assert n_transitions_0 in [0, 1]
    assert n_transitions_1 < n_transitions_100


@pytest.mark.parametrize(
    ("cooling", "annealing_rate", "expected_temp"),
    [
        ("exponential", 0.5, 5),
        ("linear", 0.75, 7.5),
        ("logarithmic", 0, 10 * math.log(2) / math.log(3)),
        ("cauchy", 0.5, 10 / 1.5),
        ("quadratic", 0.1, 10 / 1.1),
    ],
)
def test_cooling_schedules_update_temperature(cooling, annealing_rate, expected_temp):
    opt = SimulatedAnnealingOptimizer(
        search_space,
        cooling=cooling,
        annealing_rate=annealing_rate,
        start_temp=10,
    )

    opt._cool_temperature(accepted=False)

    assert np.isclose(opt.temp, expected_temp)


def test_adaptive_cooling_uses_recent_acceptance_rate():
    opt = SimulatedAnnealingOptimizer(
        search_space,
        cooling="adaptive",
        annealing_rate=0.5,
        start_temp=8,
    )

    for _ in range(opt._adaptive_window):
        opt._cool_temperature(accepted=False)

    assert np.isclose(opt.temp, 8 * 0.5 ** (opt._adaptive_window - 2))


def test_acceptance_criteria_probabilities():
    opt = SimulatedAnnealingOptimizer(search_space, acceptance="metropolis")
    opt.temp = 1
    opt._score_current = 1
    opt._score_new = 0

    assert np.isclose(opt._p_accept_default(), math.exp(-1))

    opt = SimulatedAnnealingOptimizer(search_space, acceptance="barker")
    opt.temp = 1
    opt._score_current = 1
    opt._score_new = 0

    assert np.isclose(opt._p_accept_default(), 1 / (1 + math.exp(1)))

    opt = SimulatedAnnealingOptimizer(search_space, acceptance="threshold")
    opt.temp = 0.5
    opt._score_current = 1
    opt._score_new = 0

    assert opt._p_accept_default() == 0

    opt.temp = 2

    assert opt._p_accept_default() == 1


def test_invalid_cooling_schedule_raises():
    with pytest.raises(ValueError, match="Unknown cooling schedule"):
        SimulatedAnnealingOptimizer(search_space, cooling="unsupported")


def test_invalid_acceptance_criterion_raises():
    with pytest.raises(ValueError, match="Unknown acceptance criterion"):
        SimulatedAnnealingOptimizer(search_space, acceptance="unsupported")
