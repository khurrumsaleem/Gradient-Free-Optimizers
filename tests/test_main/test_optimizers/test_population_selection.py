"""Selection behavior tests for population optimizers."""

import numpy as np

from gradient_free_optimizers import (
    DifferentialEvolutionOptimizer,
    EvolutionStrategyOptimizer,
    GeneticAlgorithmOptimizer,
)
from gradient_free_optimizers._array_backend import array

SEARCH_SPACE = {"x": np.array([0, 1, 2])}


def _seed_current_individual(opt, score=10.0):
    ind = opt.individuals[0]
    pos = array([0])

    ind._pos_new = pos.copy()
    ind._score_new = score
    ind._pos_current = pos.copy()
    ind._score_current = score
    ind._pos_best = pos.copy()
    ind._score_best = score

    opt.p_current = ind
    return ind


def _seed_population(opt, scores):
    for position, (ind, score) in enumerate(zip(opt.individuals, scores)):
        pos = array([position % len(SEARCH_SPACE["x"])])
        ind._pos_new = pos.copy()
        ind._score_new = score
        ind._pos_current = pos.copy()
        ind._score_current = score
        ind._pos_best = pos.copy()
        ind._score_best = score


def _evaluate_es_candidate(opt, individual, position, score):
    pos = array([position])
    individual._pos_new = pos.copy()
    opt.p_current = individual
    opt._current_candidate_sigma = individual.sigma
    opt._pos_new = pos.copy()
    opt._on_evaluate(score_new=score)


def test_differential_evolution_rejects_worse_trial():
    opt = DifferentialEvolutionOptimizer(
        SEARCH_SPACE,
        initialize={"random": 3},
        population=3,
        random_state=0,
    )
    ind = _seed_current_individual(opt, score=10.0)

    opt._pos_new = array([2])
    opt._on_evaluate(score_new=0.0)

    assert ind._score_current == 10.0
    assert ind._pos_current[0] == 0
    assert ind._score_best == 10.0
    assert ind._score_new_list[-1] == 0.0


def test_differential_evolution_accepts_equal_trial():
    opt = DifferentialEvolutionOptimizer(
        SEARCH_SPACE,
        initialize={"random": 3},
        population=3,
        random_state=0,
    )
    ind = _seed_current_individual(opt, score=10.0)

    opt._pos_new = array([1])
    opt._on_evaluate(score_new=10.0)

    assert ind._score_current == 10.0
    assert ind._pos_current[0] == 1


def test_evolution_strategy_replace_parents_waits_for_full_generation():
    opt = EvolutionStrategyOptimizer(
        SEARCH_SPACE,
        initialize={"random": 2},
        population=2,
        offspring=2,
        replace_parents=True,
        random_state=0,
    )
    _seed_population(opt, [10.0, 9.0])

    _evaluate_es_candidate(opt, opt.individuals[0], position=1, score=1.0)

    assert [ind._score_current for ind in opt.individuals] == [10.0, 9.0]
    assert len(opt._offspring_records) == 1

    _evaluate_es_candidate(opt, opt.individuals[1], position=2, score=2.0)

    assert sorted(ind._score_current for ind in opt.individuals) == [1.0, 2.0]
    assert len(opt._offspring_records) == 0


def test_evolution_strategy_plus_strategy_keeps_better_parents():
    opt = EvolutionStrategyOptimizer(
        SEARCH_SPACE,
        initialize={"random": 2},
        population=2,
        offspring=2,
        replace_parents=False,
        random_state=0,
    )
    _seed_population(opt, [10.0, 9.0])

    _evaluate_es_candidate(opt, opt.individuals[0], position=1, score=1.0)
    _evaluate_es_candidate(opt, opt.individuals[1], position=2, score=2.0)

    assert [ind._score_current for ind in opt.individuals] == [10.0, 9.0]
    assert len(opt._offspring_records) == 0


def test_evolution_strategy_comma_strategy_requires_enough_offspring():
    with np.testing.assert_raises(ValueError):
        EvolutionStrategyOptimizer(
            SEARCH_SPACE,
            initialize={"random": 3},
            population=3,
            offspring=2,
            replace_parents=True,
            random_state=0,
        )


def test_genetic_algorithm_rank_weighted_selection_uses_sorted_probabilities():
    opt = GeneticAlgorithmOptimizer(
        SEARCH_SPACE,
        initialize={"random": 4},
        population=4,
        random_state=0,
    )
    _seed_population(opt, [4.0, 3.0, 2.0, 1.0])

    class ChoiceSpy:
        def __init__(self):
            self.probabilities = None

        def choice(self, n_choices, p=None):
            self.probabilities = p
            return 0

    spy = ChoiceSpy()
    opt._rng = spy

    selected_idx = opt._select_rank_weighted_individual()

    assert selected_idx == 0
    assert opt.p_current is opt.pop_sorted[0]
    assert np.allclose(spy.probabilities, [0.4, 0.3, 0.2, 0.1])


def test_genetic_algorithm_rejects_worse_trial():
    opt = GeneticAlgorithmOptimizer(
        SEARCH_SPACE,
        initialize={"random": 2},
        population=2,
        random_state=0,
    )
    ind = _seed_current_individual(opt, score=10.0)

    opt._pos_new = array([2])
    opt._on_evaluate(score_new=0.0)

    assert ind._score_current == 10.0
    assert ind._pos_current[0] == 0
    assert ind._score_best == 10.0
    assert ind._score_new_list[-1] == 0.0
