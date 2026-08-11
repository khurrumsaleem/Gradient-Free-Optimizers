"""Tests for optional optimizer-internal-parameter tracking.

Covers the contract of the private backend feature
``search(..., _track_internals=True)``:

- the off-by-default behavior (no tracker, ``_internal_data is None``),
- correct record count, ordering, phase labels and per-optimizer keys,
- that tracking does not perturb the search result (same seed -> same best),
- the dashboard bridge that exposes the snapshot as
  ``SearchParams._internal_state`` to the objective/decorator.
"""

import math

import numpy as np
import pytest

import gradient_free_optimizers as gfo
from gradient_free_optimizers import (
    GeneticAlgorithmOptimizer,
    HillClimbingOptimizer,
    ParticleSwarmOptimizer,
    SimulatedAnnealingOptimizer,
)

# Population optimizers need an explicit population; everything else uses defaults.
_POPULATION_OPTIMIZERS = {
    "CMAESOptimizer",
    "ParallelTemperingOptimizer",
    "ParticleSwarmOptimizer",
    "SpiralOptimization",
    "GeneticAlgorithmOptimizer",
    "EvolutionStrategyOptimizer",
    "DifferentialEvolutionOptimizer",
}


def _public_optimizers():
    out = []
    for name in gfo.__all__:
        obj = getattr(gfo, name)
        if isinstance(obj, type) and hasattr(obj, "search"):
            out.append((name, obj))
    return out


_OPTIMIZERS = _public_optimizers()

SEARCH_SPACE = {
    "x": np.arange(-5, 5, 0.1),
    "y": np.arange(-5, 5, 0.1),
}


def objective(params):
    return -(params["x"] ** 2 + params["y"] ** 2)


def _run(optimizer, n_iter=30, _track_internals=False, **search_kwargs):
    optimizer.search(
        objective,
        n_iter=n_iter,
        verbosity=False,
        memory=False,
        _track_internals=_track_internals,
        **search_kwargs,
    )
    return optimizer


def test_internal_data_is_none_when_tracking_disabled():
    opt = _run(SimulatedAnnealingOptimizer(SEARCH_SPACE), _track_internals=False)
    assert opt._param_tracker is None
    assert opt._internal_data is None


def test_internal_data_collected_per_evaluation():
    n_iter = 30
    opt = _run(
        SimulatedAnnealingOptimizer(SEARCH_SPACE),
        n_iter=n_iter,
        _track_internals=True,
    )
    records = opt._internal_data

    assert isinstance(records, list)
    assert len(records) == n_iter

    # Iteration indices are the running evaluation counter, contiguous from 0.
    assert [r["iteration"] for r in records] == list(range(n_iter))

    # Phase labels are present and only ever init or iter, init coming first.
    phases = [r["phase"] for r in records]
    assert set(phases) <= {"init", "iter"}
    assert "iter" in phases
    first_iter_idx = phases.index("iter")
    assert all(p == "init" for p in phases[:first_iter_idx])
    assert all(p == "iter" for p in phases[first_iter_idx:])


def test_simulated_annealing_exposes_temperature_schedule():
    opt = _run(SimulatedAnnealingOptimizer(SEARCH_SPACE), _track_internals=True)
    records = opt._internal_data

    for r in records:
        assert "temperature" in r
        assert "annealing_step" in r

    temps = [r["temperature"] for r in records]
    # Exponential cooling is monotonically non-increasing once the iteration
    # phase begins; the snapshot precedes the cooling step of its own iteration.
    assert all(later <= earlier + 1e-12 for earlier, later in zip(temps, temps[1:]))
    # Temperature must actually move, otherwise we are not tracking the schedule.
    assert temps[-1] < temps[0]


def test_particle_swarm_exposes_velocity_norm():
    opt = _run(
        ParticleSwarmOptimizer(SEARCH_SPACE, population=6),
        n_iter=40,
        _track_internals=True,
    )
    records = opt._internal_data
    velocity_records = [r for r in records if "velocity_norm" in r]

    # Most evaluations are during the iteration phase where a particle is active.
    assert velocity_records
    assert all(r["velocity_norm"] >= 0.0 for r in velocity_records)


def test_tracking_does_not_change_search_result():
    seed = 42
    off = _run(
        SimulatedAnnealingOptimizer(SEARCH_SPACE, random_state=seed),
        _track_internals=False,
    )
    on = _run(
        SimulatedAnnealingOptimizer(SEARCH_SPACE, random_state=seed),
        _track_internals=True,
    )
    assert on.best_score == off.best_score
    assert on.best_para == off.best_para


def test_dashboard_bridge_attaches_internal_state_to_params():
    seen = []

    def probing_objective(params):
        seen.append(getattr(params, "_internal_state", "ABSENT"))
        return -(params["x"] ** 2 + params["y"] ** 2)

    opt = SimulatedAnnealingOptimizer(SEARCH_SPACE)
    opt.search(
        probing_objective,
        n_iter=10,
        verbosity=False,
        memory=False,
        _track_internals=True,
    )

    assert len(seen) == 10
    assert all(isinstance(s, dict) for s in seen)
    assert all("temperature" in s for s in seen)


def test_dashboard_bridge_absent_when_tracking_disabled():
    seen = []

    def probing_objective(params):
        seen.append(getattr(params, "_internal_state", "ABSENT"))
        return -(params["x"] ** 2 + params["y"] ** 2)

    opt = SimulatedAnnealingOptimizer(SEARCH_SPACE)
    opt.search(probing_objective, n_iter=10, verbosity=False, memory=False)

    assert seen
    assert all(s == "ABSENT" for s in seen)


@pytest.mark.parametrize(("name", "cls"), _OPTIMIZERS, ids=[n for n, _ in _OPTIMIZERS])
def test_all_optimizers_track_finite_scalars(name, cls):
    """Every public optimizer exposes only finite scalars (or nullable None).

    Locks in the contract for all 23 optimizers: ``_internal_data`` is a list of
    one record per evaluation, and every tracked value other than the framework
    keys is a plain int/float/bool. ``None`` is tolerated only for genuinely
    nullable keys (``best_acquisition``/``lipschitz_constant`` early on). NaN and
    inf are rejected, which is what catches the class of bug where an ``-inf``
    score sentinel leaks into a population-variance computation.
    """
    kwargs = {"population": 8} if name in _POPULATION_OPTIMIZERS else {}
    opt = cls(SEARCH_SPACE, **kwargs)
    n_iter = 12
    opt.search(
        objective,
        n_iter=n_iter,
        verbosity=False,
        memory=False,
        _track_internals=True,
    )

    records = opt._internal_data
    assert isinstance(records, list)
    assert len(records) == n_iter

    for r in records:
        for key, value in r.items():
            if key in ("iteration", "phase"):
                continue
            if value is None:
                # nullable: best_acquisition, lipschitz_constant, move_distance
                continue
            assert isinstance(
                value, int | float | bool
            ), f"{name}.{key} is not a scalar: {type(value)}"
            if isinstance(value, bool):
                continue
            assert math.isfinite(value), f"{name}.{key} is not finite: {value}"

    # Shared, optimizer-independent keys appear in every record.
    for r in records:
        assert "iters_since_best" in r
        assert isinstance(r["iters_since_best"], int)
        assert r["iters_since_best"] >= 0
        assert "move_distance" in r
        md = r["move_distance"]
        assert md is None or (isinstance(md, float) and math.isfinite(md) and md >= 0.0)


@pytest.mark.parametrize(
    "name",
    [
        "SimulatedAnnealingOptimizer",
        "ParticleSwarmOptimizer",
        "BayesianOptimizer",
        "CMAESOptimizer",
        "GeneticAlgorithmOptimizer",
    ],
)
def test_iters_since_best_invariant_in_iteration_phase(name):
    """In the iteration phase iters_since_best resets to 0 or grows by one.

    Between two consecutive iteration-phase evaluations ``nth_trial`` advances
    by exactly one, and ``best_since_iter`` either stays put (no improvement,
    counter grows by one) or jumps to the new ``nth_trial`` (improvement,
    counter resets to zero). This is asserted only for the iteration phase: the
    initialization phase records improvements with a one-step offset because it
    updates the best before incrementing ``nth_trial`` while iteration does it
    after, a pre-existing detail of the core evaluation order.
    """
    cls = getattr(gfo, name)
    kwargs = {"population": 8} if name in _POPULATION_OPTIMIZERS else {}
    opt = cls(SEARCH_SPACE, **kwargs)
    opt.search(
        objective,
        n_iter=30,
        verbosity=False,
        memory=False,
        _track_internals=True,
    )
    iter_isb = [
        r["iters_since_best"] for r in opt._internal_data if r["phase"] == "iter"
    ]
    assert len(iter_isb) >= 2
    for prev, cur in zip(iter_isb, iter_isb[1:]):
        assert cur == 0 or cur == prev + 1, f"{name}: iters_since_best {prev} -> {cur}"


def test_move_distance_is_none_in_init_and_finite_in_iteration():
    """move_distance is None while initializing and a finite step length after.

    Scattered initialization points are not optimizer steps, so the shared
    move_distance is suppressed during the init phase and only carries a real
    normalized step length once the iteration phase generates candidates from
    an accepted position.
    """
    opt = SimulatedAnnealingOptimizer(SEARCH_SPACE)
    opt.search(
        objective,
        n_iter=25,
        verbosity=False,
        memory=False,
        _track_internals=True,
    )
    records = opt._internal_data
    init_moves = [r["move_distance"] for r in records if r["phase"] == "init"]
    iter_moves = [r["move_distance"] for r in records if r["phase"] == "iter"]

    assert init_moves and all(m is None for m in init_moves)
    assert iter_moves
    assert all(
        isinstance(m, float) and math.isfinite(m) and m >= 0.0 for m in iter_moves
    )


def test_move_distance_normalizes_mixed_dimension_types():
    """Each dimension contributes distance in its appropriate internal scale."""
    opt = HillClimbingOptimizer(
        {
            "continuous": (-10.0, 10.0),
            "discrete": np.array([0, 2, 100]),
            "categorical": ["first", "second", "third"],
        }
    )
    opt.search_state = "iter"
    opt._pos_current = np.array([-10.0, 0.0, 0.0])
    opt._pos_new = np.array([0.0, 1.0, 1.0])

    move_distance = opt._collect_shared_state()["move_distance"]

    # Continuous: 10 / 20 = 0.5; discrete: 1 / 2 = 0.5;
    # categorical: any category change contributes 1.0.
    assert move_distance == pytest.approx(math.sqrt(0.5**2 + 0.5**2 + 1.0**2))


def test_move_distance_normalizes_distribution_quantiles():
    """Distribution positions are normalized in their internal quantile space."""
    scipy_stats = pytest.importorskip("scipy.stats")
    opt = HillClimbingOptimizer({"distributed": scipy_stats.norm()})
    lower, upper = opt.conv.dim_infos[0].bounds
    midpoint = (lower + upper) / 2
    opt.search_state = "iter"
    opt._pos_current = np.array([lower])
    opt._pos_new = np.array([midpoint])

    move_distance = opt._collect_shared_state()["move_distance"]

    assert move_distance == pytest.approx(0.5)


def test_genetic_algorithm_population_std_finite_when_population_exceeds_init():
    """Regression: GA population-score std must stay finite.

    With ``population`` larger than the initialization budget, several
    individuals keep the ``-inf`` score sentinel for the whole run. Filtering
    those out (like DifferentialEvolution does) is required, otherwise the
    variance evaluates to NaN and silently corrupts the tracked column.
    """
    opt = GeneticAlgorithmOptimizer(
        SEARCH_SPACE, population=30, initialize={"random": 4}
    )
    opt.search(
        objective,
        n_iter=20,
        verbosity=False,
        memory=False,
        _track_internals=True,
    )
    stds = [r["population_score_std"] for r in opt._internal_data]
    assert stds
    assert all(math.isfinite(s) for s in stds)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
