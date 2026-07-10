"""Tests for SMBO acquisition functions."""

import pytest

from gradient_free_optimizers._array_backend import array
from gradient_free_optimizers.optimizers.smb_opt.acquisition_function import (
    ProbabilityOfImprovement,
    ThompsonSampling,
    normalize_acquisition_function_name,
)


class DummySurrogate:
    """Surrogate with fixed posterior predictions."""

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def predict(self, X, return_std=False):
        if return_std:
            return array(self.mu), array(self.sigma)
        return array(self.mu)


class FixedRNG:
    """RNG that exposes Thompson Sampling score construction."""

    def normal(self, loc, scale):
        return loc + scale


def test_probability_of_improvement_prefers_higher_improvement_probability():
    surrogate = DummySurrogate(mu=[1.0, 1.5], sigma=[1.0, 1.0])
    acquisition = ProbabilityOfImprovement(surrogate, array([[0.0], [1.0]]), xi=0.0)

    scores = acquisition.calculate(X_sample=None, Y_sample=[0.0, 1.0])

    assert scores[1] > scores[0]


def test_probability_of_improvement_zero_uncertainty_scores_zero():
    surrogate = DummySurrogate(mu=[2.0], sigma=[0.0])
    acquisition = ProbabilityOfImprovement(surrogate, array([[0.0]]), xi=0.0)

    scores = acquisition.calculate(X_sample=None, Y_sample=[0.0, 1.0])

    assert scores[0] == 0.0


def test_thompson_sampling_uses_posterior_mean_and_uncertainty():
    surrogate = DummySurrogate(mu=[1.0, 1.0], sigma=[0.1, 0.5])
    acquisition = ThompsonSampling(surrogate, array([[0.0], [1.0]]), rng=FixedRNG())

    scores = acquisition.calculate(X_sample=None, Y_sample=[0.0, 1.0])

    assert list(scores) == [1.1, 1.5]


def test_acquisition_function_name_aliases():
    assert normalize_acquisition_function_name("EI") == "expected_improvement"
    assert normalize_acquisition_function_name("pi") == "probability_of_improvement"
    assert normalize_acquisition_function_name("thompson") == "thompson_sampling"


def test_invalid_acquisition_function_name():
    with pytest.raises(ValueError, match="acquisition_function"):
        normalize_acquisition_function_name("not_supported")
