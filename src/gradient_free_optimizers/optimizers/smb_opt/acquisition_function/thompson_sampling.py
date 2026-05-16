# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Thompson Sampling acquisition function."""

from gradient_free_optimizers._array_backend import array
from gradient_free_optimizers._array_backend import random as np_random

from ._utils import predict_mean_std


class ThompsonSampling:
    """Draw posterior score samples and select by sampled value."""

    def __init__(self, surrogate_model, position_l, rng=None):
        self.surrogate_model = surrogate_model
        self.position_l = position_l
        self.rng = rng if rng is not None else np_random.default_rng()

    def calculate(self, X_sample, Y_sample):
        """Draw one posterior score sample for each candidate position."""
        mu, sigma = predict_mean_std(self.surrogate_model, self.position_l)

        samples = []
        for i in range(len(mu)):
            sigma_i = max(float(sigma[i, 0]), 0.0)
            samples.append(self.rng.normal(float(mu[i, 0]), sigma_i))

        return array(samples)
