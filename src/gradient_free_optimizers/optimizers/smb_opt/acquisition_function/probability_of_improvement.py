# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Probability of Improvement acquisition function."""

from gradient_free_optimizers._array_backend import array, zeros_like
from gradient_free_optimizers._math_backend import norm_cdf

from .._normalize import normalize
from ._utils import predict_mean_std


class ProbabilityOfImprovement:
    """Probability of Improvement acquisition function."""

    def __init__(self, surrogate_model, position_l, xi):
        self.surrogate_model = surrogate_model
        self.position_l = position_l
        self.xi = xi

    def calculate(self, X_sample, Y_sample):
        """Compute improvement probabilities for all candidate positions."""
        mu, sigma = predict_mean_std(self.surrogate_model, self.position_l)
        Y_sample = normalize(array(Y_sample)).reshape(-1, 1)

        imp = mu - Y_sample.max() - self.xi

        Z = zeros_like(sigma)
        for i in range(len(sigma)):
            if sigma[i, 0] != 0:
                Z[i, 0] = imp[i, 0] / sigma[i, 0]

        acqu_func = norm_cdf(Z)

        for i in range(len(sigma)):
            if sigma[i, 0] == 0.0:
                acqu_func[i, 0] = 0.0

        return acqu_func[:, 0]
