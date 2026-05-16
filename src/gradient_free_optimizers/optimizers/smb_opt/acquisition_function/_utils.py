# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Shared helpers for SMBO acquisition functions."""

from gradient_free_optimizers._array_backend import array


def predict_mean_std(surrogate_model, position_l):
    """Predict surrogate mean and standard deviation as column vectors."""
    mu, sigma = surrogate_model.predict(position_l, return_std=True)
    mu = array(mu).reshape(-1, 1)
    sigma = array(sigma).reshape(-1, 1)

    return mu, sigma
