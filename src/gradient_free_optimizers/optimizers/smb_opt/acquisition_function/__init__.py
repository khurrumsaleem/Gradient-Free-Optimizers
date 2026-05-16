# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .expected_improvement import ExpectedImprovement
from .probability_of_improvement import ProbabilityOfImprovement
from .thompson_sampling import ThompsonSampling

_ACQUISITION_FUNCTIONS = {
    "expected_improvement": ExpectedImprovement,
    "probability_of_improvement": ProbabilityOfImprovement,
    "thompson_sampling": ThompsonSampling,
}

_ACQUISITION_FUNCTION_ALIASES = {
    "ei": "expected_improvement",
    "expected_improvement": "expected_improvement",
    "pi": "probability_of_improvement",
    "probability_improvement": "probability_of_improvement",
    "probability_of_improvement": "probability_of_improvement",
    "thompson": "thompson_sampling",
    "ts": "thompson_sampling",
    "thompson_sampling": "thompson_sampling",
}


def normalize_acquisition_function_name(acquisition_function):
    """Return the canonical acquisition function name."""
    if not isinstance(acquisition_function, str):
        raise ValueError(
            "acquisition_function must be a string, "
            f"got {type(acquisition_function).__name__}"
        )

    name = acquisition_function.strip().lower().replace("-", "_")
    if name not in _ACQUISITION_FUNCTION_ALIASES:
        valid_names = tuple(_ACQUISITION_FUNCTION_ALIASES)
        raise ValueError(
            f"acquisition_function must be one of {valid_names}, "
            f"got {acquisition_function!r}"
        )

    return _ACQUISITION_FUNCTION_ALIASES[name]


def create_acquisition_function(
    acquisition_function,
    surrogate_model,
    position_l,
    xi,
    rng=None,
):
    """Create an acquisition function from a public name or alias."""
    acquisition_function = normalize_acquisition_function_name(acquisition_function)
    acquisition_class = _ACQUISITION_FUNCTIONS[acquisition_function]

    if acquisition_function == "thompson_sampling":
        return acquisition_class(surrogate_model, position_l, rng=rng)

    return acquisition_class(surrogate_model, position_l, xi)


__all__ = [
    "ExpectedImprovement",
    "ProbabilityOfImprovement",
    "ThompsonSampling",
    "create_acquisition_function",
    "normalize_acquisition_function_name",
]
