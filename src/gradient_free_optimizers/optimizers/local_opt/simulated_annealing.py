# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Simulated Annealing Optimizer."""

import math
from collections import deque

from .stochastic_hill_climbing import StochasticHillClimbingOptimizer


class SimulatedAnnealingOptimizer(StochasticHillClimbingOptimizer):
    """Simulated Annealing optimizer inspired by metallurgical annealing.

    Uses a temperature parameter that decreases over time, controlling the
    probability of accepting worse solutions. High temperature allows more
    exploration; low temperature focuses on exploitation.

    Dimension Support:
        - Continuous: YES (inherited from HillClimbingOptimizer)
        - Categorical: YES (inherited from HillClimbingOptimizer)
        - Discrete: YES (inherited from HillClimbingOptimizer)

    The default acceptance probability follows the Metropolis criterion:
        p = exp(normalized_energy / temp)

    Where normalized_energy = (score_new - score_current) / (score_new + score_current)

    By default, the temperature decreases each iteration: temp *= annealing_rate

    Parameters
    ----------
    search_space : dict
        Dictionary mapping parameter names to search dimension definitions.
    initialize : dict, optional
        Strategy for generating initial positions.
    constraints : list, optional
        List of constraint functions.
    random_state : int, optional
        Seed for random number generation.
    rand_rest_p : float, default=0
        Probability of random restart to escape local optima.
    nth_process : int, optional
        Process index for parallel optimization.
    epsilon : float, default=0.03
        Step size for generating neighbors (fraction of search space).
    distribution : str, default="normal"
        Distribution for step sizes: "normal", "laplace", or "logistic".
    n_neighbours : int, default=3
        Number of neighbors to evaluate before selecting the best.
    annealing_rate : float, default=0.97
        Schedule-specific cooling configuration. For the default exponential
        schedule this is the multiplicative decay rate.
    start_temp : float, default=1
        Initial temperature value.
    cooling : str, default="exponential"
        Cooling schedule: "exponential", "linear", "logarithmic", "cauchy",
        "quadratic", or "adaptive".
    acceptance : str, default="metropolis"
        Acceptance criterion for worse moves: "metropolis", "barker", or
        "threshold".
    """

    name = "Simulated Annealing"
    _name_ = "simulated_annealing"
    __name__ = "SimulatedAnnealingOptimizer"

    optimizer_type = "local"
    computationally_expensive = False
    cooling_schedules = (
        "exponential",
        "linear",
        "logarithmic",
        "cauchy",
        "quadratic",
        "adaptive",
    )
    acceptance_criteria = ("metropolis", "barker", "threshold")

    _adaptive_window = 20
    _adaptive_low_acceptance = 0.2
    _adaptive_high_acceptance = 0.8

    def __init__(
        self,
        search_space,
        initialize=None,
        constraints=None,
        random_state=None,
        rand_rest_p=0,
        nth_process=None,
        boundary="clip",
        epsilon=0.03,
        distribution="normal",
        n_neighbours=3,
        annealing_rate=0.97,
        start_temp=1,
        cooling="exponential",
        acceptance="metropolis",
    ):
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
            boundary=boundary,
            epsilon=epsilon,
            distribution=distribution,
            n_neighbours=n_neighbours,
            # Note: p_accept is not used in SA, we use pure Metropolis criterion
        )
        self.annealing_rate = annealing_rate
        self.start_temp = start_temp
        self.temp = start_temp
        self.cooling = cooling
        self.acceptance = acceptance
        self._annealing_step = 0
        self._acceptance_history = deque(maxlen=self._adaptive_window)

        if cooling not in self.cooling_schedules:
            raise ValueError(
                f"Unknown cooling schedule '{cooling}'. "
                f"Choose from: {list(self.cooling_schedules)}"
            )
        if acceptance not in self.acceptance_criteria:
            raise ValueError(
                f"Unknown acceptance criterion '{acceptance}'. "
                f"Choose from: {list(self.acceptance_criteria)}"
            )
        if cooling == "adaptive" and not 0 < annealing_rate <= 1:
            raise ValueError(
                "annealing_rate must be in (0, 1] when cooling='adaptive'."
            )

    def _p_accept_default(self) -> float:
        """Calculate the configured acceptance probability.

        Note: The sign follows the maximization convention used by the base
        optimizer, so improving moves have positive normalized energy.

        Returns
        -------
        float
            Probability of accepting the current solution, in [0, 1].
        """
        if self.acceptance == "metropolis":
            return self._metropolis_acceptance()
        if self.acceptance == "barker":
            return self._barker_acceptance()
        if self.acceptance == "threshold":
            return self._threshold_acceptance()

        raise RuntimeError(f"Unsupported acceptance criterion: {self.acceptance!r}")

    def _metropolis_acceptance(self) -> float:
        """Calculate the Metropolis acceptance probability."""
        try:
            return min(1, math.exp(self._exponent))
        except OverflowError:
            return 1

    def _barker_acceptance(self) -> float:
        """Calculate the Barker logistic acceptance probability."""
        exponent = self._exponent
        if exponent >= 0:
            try:
                return 1 / (1 + math.exp(-exponent))
            except OverflowError:
                return 0

        exp_val = math.exp(exponent)
        return exp_val / (1 + exp_val)

    def _threshold_acceptance(self) -> float:
        """Calculate deterministic threshold acceptance."""
        if self._normalized_energy_state >= -self.temp:
            return 1
        return 0

    def _on_evaluate(self, score_new):
        """Evaluate with the configured acceptance criterion and cooling schedule.

        After the stochastic acceptance decision, the temperature is reduced
        according to the configured annealing schedule.

        Args:
            score_new: Score of the most recently evaluated position
        """
        previous_position = self._pos_current
        previous_score = self._score_current

        super()._on_evaluate(score_new)

        accepted = (
            self._pos_current is not previous_position
            or self._score_current != previous_score
        )
        self._cool_temperature(accepted)

    def _cool_temperature(self, accepted: bool) -> None:
        """Update the temperature according to the configured schedule."""
        self._annealing_step += 1

        if self.cooling == "adaptive":
            self._cool_adaptive(accepted)
        elif self.cooling == "exponential":
            self.temp *= self.annealing_rate
        else:
            self.temp = self._scheduled_temperature(self._annealing_step)

    def _scheduled_temperature(self, step: int) -> float:
        """Calculate the absolute temperature for non-adaptive schedules."""
        if self.cooling == "linear":
            scale = self._cooling_scale()
            return self.start_temp * max(0, 1 - scale * step)

        if self.cooling == "logarithmic":
            scale = self._cooling_scale()
            return self.start_temp * math.log(2) / math.log(2 + scale * step)

        if self.cooling == "cauchy":
            scale = self._cooling_scale()
            return self.start_temp / (1 + scale * step)

        if self.cooling == "quadratic":
            return self.start_temp / (1 + self.annealing_rate * step * step)

        raise RuntimeError(f"Unsupported cooling schedule: {self.cooling!r}")

    def _cooling_scale(self) -> float:
        """Convert annealing_rate into a non-negative cooling scale."""
        return max(0, 1 - self.annealing_rate)

    def _cool_adaptive(self, accepted: bool) -> None:
        """Cool or reheat based on the recent acceptance rate."""
        self._acceptance_history.append(accepted)
        self.temp *= self.annealing_rate

        if len(self._acceptance_history) < self._adaptive_window:
            return

        acceptance_rate = sum(self._acceptance_history) / self._adaptive_window
        if acceptance_rate < self._adaptive_low_acceptance:
            self.temp /= self.annealing_rate * self.annealing_rate
        elif acceptance_rate > self._adaptive_high_acceptance:
            self.temp *= self.annealing_rate

    def _iterate_batch(self, n):
        """Generate n positions via independent perturbations from current position."""
        return [self._generate_position() for _ in range(n)]

    def _evaluate_batch(self, positions, scores):
        """Process batch results through the standard evaluate chain."""
        for pos, score in zip(positions, scores):
            self._pos_new = pos
            self._evaluate(score)
