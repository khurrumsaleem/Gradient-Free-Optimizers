# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""In-memory collection of optimizer-internal parameters.

The tracker is created by ``search(..., track_internals=True)`` and attached
to the optimizer as ``self._param_tracker``. When tracking is not requested
the attribute stays ``None`` and the per-iteration hook in
``Search._evaluate_position`` short-circuits on a single identity check, so a
non-tracking run keeps its original hot-loop cost.
"""

from __future__ import annotations


class InternalParamTracker:
    """Collects optimizer-internal parameters once per evaluation.

    Each record is a flat dict ``{"iteration": int, "phase": str, **state}``
    where *state* is whatever the optimizer's ``_collect_state()`` returns.
    Records are kept in memory only; build a DataFrame downstream with
    ``pandas.DataFrame(opt.internal_data)`` or stream them to the dashboard.

    The records list is the single source of truth for the run. It is appended
    to in evaluation order, including the initialization phase (``phase ==
    "init"``) and the iteration phase (``phase == "iter"``).
    """

    def __init__(self) -> None:
        self.records: list[dict] = []

    def collect(self, iteration: int, phase: str, state: dict) -> None:
        """Append one per-evaluation snapshot.

        Parameters
        ----------
        iteration : int
            The running evaluation index (``Search._iter``), counting both
            initialization and iteration phases from zero.
        phase : str
            Either ``"init"`` or ``"iter"``.
        state : dict
            The optimizer-internal parameters returned by
            ``optimizer._collect_state()``. Keys are merged into the record
            directly; ``iteration`` and ``phase`` keys returned by an optimizer
            would be overwritten by the framework values and should be avoided.
        """
        record = {"iteration": iteration, "phase": phase}
        record.update(state)
        self.records.append(record)
