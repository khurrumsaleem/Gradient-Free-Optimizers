# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Optional tracking of optimizer-internal parameters.

Private backend feature, not part of the public API. Activated per run via
``search(..., _track_internals=True)``. The collected per-iteration state is
exposed through ``optimizer._internal_data`` and, when the dashboard decorator
is in use, carried to the dashboard on the ``SearchParams._internal_state``
attribute.
"""

from ._tracker import InternalParamTracker

__all__ = ["InternalParamTracker"]
