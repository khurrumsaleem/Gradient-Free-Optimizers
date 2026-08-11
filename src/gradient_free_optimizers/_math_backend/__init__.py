"""
Math backend abstraction for optional SciPy dependency.

This module provides a unified interface for mathematical functions
(statistics, linear algebra, distance calculations) with automatic fallback
to pure Python implementations when SciPy is not available.

Usage:
    from gradient_free_optimizers._math_backend import norm_cdf, norm_pdf, cdist

The backend automatically selects the fastest available implementation:
- If SciPy is installed: uses SciPy (fast, numerically stable)
- If not: uses pure Python implementations (slower but functional)

Set ``GFO_MATH_BACKEND=pure`` to exercise the pure Python implementations even
when SciPy is installed, mirroring ``GFO_ARRAY_BACKEND``.
``GFO_MATH_BACKEND=scipy`` pins SciPy and turns a missing SciPy into an error
instead of a silent downgrade.
"""

import os

_BACKEND_ENV = "GFO_MATH_BACKEND"
_VALID_BACKENDS = ("scipy", "pure")

_requested = os.environ.get(_BACKEND_ENV)
if _requested is not None and _requested not in _VALID_BACKENDS:
    raise ValueError(
        f"{_BACKEND_ENV} must be one of {_VALID_BACKENDS}, got {_requested!r}"
    )

try:
    import scipy

    # Verify scipy is fully installed, not just a namespace stub
    _ = scipy.__version__
    from scipy.linalg import cholesky as _test_cholesky

    del _test_cholesky
    SCIPY_IMPORTABLE = True
except (ImportError, AttributeError):
    SCIPY_IMPORTABLE = False

if _requested == "scipy" and not SCIPY_IMPORTABLE:
    raise ImportError(
        f"{_BACKEND_ENV}=scipy was requested but SciPy could not be imported."
    )

# SCIPY_IMPORTABLE states whether SciPy exists here, HAS_SCIPY whether the
# active backend uses it. Tests comparing the pure implementations against
# SciPy read the former so that pinning the pure backend does not skip them.
HAS_SCIPY = SCIPY_IMPORTABLE and _requested != "pure"


if HAS_SCIPY:
    from ._scipy import *

    _backend_name = "scipy"
else:
    from ._pure import *

    _backend_name = "pure"


__all__ = [
    "HAS_SCIPY",
    "SCIPY_IMPORTABLE",
    "_backend_name",
    # Statistical functions
    "norm_cdf",
    "norm_pdf",
    # Linear algebra
    "cholesky",
    "cho_solve",
    "solve",
    "solve_triangular",
    # Optimization
    "minimize",
    # Distance functions
    "cdist",
    # Special functions
    "logsumexp",
]
