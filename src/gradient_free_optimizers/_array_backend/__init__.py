"""
Array backend abstraction for optional NumPy dependency.

This module provides a unified interface for array operations with automatic
fallback: numpy (fastest) -> pure Python (functional).

Usage:
    from gradient_free_optimizers._array_backend import array, zeros, clip, rint

Set ``GFO_ARRAY_BACKEND=pure`` to exercise the pure Python backend even when
NumPy is installed. Without it the pure path can only be reached by
uninstalling NumPy, which puts a large amount of code out of local reach.
``GFO_ARRAY_BACKEND=numpy`` pins the NumPy backend and turns a missing NumPy
into an error instead of a silent downgrade.
"""

import os

_BACKEND_ENV = "GFO_ARRAY_BACKEND"
_VALID_BACKENDS = ("numpy", "pure")

_requested = os.environ.get(_BACKEND_ENV)
if _requested is not None and _requested not in _VALID_BACKENDS:
    raise ValueError(
        f"{_BACKEND_ENV} must be one of {_VALID_BACKENDS}, got {_requested!r}"
    )

try:
    import numpy

    _ = numpy.__version__
    from numpy import array as _test_array

    del _test_array
    NUMPY_IMPORTABLE = True
except (ImportError, AttributeError):
    NUMPY_IMPORTABLE = False

if _requested == "numpy" and not NUMPY_IMPORTABLE:
    raise ImportError(
        f"{_BACKEND_ENV}=numpy was requested but NumPy could not be imported."
    )

# NUMPY_IMPORTABLE states whether NumPy exists in this environment, HAS_NUMPY
# whether the active backend uses it. They differ only when the pure backend is
# pinned. Consumers read HAS_NUMPY to decide whether a NumPy fast path may be
# taken, so pinning has to switch those off as well; a forced run that still
# executed NumPy code would prove nothing. Tests that need NumPy purely as a
# reference to compare the pure backend against read NUMPY_IMPORTABLE, so
# pinning narrows what runs rather than silently skipping the comparison.
HAS_NUMPY = NUMPY_IMPORTABLE and _requested != "pure"

if HAS_NUMPY:
    from ._numpy import *

    _backend_name = "numpy"
    ndarray = numpy.ndarray
else:
    from ._pure import *
    from ._pure import GFOArray

    _backend_name = "pure"
    ndarray = GFOArray


__all__ = [
    "HAS_NUMPY",
    "NUMPY_IMPORTABLE",
    "_backend_name",
    # Array creation
    "array",
    "asarray",
    "zeros",
    "zeros_like",
    "ones",
    "empty",
    "empty_like",
    "full",
    "arange",
    "linspace",
    "meshgrid",
    "eye",
    "diag",
    # Type conversion
    "int32",
    "int64",
    "float32",
    "float64",
    # Mathematical operations
    "sum",
    "mean",
    "std",
    "var",
    "prod",
    "cumsum",
    # Element-wise math
    "exp",
    "log",
    "log10",
    "sqrt",
    "abs",
    "power",
    "square",
    "sin",
    "cos",
    # Rounding and clipping
    "clip",
    "rint",
    "round",
    "floor",
    "ceil",
    # Comparison and logic
    "maximum",
    "minimum",
    "greater",
    "less",
    "equal",
    "isnan",
    "isinf",
    "isfinite",
    # Index operations
    "argmax",
    "argmin",
    "argsort",
    "where",
    "nonzero",
    "searchsorted",
    "take",
    # Set operations
    "unique",
    "intersect1d",
    "isin",
    # Array manipulation
    "reshape",
    "transpose",
    "ravel",
    "flatten",
    "concatenate",
    "stack",
    "vstack",
    "hstack",
    "tile",
    "repeat",
    "array_split",
    "split",
    # Linear algebra
    "dot",
    "matmul",
    "outer",
    "triu",
    "invert",
    "linalg",
    # Random number generation
    "random",
    # Constants
    "inf",
    "pi",
    "e",
    "nan",
    # Utility
    "copy",
    "allclose",
    "all",
    "any",
    "ndim",
    "shape",
    "ndarray",
]
