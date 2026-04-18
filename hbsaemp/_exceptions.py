"""Custom exception and warning hierarchy for ``hbsaemp``.

All exceptions inherit from :exc:`HBSAEError`.

Hierarchy
---------
.. code-block:: text

    Exception
    └── HBSAEError
        ├── ValidationError
        │   ├── DataValidationError   — DataFrame issues
        │   ├── FormulaError          — bad formula string
        │   └── SpatialMatrixError    — v2+ adjacency/weight matrix
        ├── ModelRegistryError        — unknown family name in hbm()
        ├── ModelNotFittedError       — predict/summary before fit()
        └── EstimationError           — failure during SAE prediction

    UserWarning
    └── ConvergenceWarning            — Rhat > threshold, low ESS
"""

from __future__ import annotations

from typing import Any

__all__: list[str] = [
    "HBSAEError",
    "ValidationError",
    "DataValidationError",
    "FormulaError",
    "SpatialMatrixError",
    "ModelRegistryError",
    "ModelNotFittedError",
    "EstimationError",
    "ConvergenceWarning",
]


class HBSAEError(Exception):
    """Base for all ``hbsaemp`` errors.

    Args:
        message: Human-readable description.
        context: Machine-readable dict (e.g. ``{"column": "y", "n_nan": 3}``).
    """

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message: str = message
        self.context: dict[str, Any] = context or {}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.message!r}, context={self.context!r})"


# ── Validation ─────────────────────────────────────────────────────────────

class ValidationError(HBSAEError):
    """Parent for all input-validation failures."""


class DataValidationError(ValidationError):
    """Raised when a DataFrame fails validation.

    Args:
        message: Human-readable description.
        column: Offending column name, if applicable.
        context: Additional details.
    """

    def __init__(
        self,
        message: str,
        column: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        if column is not None:
            ctx["column"] = column
        super().__init__(message, ctx)
        self.column = column


class FormulaError(ValidationError):
    """Raised when a formula string cannot be parsed.

    Args:
        message: Human-readable description.
        formula: The formula string that failed.
        context: Additional details.
    """

    def __init__(
        self,
        message: str,
        formula: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        if formula is not None:
            ctx["formula"] = formula
        super().__init__(message, ctx)
        self.formula = formula


class SpatialMatrixError(ValidationError):
    """Raised when a spatial weight or adjacency matrix is invalid.

    .. note:: Defined in v0 for namespace stability; raised only in v2+ code.

    Args:
        message: Human-readable description.
        matrix_type: ``"car"`` or ``"sar"``.
        context: Additional details.
    """

    def __init__(
        self,
        message: str,
        matrix_type: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        if matrix_type is not None:
            ctx["matrix_type"] = matrix_type
        super().__init__(message, ctx)
        self.matrix_type = matrix_type


# ── Registry ────────────────────────────────────────────────────────────────

class ModelRegistryError(HBSAEError):
    """Raised when ``hbm()`` is called with an unknown ``family`` name.

    This is the ``hbsaemp`` equivalent of caret raising an error when an
    unknown ``method=`` string is passed to ``train()``.

    Args:
        family: The unrecognised family string.
        registered: List of currently registered family names.

    Example:
        >>> raise ModelRegistryError(
        ...     "Unknown family 'tweedie'",
        ...     family="tweedie",
        ...     registered=["gaussian", "beta", "binomial", "lognormal"],
        ... )
    """

    def __init__(
        self,
        message: str,
        family: str | None = None,
        registered: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        if family is not None:
            ctx["family"] = family
        if registered is not None:
            ctx["registered_families"] = registered
        super().__init__(message, ctx)
        self.family = family
        self.registered = registered or []


# ── Model state ─────────────────────────────────────────────────────────────

class ModelNotFittedError(HBSAEError):
    """Raised when ``predict()``, ``summary()``, or a diagnostic method is
    called before ``fit()`` has been executed.

    Example:
        >>> raise ModelNotFittedError(
        ...     "Call model.fit() before hbsae()",
        ...     context={"method": "hbsae"},
        ... )
    """


class EstimationError(HBSAEError):
    """Raised when posterior prediction or SAE computation fails."""


# ── Warnings ────────────────────────────────────────────────────────────────

class ConvergenceWarning(UserWarning):
    """Issued when MCMC convergence diagnostics indicate potential problems.

    Typical triggers: :math:`\\hat{R} > 1.01`, bulk ESS < 400.

    Example:
        >>> import warnings
        >>> warnings.warn("Rhat=1.05 for 'b_x1'", ConvergenceWarning, stacklevel=2)
    """
