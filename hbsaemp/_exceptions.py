"""Exception and warning hierarchy for hbsaemp.

All exceptions inherit from `HBSAEError`. Hierarchy:

    HBSAEError
      ValidationError
        DataValidationError    DataFrame issues
        FormulaError           bad formula string
        PriorSpecError         invalid prior specification
        SpatialMatrixError     v2+ adjacency/weight matrix
      ModelRegistryError       unknown family in create_model()
      ModelNotFittedError      predict/summary before fit()
      EstimationError          failure during SAE prediction

    ConvergenceWarning (UserWarning)
"""

from __future__ import annotations

from typing import Any

__all__: list[str] = [
    "HBSAEError",
    "ValidationError",
    "DataValidationError",
    "FormulaError",
    "PriorSpecError",
    "SpatialMatrixError",
    "ModelRegistryError",
    "ModelNotFittedError",
    "EstimationError",
    "ConvergenceWarning",
]


class HBSAEError(Exception):
    """Base for all hbsaemp errors.

    `context` is a machine-readable dict for diagnostics (e.g.
    `{"column": "y", "n_nan": 3}`).
    """

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message: str = message
        self.context: dict[str, Any] = context or {}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.message!r}, context={self.context!r})"


class ValidationError(HBSAEError):
    """Parent for all input-validation failures."""


class DataValidationError(ValidationError):
    """A DataFrame failed validation. `column` is the offending column, if known."""

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
    """A formula string failed parsing. `formula` is the offending string."""

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


class PriorSpecError(ValidationError):
    """A prior specification dict was malformed.

    Raised by `Prior` at construction so spec errors surface before `fit()`.
    `param` is the model parameter the prior was attached to, if known.
    """

    def __init__(
        self,
        message: str,
        param: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        if param is not None:
            ctx["param"] = param
        super().__init__(message, ctx)
        self.param = param


class SpatialMatrixError(ValidationError):
    """Invalid spatial weight or adjacency matrix (v2+ only)."""

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


class ModelRegistryError(HBSAEError):
    """`create_model()` was called with an unknown `family` name."""

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


class ModelNotFittedError(HBSAEError):
    """`predict()`, `summary()`, or a diagnostic was called before `fit()`."""


class EstimationError(HBSAEError):
    """Posterior prediction or SAE computation failed."""


class ConvergenceWarning(UserWarning):
    """MCMC convergence diagnostics indicate potential problems.

    Typical triggers: r-hat > 1.01, bulk or tail ESS < 100 × n_chains,
    divergent transitions, maximum tree depth hits, E-BFMI < 0.3.
    """
