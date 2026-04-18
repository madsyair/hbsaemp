"""Abstract base class and result container for all HBSAE models.

:class:`BaseModel` defines the unified interface that makes hbsaemp work
like caret: every family (Gaussian, Beta, Binomial, Lognormal) inherits
this and exposes identical ``fit()`` / ``predict()`` / ``summary()`` methods.

:class:`ModelResult` is the concrete dataclass returned by ``fit()``,
analogous to the object returned by ``caret::train()``.

Naming rationale
----------------
.. code-block:: text

    Before (R-style acronyms)  →  After (Python descriptive)
    ─────────────────────────     ──────────────────────────
    HBModel                    →  BaseModel
    HBMFit                     →  ModelResult
    HBMControl                 →  ModelConfig   (see _config.py)
    HBMGaussian                →  GaussianModel (see _gaussian.py)
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from hbsaemp._exceptions import ModelNotFittedError
from hbsaemp._logging import get_logger
from hbsaemp._types import FamilyLiteral, FormulaStr, PriorDict

if TYPE_CHECKING:
    from hbsaemp.models._config import ModelConfig

logger = get_logger(__name__)

__all__: list[str] = ["BaseModel", "ModelResult"]


# ---------------------------------------------------------------------------
# ModelResult — concrete result container  (was HBMFit)
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    """Result container for a fitted HBSAE model.

    In v0 all fields are ``None`` / defaults.
    In v1: ``backend_model`` holds a ``bambi.Model``,
           ``idata`` holds an ``arviz.InferenceData``.

    Attributes:
        backend_model: The fitted backend model (``bambi.Model`` in v1).
        idata: Posterior samples as ``arviz.InferenceData`` (v1).
        formula: Formula string used for fitting.
        family: Distribution family.
        data: The cleaned DataFrame used for fitting.
        config: MCMC configuration used for this fit.
        priors: Prior specification dict, or ``None`` for auto priors.
        is_fitted: ``True`` after ``fit()`` completes.
        fitted_at: UTC datetime when ``fit()`` completed.
        extra: Family-specific metadata (e.g. ``{"phi": array}`` for Beta).
    """

    backend_model: Any = field(default=None, repr=False)
    idata: Any = field(default=None, repr=False)
    formula: FormulaStr = ""
    family: FamilyLiteral = "gaussian"
    data: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    config: Any = field(default=None)           # ModelConfig — avoids circular
    priors: PriorDict | None = field(default=None, repr=False)
    is_fitted: bool = False
    fitted_at: datetime | None = None
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return (
            f"ModelResult(family={self.family!r}, "
            f"formula={self.formula!r}, "
            f"n={len(self.data)}, status={status})"
        )

    def summary(self) -> str:
        """Human-readable summary. Delegates to ArviZ in v1."""
        if not self.is_fitted:
            return f"ModelResult [{self.family}] — not fitted yet."
        return (
            f"ModelResult [{self.family}]\n"
            f"  Formula : {self.formula}\n"
            f"  n       : {len(self.data)}\n"
            f"  Fitted  : {self.fitted_at}\n"
            f"  Backend : {type(self.backend_model).__name__ if self.backend_model else 'stub'}"
        )


# ---------------------------------------------------------------------------
# BaseModel — abstract base class  (was HBModel)
# ---------------------------------------------------------------------------

class BaseModel(abc.ABC):
    """Abstract base for all HBSAE distribution models.

    Subclasses (:class:`GaussianModel`, :class:`BetaModel`, etc.) must
    implement :meth:`fit` and :meth:`predict`.

    The constructor intentionally accepts all possible keyword arguments so
    that :func:`~hbsaemp.models._factory.hbm` can instantiate any family
    with the same call signature.

    Args:
        formula: R/lme4-style formula, e.g. ``"y ~ x1 + (1|group)"``.
        family: Distribution family name.
        data: Input DataFrame.
        config: MCMC configuration (:class:`~hbsaemp.models._config.ModelConfig`).
        priors: Optional prior dict; ``None`` = Bambi auto priors.
        group: Grouping column for random effects.
        handle_missing: Missing data strategy (``"deleted"`` in v1).
        **kwargs: Family-specific extras (``trials``, ``n``, ``deff``, …).
    """

    def __init__(
        self,
        formula: FormulaStr,
        family: FamilyLiteral,
        data: pd.DataFrame,
        config: ModelConfig,
        *,
        priors: PriorDict | None = None,
        group: str | None = None,
        handle_missing: str = "deleted",
        **kwargs: Any,
    ) -> None:
        self._formula = formula
        self._family = family
        self._data = data
        self._config = config
        self._priors = priors
        self._group = group
        self._handle_missing = handle_missing
        self._kwargs = kwargs
        self._result: ModelResult | None = None

        logger.debug(
            "Created %s(family=%r, n=%d)", type(self).__name__, family, len(data)
        )

    # ── Read-only properties ─────────────────────────────────────────────────

    @property
    def formula(self) -> FormulaStr:
        """Formula string."""
        return self._formula

    @property
    def family(self) -> FamilyLiteral:
        """Distribution family."""
        return self._family

    @property
    def data(self) -> pd.DataFrame:
        """Training data (read-only)."""
        return self._data

    @property
    def config(self) -> ModelConfig:
        """MCMC configuration."""
        return self._config

    @property
    def is_fitted(self) -> bool:
        """``True`` after :meth:`fit` completes."""
        return self._result is not None and self._result.is_fitted

    @property
    def result(self) -> ModelResult:
        """The :class:`ModelResult` from the last :meth:`fit` call.

        Raises:
            ModelNotFittedError: If :meth:`fit` has not been called.
        """
        if self._result is None or not self._result.is_fitted:
            raise ModelNotFittedError(
                f"Call {type(self).__name__}.fit() before accessing .result.",
                context={"family": self._family, "formula": self._formula},
            )
        return self._result

    # ── Abstract interface ───────────────────────────────────────────────────

    @abc.abstractmethod
    def fit(self) -> ModelResult:
        """Fit the model using the MCMC backend.

        Returns:
            :class:`ModelResult` with posterior samples and metadata.

        Raises:
            NotImplementedError: In v0 stubs.
            DataValidationError: If data fails pre-fit checks.
        """

    @abc.abstractmethod
    def predict(
        self,
        new_data: pd.DataFrame | None = None,
        *,
        kind: str = "response",
        n_samples: int | None = None,
    ) -> np.ndarray:
        """Draw posterior predictive samples.

        Args:
            new_data: Out-of-sample data. ``None`` uses training data.
            kind: ``"response"`` (default) or ``"linear"`` (link scale).
            n_samples: Number of posterior draws to use.

        Returns:
            Array of shape ``(n_samples, n_areas)``.

        Raises:
            ModelNotFittedError: If :meth:`fit` has not been called.
            NotImplementedError: In v0 stubs.
        """

    # ── Concrete helpers ─────────────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable model summary.

        In v0 returns a placeholder.  In v1, delegates to ``arviz.summary``.
        """
        if not self.is_fitted:
            return (
                f"{type(self).__name__} [not fitted]\n"
                f"  Formula : {self._formula}\n"
                f"  Family  : {self._family}\n"
                f"  n       : {len(self._data)}\n"
                f"  Config  : draws={self._config.draws}, "
                f"chains={self._config.chains}"
            )
        return self._result.summary()  # type: ignore[union-attr]

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return (
            f"{type(self).__name__}("
            f"family={self._family!r}, "
            f"n={len(self._data)}, "
            f"status={status})"
        )
