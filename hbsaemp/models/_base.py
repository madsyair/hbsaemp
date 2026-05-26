"""Abstract base class and result container for all HBSAE models.

:class:`BaseModel` defines the unified interface that makes hbsaemp work
like caret: every family (Gaussian, Beta, Binomial, Lognormal) inherits
this and exposes identical ``fit()`` / ``predict()`` / ``summary()`` methods.

Since v1+, ``fit()`` and ``predict()`` are **concrete** in :class:`BaseModel`.
The MCMC pipeline (parse → validate → preprocess → Bambi → store result)
and the predictive draw extraction are shared across all families.
Subclasses customise only the family-specific parts via small hooks:

* :meth:`_extra_pipeline_kwargs` *(abstract)* — kwargs forwarded to validator
  and preprocessor.
* :meth:`_build_formula_and_link` *(abstract)* — Bambi formula + link spec.
* :meth:`_extra_result_dict` *(abstract)* — family-specific keys merged into
  ``ModelResult.extra``.
* :meth:`_pre_fit_checks` *(optional)* — early guards before ``bambi`` import.
* :meth:`_bambi_family` *(optional)* — override the Bambi family string.

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
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, ClassVar

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
        family: Distribution family (user-facing label — e.g. ``"lognormal"``
            even when the Bambi backend family is ``"gaussian"``).
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

    Subclasses (:class:`GaussianModel`, :class:`BetaModel`, etc.) inherit
    the concrete :meth:`fit` and :meth:`predict` implementations and
    customise only the family-specific hooks:

    Abstract hooks (subclass **must** implement)
        * :meth:`_build_formula_and_link` — Bambi formula and link spec
          (returns ``(formula, link)``).

    Concrete hooks with defaults (set :attr:`_EXTRA_FIELD_NAMES` or override)
        * :meth:`_extra_pipeline_kwargs` — kwargs forwarded to
          :class:`~hbsaemp.data._validator.DataValidator` and
          :class:`~hbsaemp.data._preprocessor.DataPreprocessor`.
          Default reads :attr:`_EXTRA_FIELD_NAMES`.
        * :meth:`_extra_result_dict` — family-specific keys merged into
          ``ModelResult.extra``.  Default delegates to
          :meth:`_extra_pipeline_kwargs`.
        * :meth:`_pre_fit_checks` — early guards before ``bambi`` is
          imported.  Default is a no-op.
        * :meth:`_bambi_family` — Bambi family string.  Default returns
          :attr:`_family`; :class:`LognormalModel` overrides this to
          return ``"gaussian"`` since lognormal is fit as Gaussian on
          log-scale data.
        * :attr:`_preproc_family` — family string for validator and
          preprocessor.  Default returns :attr:`_family`; overridden by
          :class:`LognormalModel` for the same reason.

    Subclasses **must** set :attr:`_MEAN_PARAM_KEY` (``"mu"`` or ``"p"``)
    and declare :attr:`_EXTRA_FIELD_NAMES` as a class-level tuple of the
    family-specific attribute names that the pipeline and result dict need.

    Family-specific kwargs (``n_col``, ``deff_col``, ``trials_col``,
    ``sampling_var_col``, ``squeeze``, ``link``) are consumed by the
    subclass constructors; unknown kwargs are not silently accepted here.
    See :data:`~hbsaemp.models._factory._FAMILY_PARAMS` for the dispatch.

    Args:
        formula: R/lme4-style formula, e.g. ``"y ~ x1 + (1|group)"``.
        family: Distribution family name (user-facing label).
        data: Input DataFrame.
        config: MCMC configuration (:class:`~hbsaemp.models._config.ModelConfig`).
        priors: Optional prior dict; ``None`` = Bambi auto priors.
        group: Grouping column for random effects.
        handle_missing: Missing data strategy (``"deleted"`` in v1).
    """

    #: Subclass must set this — key in ``idata.posterior`` used by
    #: :meth:`predict` when ``kind="response_params"``.
    #:
    #:   - ``"mu"`` for Gaussian / Beta / Lognormal (mean parameter)
    #:   - ``"p"`` for Binomial (success-probability parameter)
    _MEAN_PARAM_KEY: ClassVar[str]
    #: Family-specific attribute names forwarded to the data pipeline and
    #: ``ModelResult.extra``.  Each entry ``name`` maps to ``self._<name>``.
    #: Subclasses set this tuple; the default empty tuple means no extra kwargs.
    _EXTRA_FIELD_NAMES: ClassVar[tuple[str, ...]] = ()

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
    ) -> None:
        self._formula = formula
        self._family = family
        self._data = data
        self._config = config
        self._priors = priors
        self._group = group
        self._handle_missing = handle_missing
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
        """Distribution family (user-facing label)."""
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

    # ── Abstract hooks (subclass must implement) ─────────────────────────────

    def _extra_pipeline_kwargs(self) -> dict[str, Any]:
        """Family-specific keyword arguments for the data pipeline.

        The returned dict is unpacked into both
        :meth:`~hbsaemp.data._validator.DataValidator.validate` and
        :meth:`~hbsaemp.data._preprocessor.DataPreprocessor.process`
        by :meth:`_run_pipeline` and :meth:`_preprocess_new_data`.

        Default reads :attr:`_EXTRA_FIELD_NAMES` and returns
        ``{name: self._<name>}`` for each declared field.  Subclasses
        set :attr:`_EXTRA_FIELD_NAMES`; override this method only when
        the lookup logic itself must differ.

        Returns:
            Dict of keyword arguments (may be empty).
        """
        return {name: getattr(self, f"_{name}") for name in self._EXTRA_FIELD_NAMES}

    @abc.abstractmethod
    def _build_formula_and_link(
        self,
        bmb_module: Any,
        response: str,
    ) -> tuple[Any, Any]:
        """Build the Bambi formula and link specification for ``bmb.Model``.

        Called by :meth:`fit` after the data pipeline has run.

        Args:
            bmb_module: The imported ``bambi`` module — passed in so the
                subclass need not import ``bambi`` itself.  Use it to
                build distributional formulas via ``bmb_module.Formula``.
            response: Response column name as parsed from ``self._formula``
                (e.g. ``"y"``).  Useful for families that rewrite the
                formula LHS (e.g. Binomial's ``"y | trials(n) ~ rhs"``).

        Returns:
            ``(formula, link)`` pair.  ``formula`` may be a plain string
            or a :class:`bambi.Formula` object.  ``link`` may be a string
            (single link) or a dict (distributional models — must include
            **all** parameter keys, e.g. ``{"mu": "logit", "kappa": "log"}``).
        """

    def _extra_result_dict(self) -> dict[str, Any]:
        """Family-specific keys merged into ``ModelResult.extra``.

        :meth:`fit` always adds ``"response"`` and ``"group"`` to
        ``extra`` automatically.  Subclasses declare :attr:`_EXTRA_FIELD_NAMES`
        for the further keys their downstream consumers (predict,
        estimate_areas, update) need to read back later.

        Default delegates to :meth:`_extra_pipeline_kwargs` — the pipeline
        and result dicts are identical for all built-in families.  Override
        only when the result dict must differ from the pipeline dict.

        Returns:
            Dict of extra keys (may be empty).
        """
        return self._extra_pipeline_kwargs()

    # ── Concrete hooks (overridable) ─────────────────────────────────────────

    def _pre_fit_checks(self) -> None:
        """Early guard hook — runs before the ``bambi`` import in :meth:`fit`.

        Default is a no-op.  Subclasses override when they need to fail
        fast on missing required attributes before the (potentially slow)
        ``bambi`` import.  Example: :class:`BinomialModel` raises if
        ``trials_col`` was not supplied.
        """
        return None

    def _bambi_family(self) -> str:
        """The family string passed to ``bambi.Model(..., family=...)``.

        Default returns :attr:`_family` (the user-facing label).  Override
        when the Bambi backend family differs — for example,
        :class:`LognormalModel` returns ``"gaussian"`` because the model
        is fit as Gaussian on a log-scale response.  ``ModelResult.family``
        still keeps the user-facing label.
        """
        return self._family

    def _response_pp_key(self, response: str) -> str:
        """Key under which ``idata.posterior_predictive`` stores the response.

        Default returns *response* unchanged — correct for plain Gaussian /
        Beta / Lognormal formulae.  :class:`BinomialModel` overrides because
        Bambi 0.14+ stores PP under the wrapped LHS ``p(y, trials_col)``
        rather than the bare response column.
        """
        return response

    def _workaround_priors(self, bmb_module: Any) -> dict[str, Any]:
        """Auto-injected priors that work around backend (Bambi) bugs.

        Default returns an empty dict.  Subclasses override when the backend
        needs help avoiding internal failures (e.g. distributional components
        whose design matrix would otherwise be empty — see
        :class:`BetaModel` and :class:`GaussianModel`).

        User-supplied priors (``self._priors``) take precedence — the merge
        happens in :meth:`fit` with the workaround dict as the lower-priority
        base layer.
        """
        return {}

    @property
    def _preproc_family(self) -> str:
        """Family string passed to :class:`~hbsaemp.data._validator.DataValidator`
        and :class:`~hbsaemp.data._preprocessor.DataPreprocessor`.

        Subclasses may override this when the validator/preprocessor family
        differs from the user-facing :attr:`family`.
        :class:`LognormalModel` returns ``"gaussian"`` because its log-scale
        response can be negative (failing the lognormal positivity check).
        """
        return self._family

    # ── Pipeline helpers (concrete) ──────────────────────────────────────────

    def _run_pipeline(
        self,
        data: pd.DataFrame,
    ) -> tuple[str, list[str], list[str], str | None, pd.DataFrame]:
        """Common parse → validate → preprocess pipeline used by :meth:`fit`.

        Delegates family-specific kwargs to :meth:`_extra_pipeline_kwargs`
        and the preprocessor family to :attr:`_preproc_family`.

        Args:
            data: Raw training DataFrame (usually ``self._data``).

        Returns:
            Tuple ``(response, predictors, random_groups, group_col, df_clean)``
            where *df_clean* is a preprocessed copy ready for
            ``bambi.Model``.
        """
        from hbsaemp.data._preprocessor import DataPreprocessor
        from hbsaemp.data._validator import DataValidator
        from hbsaemp.utils._formula import parse_formula

        parsed = parse_formula(self._formula)
        response: str = parsed["response"]
        predictors: list[str] = parsed["fixed"]
        random_groups: list[str] = parsed["random_groups"]
        # Explicit group kwarg takes priority; fall back to parsed RE groups.
        group_col: str | None = self._group or (
            random_groups[0] if random_groups else None
        )

        extra_kwargs = self._extra_pipeline_kwargs()

        DataValidator(self._handle_missing).validate(
            data, response, predictors,
            family=self._preproc_family,
            group=group_col,
            **extra_kwargs,
        )
        df_clean: pd.DataFrame = DataPreprocessor(self._handle_missing).process(
            data, response, predictors,
            group=group_col,
            family=self._preproc_family,
            **extra_kwargs,
        )
        return response, predictors, random_groups, group_col, df_clean

    def _preprocess_new_data(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """Re-apply training-time preprocessing to out-of-sample data.

        Called by :meth:`predict` when ``new_data`` is not ``None``.
        Adds required offset columns (e.g. ``log_phi`` for Beta,
        ``log_sqrt_D`` for Gaussian/Lognormal FH) so that Bambi's
        distributional formula can evaluate them on the new rows.

        Validation is **skipped** — only the transform step is run.

        Args:
            new_data: Out-of-sample DataFrame with the same column
                structure as the training data.

        Returns:
            Preprocessed copy of ``new_data``.

        Raises:
            ModelNotFittedError: If :meth:`fit` has not been called.
        """
        from hbsaemp.data._preprocessor import DataPreprocessor
        from hbsaemp.utils._formula import parse_formula

        result = self.result  # raises ModelNotFittedError if not fitted
        response: str = result.extra["response"]
        predictors: list[str] = parse_formula(self._formula)["fixed"]
        group_col: str | None = result.extra.get("group")

        return DataPreprocessor(self._handle_missing).process(
            new_data, response, predictors,
            group=group_col,
            family=self._preproc_family,
            **self._extra_pipeline_kwargs(),
        )

    # ── predict() kind normalisation ─────────────────────────────────────────

    # Map of deprecated kind aliases → canonical Bambi 0.14+ kind.
    _KIND_ALIASES: dict[str, str] = {"pps": "response", "mean": "response_params"}
    # Kinds removed in Bambi 0.14+ — raise immediately before touching idata.
    _KIND_REMOVED: frozenset[str] = frozenset({"linear"})
    # Only these two canonical values are forwarded to ``bambi.Model.predict()``.
    _KIND_VALID: frozenset[str] = frozenset({"response", "response_params"})

    def _normalize_predict_kind(self, kind: str) -> str:
        """Validate and normalise the ``kind`` argument for :meth:`predict`.

        Maps deprecated Bambi aliases to their Bambi 0.14+ equivalents,
        rejects removed values, and raises for anything else — **before**
        the kind string is forwarded to ``bambi.Model.predict()``.

        Canonical mapping
        -----------------
        * ``"response"`` → unchanged (posterior predictive :math:`Y_{rep}`)
        * ``"response_params"`` → unchanged (posterior mean µ / p / κ)
        * ``"pps"`` → ``"response"`` + :class:`FutureWarning`
        * ``"mean"`` → ``"response_params"`` + :class:`FutureWarning`
        * ``"linear"`` → :class:`ValueError` (removed in Bambi 0.14+)
        * anything else → :class:`ValueError`

        Args:
            kind: Raw kind string supplied by the caller.

        Returns:
            Canonical kind string (``"response"`` or ``"response_params"``)
            accepted by Bambi 0.14+.

        Raises:
            ValueError: For unsupported or removed kind values.
        """
        if kind in self._KIND_REMOVED:
            raise ValueError(
                f"kind={kind!r} was removed in Bambi 0.14+. "
                f"Use 'response' (posterior predictive Y_rep) or "
                f"'response_params' (posterior mean parameter µ/p/κ)."
            )

        if kind in self._KIND_ALIASES:
            canonical = self._KIND_ALIASES[kind]
            warnings.warn(
                f"kind={kind!r} is deprecated; use {canonical!r} instead. "
                f"This alias will be removed in a future version of hbsaemp.",
                FutureWarning,
                # stacklevel 3: warn() ← _normalize_predict_kind() ← predict() ← caller
                stacklevel=3,
            )
            return canonical

        if kind not in self._KIND_VALID:
            raise ValueError(
                f"Unsupported kind={kind!r}. "
                f"Valid values: 'response' (posterior predictive Y_rep) or "
                f"'response_params' (posterior mean parameter µ/p/κ)."
            )

        return kind

    def _build_bambi_priors(self, priors: PriorDict | None) -> dict | None:
        """Convert user prior dict to Bambi ``Prior`` objects.

        Args:
            priors: Dict of the form
                ``{"x1": {"dist": "Normal", "mu": 0, "sigma": 1}}``,
                or ``None`` for Bambi auto-priors.

        Returns:
            Dict of ``{param: bmb.Prior(...)}`` ready for
            ``bmb.Model(..., priors=...)``, or ``None``.

        Raises:
            ImportError: If ``bambi`` is not installed.
        """
        if priors is None:
            return None
        try:
            import bambi as bmb
        except ImportError as exc:
            raise ImportError(
                "Prior conversion requires bambi>=0.14. "
                "Install with: pip install 'hbsaemp[bambi]'"
            ) from exc
        return {
            k: bmb.Prior(v["dist"], **{kk: vv for kk, vv in v.items() if kk != "dist"})
            for k, v in priors.items()
        }

    # ── Concrete fit() and predict() ─────────────────────────────────────────

    def fit(self) -> ModelResult:
        """Fit the HBSAE model via Bambi MCMC.

        Shared pipeline used by every family subclass:

        1. :meth:`_pre_fit_checks` — early guard hook (default no-op).
        2. Lazy import of ``bambi`` (centralised :class:`ImportError` handling).
        3. :meth:`_run_pipeline` — parse formula → validate data →
           preprocess (adds family-specific offset columns).
        4. :meth:`_build_formula_and_link` — subclass hook returns the
           Bambi formula (plain string or :class:`bambi.Formula`) and
           link spec (string or dict).
        5. :meth:`_build_bambi_priors` — convert user priors to Bambi
           :class:`bambi.Prior` objects.
        6. Build ``bambi.Model`` with family from :meth:`_bambi_family`.
        7. ``bambi.Model.fit()`` with
           ``idata_kwargs={"log_likelihood": True}`` (mandatory for LOO/WAIC).
        8. Build and store :class:`ModelResult` — ``family`` keeps the
           user-facing label (e.g. ``"lognormal"``) even when the Bambi
           backend family differs.

        Returns:
            The fitted :class:`ModelResult` (also stored on ``self._result``).

        Raises:
            ImportError: If ``bambi`` is not installed.
            DataValidationError: If the data fails validator checks.
        """
        # 1. Family-specific pre-flight checks (raise BEFORE bambi import).
        self._pre_fit_checks()

        # 2. Lazy import of bambi.  Centralised so subclass hooks never import it.
        try:
            import bambi as bmb
        except ImportError as exc:
            raise ImportError(
                f"{type(self).__name__}.fit() requires bambi>=0.14. "
                "Install with: pip install 'hbsaemp[bambi]'"
            ) from exc

        # 3. parse → validate → preprocess
        response, predictors, _, group_col, df_clean = self._run_pipeline(self._data)
        logger.info(
            "%s.fit(): response=%r, predictors=%r, group=%r, n=%d",
            type(self).__name__, response, predictors, group_col, len(self._data),
        )

        # 4. Family-specific Bambi formula + link
        bambi_formula, link = self._build_formula_and_link(bmb, response)

        # 5. Priors — workaround layer first, user priors last so they win.
        user_priors = self._build_bambi_priors(self._priors)
        workaround = self._workaround_priors(bmb)
        bambi_priors: dict[str, Any] | None = (
            {**workaround, **(user_priors or {})}
            if (workaround or user_priors)
            else None
        )

        # 6. Build Bambi model (use _bambi_family() so lognormal maps to gaussian).
        bmb_family = self._bambi_family()
        logger.debug(
            "Building bambi.Model (family=%r, link=%r)", bmb_family, link
        )
        bmodel = bmb.Model(
            bambi_formula,
            df_clean,
            family=bmb_family,
            link=link,
            priors=bambi_priors,
        )

        # 7. Sample
        sampler_kwargs = self._config.to_sampler_kwargs()
        logger.info(
            "Starting MCMC: draws=%d, tune=%d, chains=%d, cores=%d",
            sampler_kwargs["draws"], sampler_kwargs["tune"],
            sampler_kwargs["chains"], sampler_kwargs["cores"],
        )
        idata = bmodel.fit(
            **sampler_kwargs,
            idata_kwargs={"log_likelihood": True},
        )

        # 8. Store result — keep user-facing family label.
        self._result = ModelResult(
            backend_model=bmodel,
            idata=idata,
            formula=self._formula,
            family=self._family,
            data=df_clean,
            config=self._config,
            priors=self._priors,
            is_fitted=True,
            fitted_at=datetime.now(timezone.utc),
            extra={
                "response": response,
                "group": group_col,
                **self._extra_result_dict(),
            },
        )

        logger.info("%s.fit() complete.", type(self).__name__)
        return self._result

    def predict(
        self,
        new_data: pd.DataFrame | None = None,
        *,
        kind: str = "response",
        n_samples: int | None = None,
    ) -> np.ndarray:
        """Draw posterior samples.

        Supported ``kind`` values (Bambi 0.14+):

        * ``"response"`` *(default)* — posterior predictive samples
          :math:`Y_{rep}`; extracted from
          ``idata.posterior_predictive[response]``.
        * ``"response_params"`` — posterior of the mean/latent parameter
          (``"mu"`` for Gaussian / Beta / Lognormal, ``"p"`` for Binomial);
          extracted from ``idata.posterior[_MEAN_PARAM_KEY]``.

        Deprecated aliases (handled internally, **not** forwarded to Bambi):

        * ``"pps"`` → normalised to ``"response"``; triggers :class:`FutureWarning`.
        * ``"mean"`` → normalised to ``"response_params"``; triggers
          :class:`FutureWarning`.

        Removed in Bambi 0.14+ (raises :class:`ValueError` immediately):

        * ``"linear"``

        Args:
            new_data: Out-of-sample data. ``None`` uses training data.
                For Binomial, *new_data* must include ``trials_col``.
            kind: Prediction kind (see above). Default ``"response"``.
            n_samples: Number of posterior draws to return. ``None`` = all.

        Returns:
            Array of shape ``(total_draws, n_obs)`` where
            ``total_draws = chains × draws``.  For :class:`LognormalModel`
            the values are on the **log scale** — apply :func:`numpy.exp`
            to recover the original-scale response.

        Raises:
            ModelNotFittedError: If :meth:`fit` has not been called.
            ValueError: For unsupported or removed *kind* values.
        """
        result = self.result  # raises ModelNotFittedError if not fitted
        response: str = result.extra["response"]
        bmodel = result.backend_model
        idata = result.idata

        # Normalise before calling Bambi: maps deprecated aliases, rejects
        # removed / unknown kinds. Only "response" or "response_params" reach Bambi.
        kind = self._normalize_predict_kind(kind)

        logger.debug(
            "%s.predict(): kind=%r, new_data=%s",
            type(self).__name__,
            kind,
            "None (in-sample)" if new_data is None else f"shape={new_data.shape}",
        )

        # Preprocess new_data so any required offset columns are present.
        # In-sample falls back to the stored training data instead of None:
        # Bambi 0.14+ raises a TypeError ("'str' cannot be interpreted as an
        # integer") on distributional models with offset() when data=None,
        # because the offset column lookup expects an explicit DataFrame.
        processed = (
            self._preprocess_new_data(new_data)
            if new_data is not None
            else result.data
        )
        bmodel.predict(idata, data=processed, kind=kind, inplace=True)

        # ── Extract draws ────────────────────────────────────────────────────
        # kind is always canonical after _normalize_predict_kind():
        #   "response"        → idata.posterior_predictive[_response_pp_key(response)]
        #   "response_params" → idata.posterior[_MEAN_PARAM_KEY]
        if kind == "response":
            draws_da = idata.posterior_predictive[self._response_pp_key(response)]
        else:  # "response_params"
            draws_da = idata.posterior[self._MEAN_PARAM_KEY]

        # stack() combines named dims — safe against axis-order changes in ArviZ.
        # Result dims: (obs_dim, "sample"), shape (n_obs, n_samples).
        # .T → (n_samples, n_obs) matching the documented return shape.
        flat: np.ndarray = draws_da.stack(sample=("chain", "draw")).values.T

        if n_samples is not None:
            flat = flat[:n_samples]

        logger.debug(
            "%s.predict(): output shape=%s", type(self).__name__, flat.shape
        )
        return flat

    # ── Misc ─────────────────────────────────────────────────────────────────

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
