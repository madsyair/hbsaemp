"""BaseModel and ModelResult — concrete fit/predict pipeline shared by all families.

The MCMC pipeline (parse formula -> validate -> preprocess -> Bambi -> store
result) lives in `BaseModel.fit()`; subclasses supply only small hooks
(`_build_formula_and_link`, `_workaround_priors`, etc.).
"""

from __future__ import annotations

import abc
import warnings
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from hbsaemp._exceptions import ModelNotFittedError
from hbsaemp._logging import get_logger
from hbsaemp._types import FamilyLiteral, FormulaStr, PriorDict
from hbsaemp.models._family_spec import FAMILY_SPECS, FamilySpec

if TYPE_CHECKING:
    from hbsaemp.models._config import ModelConfig

logger = get_logger(__name__)

__all__: list[str] = ["BaseModel", "ModelResult"]


# ModelResult

@dataclass
class ModelResult:
    """Container returned by `BaseModel.fit()`.

    Attributes:
        backend_model: Fitted `bambi.Model`.
        idata: Posterior samples (`arviz.InferenceData`).
        formula: Original formula string.
        family: User-facing family label.
        data: Preprocessed DataFrame actually fit.
        config: `ModelConfig` used for sampling.
        priors: User prior dict, or `None` if Bambi auto-priors were used.
        is_fitted: True after `fit()` completes.
        fitted_at: UTC datetime when `fit()` completed.
        extra: Family-specific metadata (response col, group col, etc.).
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
        """One-screen text summary of the fitted result."""
        if not self.is_fitted:
            return f"ModelResult [{self.family}] — not fitted yet."
        return (
            f"ModelResult [{self.family}]\n"
            f"  Formula : {self.formula}\n"
            f"  n       : {len(self.data)}\n"
            f"  Fitted  : {self.fitted_at}\n"
            f"  Backend : {type(self.backend_model).__name__ if self.backend_model else 'stub'}"
        )


# BaseModel

class BaseModel(abc.ABC):
    """Abstract base for all HBSAE distribution models.

    Subclasses inherit the concrete `fit()` / `predict()` and override only
    family-specific *behavior* hooks, each documented on its own method. The
    single required override is `_build_formula_and_link`. All family metadata
    (mean parameter, links, pipeline fields, backend family) is read from
    `FAMILY_SPECS[self._family]` via `self._spec` — subclasses never re-declare
    it.

    Args:
        formula: R/lme4-style formula, e.g. `"y ~ x1 + (1|group)"`.
        family: User-facing family label.
        data: Input DataFrame.
        config: MCMC configuration.
        priors: Optional prior dict; `None` means Bambi auto-priors.
        group: Grouping column for random effects.
        handle_missing: Missing data strategy (`"deleted"` only in v1).
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

    # Read-only properties

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

    # Family metadata (single source: FAMILY_SPECS[self._family])

    @property
    def _spec(self) -> FamilySpec:
        """Immutable metadata for `self._family` (from `FAMILY_SPECS`)."""
        try:
            return FAMILY_SPECS[self._family]
        except KeyError as exc:
            raise ValueError(
                f"No FamilySpec registered for family={self._family!r}. "
                f"Known families: {sorted(FAMILY_SPECS)}."
            ) from exc

    @property
    def _mean_param_key(self) -> str:
        """Posterior key for the mean/probability parameter (`"mu"` / `"p"`).

        Read by `predict(kind="response_params")`.
        """
        return self._spec.mean_param_key

    @property
    def _default_link(self) -> str:
        """Default link for the family mean parameter when no `link=` given."""
        return self._spec.default_link

    def _validate_link(self) -> None:
        """Reject `self._link` if not in the family's supported links.

        Single source of truth: `FAMILY_SPECS[self._family].supported_links`.
        Called from `_pre_fit_checks()`, so the check runs before the bambi
        import on every `fit()`.
        """
        link = getattr(self, "_link", None)
        if link not in self._spec.supported_links:
            raise ValueError(
                f"link={link!r} not supported for family={self._family!r}. "
                f"Valid links: {sorted(self._spec.supported_links)}"
            )

    # Abstract hooks (subclass must implement)

    def _extra_pipeline_kwargs(self) -> dict[str, Any]:
        """Family-specific kwargs unpacked into validator and preprocessor.

        Reads `FAMILY_SPECS[self._family].pipeline_fields` and returns
        `{name: self._<name>}` for each declared field. A field declared in
        the spec but missing on the instance raises immediately — the
        single-source contract must not drift silently. Override only when
        the lookup logic itself must differ.
        """
        missing = [
            name for name in self._spec.pipeline_fields
            if not hasattr(self, f"_{name}")
        ]
        if missing:
            raise AttributeError(
                f"{type(self).__name__} is missing attribute(s) "
                f"{['_' + m for m in missing]} declared in "
                f"FAMILY_SPECS[{self._family!r}].pipeline_fields — the "
                f"subclass constructor must set them."
            )
        return {name: getattr(self, f"_{name}") for name in self._spec.pipeline_fields}

    @abc.abstractmethod
    def _build_formula_and_link(
        self,
        bmb_module: Any,
        response: str,
    ) -> tuple[Any, Any]:
        """Build the Bambi formula and link spec for `bmb.Model`.

        Args:
            bmb_module: Imported `bambi` module (subclass doesn't import it
                directly). Use `bmb_module.Formula` for distributional models.
            response: Response column name parsed from `self._formula`.

        Returns:
            `(formula, link)`. `formula` is a string or `bambi.Formula`;
            `link` is a string (single link) or a dict (distributional —
            must include all parameter keys).
        """

    def _extra_result_dict(self) -> dict[str, Any]:
        """Family-specific keys merged into `ModelResult.extra`.

        `fit()` always adds `"response"` and `"group"`. Default delegates
        to `_extra_pipeline_kwargs()`; override only when result keys must
        differ from pipeline keys.
        """
        return self._extra_pipeline_kwargs()

    # Concrete hooks (overridable)

    def _pre_fit_checks(self) -> None:
        """Early guards run before the `bambi` import.

        Base implementation validates `self._link` against the family spec.
        Subclasses that need extra fail-fast checks (e.g. `BinomialModel`
        requires `trials_col`; `BetaModel` guards `squeeze`) override this and
        call `super()._pre_fit_checks()` to keep link validation.
        """
        self._validate_link()

    def _bambi_family(self) -> str:
        """Family string for ``bambi.Model(family=...)``.

        Read from `FAMILY_SPECS`.
        """
        return self._spec.bambi_family

    def _response_pp_key(self, response: str) -> str:
        """Key under which `posterior_predictive` stores the response.

        Defaults to *response*; `BinomialModel` overrides (Bambi wraps it as
        `p(y, trials_col)`).
        """
        return response

    def _workaround_priors(self, bmb_module: Any) -> dict[str, Any]:
        """Auto-injected priors that work around Bambi backend bugs.

        Default returns `{}`. User priors (`self._priors`) take precedence —
        `fit()` merges with the workaround dict as the lower-priority base.
        """
        return {}

    @property
    def _preproc_family(self) -> str:
        """Family for validator and preprocessor.

        Read from `FAMILY_SPECS`.
        """
        return self._spec.preproc_family

    # Hook helpers (concrete utilities for subclass hooks)

    def _build_distributional_formula(
        self,
        bmb_module: Any,
        main_formula: str,
        *,
        param: str,
        offset_col: str | None,
        mu_link: str,
        mu_key: str = "mu",
    ) -> tuple[Any, Any]:
        """Build a Bambi distributional formula for Fay-Herriot offset models.

        Shared between Gaussian and Beta. When `offset_col` is
        None, returns the plain formula + single link.

        The secondary sub-formula uses `"1 + offset(...)"` (not `"0 + ..."`)
        as a Bambi 0.18+ workaround — the `0+` form leaves an empty common
        design matrix and crashes `DistributionalComponent.predict()`. The
        intercept is clamped near zero by `_pin_intercept_prior`.

        Args:
            bmb_module: Imported `bambi` module.
            main_formula: Response-side formula, e.g. `"y ~ x1 + x2"`.
            param: Distributional parameter name (`"sigma"`, `"kappa"`).
            offset_col: Pre-computed offset column (e.g. `"log_sqrt_D"`),
                or None for plain regression.
            mu_link: Link function for the mean parameter.
            mu_key: Mean-parameter key in the link dict (`"mu"` or `"p"`).

        Returns:
            `(formula, link)` ready for `bmb.Model`.
        """
        if offset_col is not None:
            formula = bmb_module.Formula(
                main_formula, f"{param} ~ 1 + offset({offset_col})"
            )
            # Both keys required — partial dict raises KeyError in Bambi backend.
            link: Any = {mu_key: mu_link, param: "log"}
        else:
            formula = main_formula
            link = mu_link
        return formula, link

    def _pin_intercept_prior(
        self,
        bmb_module: Any,
        *,
        param: str,
        active: bool,
    ) -> dict[str, Any]:
        """Tight `Normal(0, 1e-3)` prior on `{param}_Intercept` when *active*.

        Used by the distributional offset workaround to keep `param ≈ offset`
        (see `_build_distributional_formula`). Returns `{}` when not active.
        """
        if active:
            return {
                f"{param}_Intercept": bmb_module.Prior("Normal", mu=0.0, sigma=1e-3),
            }
        return {}

    # Pipeline helpers (concrete)

    def _run_pipeline(
        self,
        data: pd.DataFrame,
    ) -> tuple[str, list[str], list[str], str | None, pd.DataFrame]:
        """Parse formula, validate, preprocess; returns the cleaned frame.

        Returns `(response, predictors, random_groups, group_col, df_clean)`
        where `df_clean` is the preprocessed copy ready for `bambi.Model`.
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

        Adds required offset columns (`log_phi`, `log_sqrt_D`) so Bambi's
        distributional formula can evaluate them. Validation is skipped.

        Raises:
            ModelNotFittedError: If `fit()` has not been called.
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

    # predict() kind normalisation

    # Map of deprecated kind aliases → canonical Bambi 0.18+ kind.
    # Scheduled for removal in hbsaemp 2.0.0 — see `_ALIAS_REMOVAL_VERSION`.
    _KIND_ALIASES: dict[str, str] = {"pps": "response", "mean": "response_params"}
    #: Release that drops `_KIND_ALIASES`. A deprecation without a version is a
    #: promise nobody can act on, so name it here and in the warning message.
    _ALIAS_REMOVAL_VERSION: str = "2.0.0"
    # Kinds removed in Bambi 0.18+ — raise immediately before touching idata.
    _KIND_REMOVED: frozenset[str] = frozenset({"linear"})
    # Only these two canonical values are forwarded to ``bambi.Model.predict()``.
    _KIND_VALID: frozenset[str] = frozenset({"response", "response_params"})

    def _normalize_predict_kind(self, kind: str) -> str:
        """Validate and normalise the `kind` arg before forwarding to Bambi.

        Mapping:

        * `"response"` / `"response_params"` — passthrough.
        * `"pps"` -> `"response"` (FutureWarning; removed in 2.0.0).
        * `"mean"` -> `"response_params"` (FutureWarning; removed in 2.0.0).
        * `"linear"` — removed in Bambi 0.18+ -> ValueError.
        * other — ValueError.
        """
        if kind in self._KIND_REMOVED:
            raise ValueError(
                f"kind={kind!r} was removed in Bambi 0.18+. "
                f"Use 'response' (posterior predictive Y_rep) or "
                f"'response_params' (posterior mean parameter µ/p/κ)."
            )

        if kind in self._KIND_ALIASES:
            canonical = self._KIND_ALIASES[kind]
            warnings.warn(
                f"kind={kind!r} is deprecated; use {canonical!r} instead. "
                f"This alias will be removed in hbsaemp "
                f"{self._ALIAS_REMOVAL_VERSION}.",
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
        """Convert user prior dict / `Prior` objects to `bambi.Prior`.

        Accepts either `Prior` instances (validated at construction) or plain
        dicts like `{"dist": "Normal", "mu": 0, "sigma": 1}` (validated here
        via `Prior.from_dict`). Returns `None` for Bambi's auto-priors.

        Raises:
            ImportError: If `bambi` is not installed.
            PriorSpecError: If any dict entry lacks a `"dist"` key.
        """
        if priors is None:
            return None
        bmb = self._import_bambi()

        from hbsaemp.models._prior import Prior

        result: dict[str, Any] = {}
        for k, v in priors.items():
            if isinstance(v, Prior):
                result[k] = v.to_bambi(bmb)
            else:
                # Legacy dict format — coerce via Prior.from_dict (validates here).
                result[k] = Prior.from_dict(v, param=k).to_bambi(bmb)
        return result

    # Pre-fit inspection (Bambi-free)

    @property
    def response_name(self) -> str:
        """Response column parsed from the formula — available before `fit()`.

        Lets callers label plots and tables without re-parsing the formula or
        waiting for `ModelResult.extra["response"]`.
        """
        from hbsaemp.utils._formula import parse_formula

        return str(parse_formula(self._formula)["response"])

    def check_data(self) -> pd.DataFrame:
        """Run the pre-flight checks and data pipeline without importing bambi.

        Same work `fit()` does before it touches Bambi: family pre-flight
        checks, then parse -> validate -> preprocess. Callers get the frame
        `fit()` would actually use, so missing-row counts and family-domain
        violations surface *before* paying for MCMC.

        Returns:
            The preprocessed copy of the training data (offset columns added,
            rows with missing values dropped per `handle_missing`).

        Raises:
            DataValidationError: If the data fails validator checks.
            ValueError: From `_pre_fit_checks` (unsupported link, missing
                `trials`, invalid `squeeze`, ...).
        """
        self._pre_fit_checks()
        _, _, _, _, df_clean = self._run_pipeline(self._data)
        return df_clean

    # Build / fit seam

    def _import_bambi(self) -> Any:
        """Lazy-import bambi with the package's standard ImportError message.

        The package's single Bambi import site: subclass hooks receive the
        module as an argument and never import it themselves, and
        `_build_bambi_priors` routes through here too (locked by
        `test_bambi_handoff_only_in_base`).

        Callers must run `_pre_fit_checks()` *before* this, so a bad link or a
        missing `trials` column raises its own `ValueError` rather than an
        `ImportError` on a machine without bambi.
        """
        try:
            import bambi as bmb
        except ImportError as exc:
            raise ImportError(
                f"{type(self).__name__} requires bambi>=0.18. "
                "Install with: pip install 'hbsaemp[bambi]'"
            ) from exc
        return bmb

    def _build_backend(
        self, bmb_module: Any
    ) -> tuple[Any, str, str | None, pd.DataFrame]:
        """Everything `fit()` does *except* sampling; returns a built model.

        pre_fit_checks -> run_pipeline (parse + validate + preprocess) ->
        build formula+link -> merge priors -> `bmb.Model(...)` -> `.build()`.

        The returned model is built but **unfitted**: no sampling has run and
        `self._result` is untouched. Mirrors Bambi's own `Model.build()` /
        `Model.fit()` split, which lets `prior_predictive_idata()` reuse the
        entire pipeline instead of duplicating it.

        Args:
            bmb_module: Imported `bambi` module (from `_import_bambi`).

        Returns:
            `(bmodel, response, group_col, df_clean)`.

        Raises:
            DataValidationError: If the data fails validator checks.
        """
        # 1. Family-specific pre-flight checks (cheap, fail before any building).
        self._pre_fit_checks()

        # 2. parse → validate → preprocess
        response, predictors, _, group_col, df_clean = self._run_pipeline(self._data)
        logger.info(
            "%s: response=%r, predictors=%r, group=%r, n=%d",
            type(self).__name__, response, predictors, group_col, len(self._data),
        )

        # 3. Family-specific Bambi formula + link
        bambi_formula, link = self._build_formula_and_link(bmb_module, response)

        # 4. Priors — workaround layer first, user priors last so they win.
        user_priors = self._build_bambi_priors(self._priors)
        workaround = self._workaround_priors(bmb_module)
        bambi_priors: dict[str, Any] | None = (
            {**workaround, **(user_priors or {})}
            if (workaround or user_priors)
            else None
        )

        # 5. Build Bambi model using the family metadata from FAMILY_SPECS.
        bmb_family = self._bambi_family()
        logger.debug(
            "Building bambi.Model (family=%r, link=%r)", bmb_family, link
        )
        bmodel = bmb_module.Model(
            bambi_formula,
            df_clean,
            family=bmb_family,
            link=link,
            priors=bambi_priors,
        )

        # 6. Construct the PyMC model. Idempotent — Bambi's fit() calls it
        # behind a `built` flag — but doing it here makes "built" a
        # postcondition rather than an accident, which prior_predictive()
        # depends on (it starts with its own _check_built()).
        bmodel.build()

        return bmodel, response, group_col, df_clean

    def prior_predictive_idata(
        self,
        *,
        draws: int = 500,
        var_names: list[str] | None = None,
        random_seed: int | None = None,
    ) -> Any:
        """Sample the prior predictive distribution — no MCMC, no fitting.

        Runs the full build pipeline (validate -> preprocess -> offset columns
        -> workaround priors -> user priors -> bambi) and then samples from the
        prior only. The model stays unfitted: `self._result` is not written and
        `is_fitted` stays `False`.

        Counterpart of `predictive_idata()` on the posterior side — both return
        a fresh `InferenceData` and mutate nothing.

        Args:
            draws: Number of prior draws. Default 500 (Bambi's own default).
            var_names: Restrict the sampled variables; `None` samples all.
            random_seed: Seed for reproducibility. Falls back to
                `config.random_seed` when `None`.

        Returns:
            `arviz.InferenceData` with `prior`, `prior_predictive` and
            `observed_data` groups.

        Raises:
            ImportError: If `bambi` is not installed.
            DataValidationError: If the data fails validator checks.
        """
        # Cheap family checks first, so bambi's ImportError never masks a bad
        # link or a missing `trials` column. `_build_backend` repeats them.
        self._pre_fit_checks()

        bmb = self._import_bambi()
        bmodel, _response, _group_col, _df_clean = self._build_backend(bmb)

        seed = random_seed if random_seed is not None else self._config.random_seed
        logger.info(
            "%s.prior_predictive_idata(): draws=%d, seed=%r",
            type(self).__name__, draws, seed,
        )
        return bmodel.prior_predictive(
            draws=draws, var_names=var_names, random_seed=seed
        )

    # Concrete fit() and predict()

    def fit(self) -> ModelResult:
        """Fit the model via Bambi MCMC and return a `ModelResult`.

        Pipeline (shared across all families): `_build_backend()` — pre_fit
        checks, parse + validate + preprocess, formula+link, prior merge,
        `bmb.Model` — then sampling.

        Log-likelihood is computed post-sampling via
        `bmodel.compute_log_likelihood(idata)` (PyMC 6.0 / Bambi 0.18 pattern) —
        required for LOO downstream.

        Raises:
            ImportError: If `bambi` is not installed.
            DataValidationError: If the data fails validator checks.
        """
        # 1. Family-specific pre-flight checks, BEFORE the bambi import, so a bad
        # link or a missing `trials` column reports its own error even when
        # bambi is absent. `_build_backend` runs them again (idempotent).
        self._pre_fit_checks()

        # 2-6. Lazy bambi import, then everything up to (not including) sampling.
        bmb = self._import_bambi()
        bmodel, response, group_col, df_clean = self._build_backend(bmb)

        # 7. Sample
        sampler_kwargs = self._config.to_sampler_kwargs()
        logger.info(
            "Starting MCMC: draws=%d, tune=%d, chains=%d, cores=%d",
            sampler_kwargs["draws"], sampler_kwargs["tune"],
            sampler_kwargs["chains"], sampler_kwargs["cores"],
        )
        idata = bmodel.fit(**sampler_kwargs)

        # 7b. Compute log-likelihood post-sampling. PyMC 6.0 / Bambi 0.18
        # deprecated requesting it inline via fit(idata_kwargs={"log_likelihood": True}).
        # data=None reuses Bambi's internal training frame (df_clean, which already
        # carries any offset columns). LOO downstream requires this group; let a
        # failure here propagate rather than yield an idata that breaks silently later.
        bmodel.compute_log_likelihood(idata)

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
            fitted_at=datetime.now(UTC),
            extra={
                "response": response,
                "group": group_col,
                **self._extra_result_dict(),
            },
        )

        logger.info("%s.fit() complete.", type(self).__name__)
        return self._result

    def _predict_idata(
        self, new_data: pd.DataFrame | None = None, *, kind: str
    ) -> Any:
        """Return a fresh `InferenceData` with the predicted group populated.

        `inplace=False` → Bambi returns a NEW `InferenceData` (the original
        posterior merged with the requested prediction group). `result.idata`
        holds only the sampled parameter posteriors — the response params
        (`mu`/`p`) are computed on demand here, never stored at `fit()` time —
        so it is never overwritten. Callers read draws from the returned
        object, never from `result.idata`.

        `kind` must already be canonical (`"response"` / `"response_params"`);
        `predict()` normalises before calling. The public `predictive_idata()`
        wraps this for the `"response"` case so `compare_models()` and advanced
        users get mutation-free PPC without reaching into a private method.
        """
        result = self.result  # raises ModelNotFittedError if not fitted
        bmodel = result.backend_model

        # In-sample falls back to the stored training data instead of None:
        # Bambi 0.18+ raises a TypeError ("'str' cannot be interpreted as an
        # integer") on distributional models with offset() when data=None,
        # because the offset column lookup expects an explicit DataFrame.
        processed = (
            self._preprocess_new_data(new_data)
            if new_data is not None
            else result.data
        )
        return bmodel.predict(result.idata, data=processed, kind=kind, inplace=False)

    def predict(
        self,
        new_data: pd.DataFrame | None = None,
        *,
        kind: str = "response",
        n_samples: int | None = None,
    ) -> np.ndarray:
        """Draw posterior samples; returns `(total_draws, n_obs)`.

        Args:
            new_data: Out-of-sample data. `None` uses training data.
                For Binomial, must include `trials_col`.
            kind:
                - `"response"` (default): posterior predictive Y_rep, from
                  `idata.posterior_predictive[response]`.
                - `"response_params"`: posterior of the mean parameter
                  (`mu` or `p`), from `idata.posterior[self._mean_param_key]`.
                - `"pps"` / `"mean"`: deprecated aliases (FutureWarning).
                - `"linear"`: removed in Bambi 0.18+ -> ValueError.
            n_samples: Number of posterior draws to return; `None` = all.

        Raises:
            ModelNotFittedError: If `fit()` has not been called.
            ValueError: For unsupported or removed *kind* values.
        """
        result = self.result  # raises ModelNotFittedError if not fitted
        response: str = result.extra["response"]

        # Normalise before calling Bambi: maps deprecated aliases, rejects
        # removed / unknown kinds. Only "response" or "response_params" reach Bambi.
        kind = self._normalize_predict_kind(kind)

        logger.debug(
            "%s.predict(): kind=%r, new_data=%s",
            type(self).__name__,
            kind,
            "None (in-sample)" if new_data is None else f"shape={new_data.shape}",
        )

        # inplace=False → Bambi returns a NEW idata; result.idata stays intact
        # (it holds only the sampled parameter posteriors; mu/p are computed on
        # demand, never stored at fit() time).
        pred_idata = self._predict_idata(new_data, kind=kind)

        # Extract draws. kind is always canonical after _normalize_predict_kind():
        #   "response"        → pred_idata.posterior_predictive[_response_pp_key(response)]
        #   "response_params" → pred_idata.posterior[_mean_param_key]
        if kind == "response":
            draws_da = pred_idata.posterior_predictive[self._response_pp_key(response)]
        else:  # "response_params"
            draws_da = pred_idata.posterior[self._mean_param_key]

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

    def predictive_idata(self, new_data: pd.DataFrame | None = None) -> Any:
        """Return a fresh idata with ``posterior_predictive`` populated.

        Does NOT mutate ``result.idata``. Public entry point for
        posterior-predictive work: used internally by ``compare_models()`` for
        its pp-check plot, and available to advanced users who want a custom PPC
        without corrupting the stored idata. Wraps the private ``_predict_idata``
        (``inplace=False``). On ArviZ 1.1 the returned object is a DataTree —
        access groups as attributes (``idata.posterior_predictive``).

        Args:
            new_data: Out-of-sample data. ``None`` uses the training data.

        Raises:
            ModelNotFittedError: If ``fit()`` has not been called.
        """
        return self._predict_idata(new_data, kind="response")

    # Misc

    def summary(self) -> str:
        """Human-readable model summary.

        Before `fit()`, returns a ``[not fitted]`` placeholder; afterwards
        delegates to `ModelResult.summary()`.
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
