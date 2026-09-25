"""Single source of truth for family metadata *and* behavior.

`FAMILY_SPECS` is keyed by the user-facing family label (`"gaussian"` etc.);
every component that needs to know anything about a family reads it here —
metadata (backend family, mean param, links, pipeline fields, kwarg mapping)
*and* behavior (response-domain check, offset preprocessing, and the formula
addition template). Behavior is stored as **module-level functions** (never
subclass method references), so this module imports nothing from `hbsaemp`
except the exception hierarchy (`_exceptions`) and the logger (`_logging`) —
neither has a back-dependency, so the module stays cycle-free.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from hbsaemp._exceptions import DataValidationError, ModelRegistryError
from hbsaemp._logging import get_logger

logger = get_logger(__name__)

__all__: list[str] = [
    "FamilySpec",
    "FixedParam",
    "FAMILY_SPECS",
    "USER_FIXED_COL",
    "apply_link",
    "list_families",
    "get_family_spec",
    "pin_source_columns",
]

#: Raise `DataValidationError` when the response (or its auxiliary columns)
#: violates the family's domain; return `None` on success. `ctx` carries the
#: family's pipeline kwargs (`n_col`, `deff_col`, `sampling_var_col`,
#: `trials_col`, `squeeze`).
ResponseCheck = Callable[[pd.DataFrame, str, dict], None]
#: Return a transformed copy of the frame with any offset columns added
#: (e.g. `log_phi`, `log_sqrt_D`). `ctx` carries the same pipeline kwargs.
Preprocess = Callable[[pd.DataFrame, str, dict], pd.DataFrame]

#: Offset columns written by the preprocess functions below and declared as
#: `FixedParam.offset_col` — one name each, so spec and transform cannot drift.
_GAUSSIAN_OFFSET = "log_sqrt_D"
_BETA_OFFSET = "log_phi"

#: Column a user-supplied pin is materialised into, by parameter name.
#: The counterpart of hbsaems' `.hbsaems_<par>_fixed`; no leading dot, because
#: `formulae` needs the offset term to be a valid Python identifier.
USER_FIXED_COL = "hbsaemp_{param}_fixed"


def pin_source_columns(fixed_params: dict[str, str | float] | None) -> list[str]:
    """Data columns a caller-supplied pin names; a scalar pin names none.

    The one place that rule is written. Every layer handling caller pins needs
    it — the validator checks the columns exist and are numeric, the
    preprocessor drops rows where they are NaN, and `update_model()` carries
    them onto replacement data — and a second copy of the rule is exactly how
    a pin column ends up treated differently from a survey-design column.
    """
    return [src for src in (fixed_params or {}).values() if isinstance(src, str)]


def apply_link(values: np.ndarray, link: str) -> np.ndarray:
    """Map parameter values onto the scale the offset term is read on.

    A pin says "this parameter equals these values". Bambi adds `offset(col)`
    to the parameter's *linear predictor*, so the column must hold the values
    already transformed by the parameter's link — `log` for the positive-scale
    parameters (`sigma`, `kappa`) that SAE pins in practice.

    Raises:
        DataValidationError: Values outside the link's domain.
        ModelRegistryError: Link has no transform defined here.
    """
    if link == "identity":
        return values
    if link == "log":
        if np.any(values <= 0):
            n_bad = int(np.sum(values <= 0))
            raise DataValidationError(
                f"A parameter pinned through a 'log' link must be positive. "
                f"Found {n_bad} non-positive value(s).",
                context={"n_nonpositive": n_bad},
            )
        return np.log(values)
    raise ModelRegistryError(
        f"No transform defined for link {link!r}. Known: 'identity', 'log'."
    )


@dataclass(frozen=True)
class FixedParam:
    """A distributional parameter computed from data instead of sampled.

    The Bayesian default is that every parameter is stochastic. Small-area
    models break that on purpose: the Beta precision is ``phi = n/deff - 1``
    and the Gaussian Fay-Herriot scale is ``sqrt(D)``, both known from the
    survey design. Such a parameter is pinned by feeding its pre-computed
    column in as an ``offset()`` on the parameter's own sub-formula, then
    clamping that sub-formula's intercept to ~0 so the offset is all that
    remains. It is the Bambi counterpart of a WinBUGS logical node.

    Attributes:
        param: Distributional parameter name (`"sigma"`, `"kappa"`).
        link: Link applied to *param*; the offset column is stored on that
            scale (`preprocess` writes `log(...)` for a `"log"` link).
        offset_col: Column `preprocess` adds, carrying the pinned values.
        source_fields: Pipeline fields (`self._<name>`) naming the data
            columns *offset_col* is derived from. All must be supplied for
            the pin to apply; an empty tuple means it needs no design column
            and therefore always applies.
    """

    param: str
    link: str
    offset_col: str
    source_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class FamilySpec:
    """Immutable metadata and behavior for one HBSAE family.

    Attributes:
        bambi_family: Family string passed to `bambi.Model(family=...)`.
        mean_param_key: Name of the mean parameter in `idata.posterior`:
            `"mu"` for Gaussian and Beta, `"p"` for Binomial.
        default_link: Link for the mean parameter when no `link=` is given.
        supported_links: Valid link names.
        pipeline_fields: Names of the family's own fields, passed to the
            validator and the preprocessor.
        user_params: Maps each pipeline field to the keyword argument of
            `create_model()` that sets it.
        response_check: Domain check run by `DataValidator.validate()`, or
            `None`. Raises `DataValidationError` on failure.
        preprocess: Offset or transformation step run by
            `DataPreprocessor.process()`, or `None`.
        addition_template: `str.format` template for the formula left-hand
            side when the family wraps the response, e.g. Binomial
            `"p({response}, {trials_col})"`; `None` for a bare response. It may
            use `response` and any pipeline field by name.
        addition_field: The pipeline field that `addition_var` fills
            (`"trials_col"` for Binomial). Set together with
            `addition_template`; `None` if the family has no addition term.
        fixed_params: Distributional parameters computed from the survey
            design instead of sampled. Empty when every parameter is sampled.
        required_params: Keyword arguments `create_model()` requires, e.g.
            `trials` for Binomial. Arguments required together are defined by
            `FixedParam.source_fields` instead.
    """

    bambi_family: str
    mean_param_key: str
    default_link: str
    supported_links: frozenset[str]
    pipeline_fields: tuple[str, ...]
    user_params: dict[str, str] = field(default_factory=dict)
    # Behavior lives here too — read by the data layer and formula builders.
    response_check: ResponseCheck | None = None
    preprocess: Preprocess | None = None
    addition_template: str | None = None
    addition_field: str | None = None
    fixed_params: tuple[FixedParam, ...] = ()
    required_params: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject a spec whose wiring cannot work, at import time.

        Both rules are things the machinery dereferences later, where the
        failure would be a bare `KeyError` far from its cause:

        * every `FixedParam.source_fields` name is looked up in `user_params`
          by the factory's arity and double-pin checks;
        * `addition_template` and `addition_field` are one mechanism — a
          template with no field has no `addition_var` to fill it, and a field
          with no template is never read.
        """
        for fp in self.fixed_params:
            unknown = [f for f in fp.source_fields if f not in self.user_params]
            if unknown:
                raise ModelRegistryError(
                    f"FixedParam({fp.param!r}) names source field(s) {unknown} "
                    f"that are not in user_params {sorted(self.user_params)}."
                )
        if (self.addition_template is None) != (self.addition_field is None):
            raise ModelRegistryError(
                "addition_template and addition_field must be set together; "
                f"got template={self.addition_template!r}, "
                f"field={self.addition_field!r}."
            )
        if self.addition_field is not None and (
            self.addition_field not in self.pipeline_fields
        ):
            raise ModelRegistryError(
                f"addition_field={self.addition_field!r} is not one of "
                f"pipeline_fields {list(self.pipeline_fields)}."
            )

    @property
    def pinnable_params(self) -> dict[str, str]:
        """Map each parameter this family can pin to that parameter's link.

        A parameter is pinnable only if the family declares a `FixedParam`
        for it. Binomial declares none, so it accepts no `fixed_params=`.
        """
        return {fp.param: fp.link for fp in self.fixed_params}

    @property
    def offset_source_fields(self) -> tuple[str, ...]:
        """Every design column the fixed parameters are computed from.

        `update_model()` copies these columns over when replacement data
        lacks them, as hbsaems does with its `.hbsaems_<par>_fixed` columns.
        """
        return tuple(
            name for fp in self.fixed_params for name in fp.source_fields
        )


# Family behavior — module-level functions (cycle-free: only numpy/pandas/
# _exceptions/_logging). Each reads the family's pipeline kwargs from `ctx`.

def _check_gaussian(df: pd.DataFrame, response: str, ctx: dict) -> None:
    """Reject non-positive sampling variances for the Gaussian Fay-Herriot offset."""
    sampling_var_col = ctx.get("sampling_var_col")
    if sampling_var_col is not None:
        D = df[sampling_var_col].dropna()
        n_bad = int((D <= 0).sum())
        if n_bad:
            raise DataValidationError(
                f"Gaussian FH: {sampling_var_col!r} must be positive. "
                f"Found {n_bad} non-positive value(s).",
                column=sampling_var_col,
                context={"n_nonpositive": n_bad},
            )


def _preprocess_gaussian(df: pd.DataFrame, response: str, ctx: dict) -> pd.DataFrame:
    """Add ``log_sqrt_D = 0.5 * log(D)`` for Gaussian Fay-Herriot."""
    sampling_var_col = ctx.get("sampling_var_col")
    if sampling_var_col is not None:
        D = df[sampling_var_col].to_numpy(dtype=float)
        df[_GAUSSIAN_OFFSET] = 0.5 * np.log(D)
        logger.debug(
            "Gaussian FH: log_sqrt_D in [%.3f, %.3f].",
            df[_GAUSSIAN_OFFSET].min(), df[_GAUSSIAN_OFFSET].max(),
        )
    return df


def _check_beta(df: pd.DataFrame, response: str, ctx: dict) -> None:
    """Validate the Beta response domain and the precision phi = n/deff - 1.

    With ``squeeze=True`` the response domain is relaxed to closed ``[0, 1]``
    (the Smithson-Verkuilen transform will map boundary values into the open
    interval before fitting); otherwise it must be strictly open ``(0, 1)``.
    """
    n_col = ctx.get("n_col")
    deff_col = ctx.get("deff_col")
    squeeze = ctx.get("squeeze", False)
    y = df[response].dropna()
    if squeeze:
        # Smithson-Verkuilen will map [0,1] → (0,1); allow closed interval.
        n_bad = int(((y < 0) | (y > 1)).sum())
        domain_msg = "[0, 1]"
    else:
        # No squeeze: Bambi Beta requires strictly open (0, 1).
        n_bad = int(((y <= 0) | (y >= 1)).sum())
        domain_msg = "(0, 1)"
    if n_bad:
        raise DataValidationError(
            f"Beta family requires response in {domain_msg}. "
            f"Found {n_bad} value(s) outside this range in {response!r}. "
            + ("" if squeeze else "Pass squeeze=True to handle boundary values 0 and 1."),
            column=response,
            context={"n_boundary_values": n_bad},
        )

    if n_col is not None and deff_col is not None:
        phi_df = df[[n_col, deff_col]].dropna()
        if (phi_df[n_col] <= 0).any():
            raise DataValidationError(
                f"{n_col!r}: sample sizes must be positive.", column=n_col
            )
        if (phi_df[deff_col] <= 0).any():
            raise DataValidationError(
                f"{deff_col!r}: design effects must be positive.", column=deff_col
            )
        phi = phi_df[n_col].values / phi_df[deff_col].values - 1
        n_bad_phi = int((phi <= 0).sum())
        if n_bad_phi:
            raise DataValidationError(
                f"Beta precision phi = n/deff - 1 must be > 0. "
                f"Found {n_bad_phi} row(s) where n/deff <= 1.",
                context={"n_invalid_phi": n_bad_phi},
            )


def _preprocess_beta(df: pd.DataFrame, response: str, ctx: dict) -> pd.DataFrame:
    """Add ``log_phi`` column; optionally apply the Smithson-Verkuilen squeeze.

    Only active when both ``n_col`` and ``deff_col`` are provided.
    ``log_phi = log(n/deff - 1)`` is always computed when the precision is
    pinned from data — it is required by the Bambi distributional formula
    ``"kappa ~ 1 + offset(log_phi)"``. The squeeze ``(y*(n-1)+0.5)/n`` is
    applied only when ``squeeze=True`` (for datasets with boundary ``y=0``/
    ``y=1``); otherwise the response is left untouched.
    """
    n_col = ctx.get("n_col")
    deff_col = ctx.get("deff_col")
    squeeze = ctx.get("squeeze", False)
    if n_col is not None and deff_col is not None:
        n_vals = df[n_col].to_numpy(dtype=float)
        deff_vals = df[deff_col].to_numpy(dtype=float)

        if squeeze:
            y = df[response].to_numpy(dtype=float)
            df[response] = (y * (n_vals - 1) + 0.5) / n_vals
            logger.debug(
                "Beta: Smithson-Verkuilen squeeze applied (squeeze=True)."
            )

        df[_BETA_OFFSET] = np.log(n_vals / deff_vals - 1)
        logger.debug(
            "Beta: log_phi in [%.3f, %.3f].",
            df[_BETA_OFFSET].min(), df[_BETA_OFFSET].max(),
        )
    return df


def _check_binomial(df: pd.DataFrame, response: str, ctx: dict) -> None:
    """Require non-negative integer successes not exceeding positive-integer trials."""
    trials_col = ctx.get("trials_col")
    y = df[response].dropna()
    if (y < 0).any():
        raise DataValidationError(
            f"Binomial family: {response!r} must be non-negative.", column=response
        )
    if not (y % 1 == 0).all():
        raise DataValidationError(
            f"Binomial family: {response!r} must contain integers.", column=response
        )
    if trials_col is not None:
        check = df[[response, trials_col]].dropna()
        n = check[trials_col]
        if (n < 1).any() or not (n % 1 == 0).all():
            raise DataValidationError(
                f"Binomial family: {trials_col!r} must contain positive integers.",
                column=trials_col,
            )
        n_bad = int((check[response].values > check[trials_col].values).sum())
        if n_bad:
            raise DataValidationError(
                f"Binomial family: {response!r} must not exceed {trials_col!r}. "
                f"Found {n_bad} row(s) where y > n.",
                column=response,
                context={"n_exceeding_trials": n_bad},
            )


# Adding a family that reuses the existing pipeline fields is a two-entry
# change: (1) a FamilySpec entry here carrying metadata AND behavior
# (response_check / preprocess / addition_template / fixed_params /
# required_params), plus (2) a MODEL_REGISTRY entry in _factory.py pointing at
# a subclass whose only job is to store those fields. Everything the offset
# machinery needs — which parameter is pinned, on what link, from which column,
# and when the pin applies — is read off `fixed_params`, so no formula or prior
# code is written per family. A subclass hook is needed ONLY for formula
# behavior a template cannot express (Binomial rewrites its LHS) or a guard a
# spec cannot state (Beta's squeeze/offset combination).
#
# That holds even for a family introducing a *new* pipeline field:
# `DataValidator.validate()` and `DataPreprocessor.process()` take the family's
# fields as **kwargs and derive which of them name data columns (the string
# values), so a field they have never heard of needs no edit there.
FAMILY_SPECS: dict[str, FamilySpec] = {
    "gaussian": FamilySpec(
        bambi_family="gaussian",
        mean_param_key="mu",
        default_link="identity",
        supported_links=frozenset({"identity", "log"}),
        pipeline_fields=("sampling_var_col",),
        user_params={"sampling_var_col": "sampling_var"},
        response_check=_check_gaussian,
        preprocess=_preprocess_gaussian,
        fixed_params=(
            FixedParam("sigma", "log", _GAUSSIAN_OFFSET, ("sampling_var_col",)),
        ),
    ),
    "beta": FamilySpec(
        bambi_family="beta",
        mean_param_key="mu",
        default_link="logit",
        supported_links=frozenset({"logit", "probit"}),
        pipeline_fields=("n_col", "deff_col", "squeeze"),
        user_params={
            "n_col": "n",
            "deff_col": "deff",
            "squeeze": "squeeze",
        },
        response_check=_check_beta,
        preprocess=_preprocess_beta,
        fixed_params=(
            FixedParam("kappa", "log", _BETA_OFFSET, ("n_col", "deff_col")),
        ),
    ),
    "binomial": FamilySpec(
        bambi_family="binomial",
        mean_param_key="p",
        default_link="logit",
        supported_links=frozenset({"logit", "probit"}),
        pipeline_fields=("trials_col",),
        user_params={"trials_col": "trials"},
        response_check=_check_binomial,
        preprocess=None,  # binomial needs no offset transform
        addition_template="p({response}, {trials_col})",
        addition_field="trials_col",
        # Nothing is pinned: p is the only parameter and it is the estimand.
        fixed_params=(),
        required_params=("trials",),
    ),
}


def list_families() -> list[str]:
    """Return the registered family names, sorted (mirrors R `list_hbsae_models()`)."""
    return sorted(FAMILY_SPECS)


def get_family_spec(name: str) -> FamilySpec:
    """Return the `FamilySpec` for family *name* (mirrors R `get_hbsae_model()`).

    Raises:
        ModelRegistryError: When *name* is not a registered family.
    """
    try:
        return FAMILY_SPECS[name]
    except KeyError:
        raise ModelRegistryError(
            f"Unknown family {name!r}. Registered families: {list_families()}.",
            family=name,
            registered=list_families(),
        ) from None
