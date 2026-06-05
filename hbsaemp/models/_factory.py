"""Single entry point `create_model` (alias `hbm`) and `MODEL_REGISTRY`.

The `family=` argument selects a `BaseModel` subclass; family-specific kwargs
are dispatched via `FAMILY_SPECS`. To register a new family, add an entry to
both `MODEL_REGISTRY` and `FAMILY_SPECS`.
"""

from __future__ import annotations

import pandas as pd

from hbsaemp._exceptions import ModelRegistryError
from hbsaemp._logging import get_logger
from hbsaemp._types import FamilyLiteral, FormulaStr, MissingStrategyLiteral, PriorDict
from hbsaemp.models._base import BaseModel
from hbsaemp.models._beta import BetaModel
from hbsaemp.models._binomial import BinomialModel
from hbsaemp.models._config import DEFAULT_CONFIG, ModelConfig
from hbsaemp.models._family_spec import FAMILY_SPECS
from hbsaemp.models._gaussian import GaussianModel

logger = get_logger(__name__)

__all__: list[str] = ["create_model", "hbm", "MODEL_REGISTRY"]


# MODEL_REGISTRY — maps family strings to model classes
MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "gaussian":  GaussianModel,
    "beta":      BetaModel,
    "binomial":  BinomialModel,
}

# Reserved family names fail with a roadmap-aware message instead of looking
# like arbitrary typos. Implement them only after their public contract is set.
_PLANNED_FAMILIES: dict[str, str] = {
    "lognormal": "V2",
}


def _validate_family_args(family: str, provided: set[str]) -> None:
    """Reject user-facing kwargs not in the chosen family's spec.

    The allowed set is derived from `FAMILY_SPECS[family].user_params.values()`.
    Caller-side, `None`-valued args and `squeeze=False` are excluded from
    *provided* (treated as not supplied).

    Raises:
        ValueError: When `provided - allowed` is non-empty.
    """
    allowed = set(FAMILY_SPECS[family].user_params.values())
    invalid = provided - allowed
    if invalid:
        raise ValueError(
            f"Argument(s) not valid for family={family!r}: {sorted(invalid)}. "
            f"Valid for {family!r}: {sorted(allowed)}."
        )


def _resolve_group_formula(formula: FormulaStr, group: str | None) -> FormulaStr:
    """Append a random intercept for *group* unless the formula already has one.

    An explicit random-effect term in the user formula wins over the convenience
    `group=` argument. `parse_formula()` recognises every spacing and slope
    variant (`(1|g)`, `(1 | g)`, `(x1 | g)`, `(0 + x1 | g)`), so this structural
    check replaces the old literal ``f"(1|{group})" in formula`` substring test,
    which missed spaced/slope forms and double-injected a duplicate intercept.
    """
    if group is None:
        return formula

    from hbsaemp.utils._formula import parse_formula

    random_groups = set(parse_formula(formula)["random_groups"])
    if group in random_groups:
        return formula
    return f"{formula} + (1|{group})"


# create_model — unified entry point

def create_model(
    formula: FormulaStr,
    family: FamilyLiteral,
    data: pd.DataFrame,
    *,
    config: ModelConfig | None = None,
    priors: PriorDict | None = None,
    group: str | None = None,
    handle_missing: MissingStrategyLiteral = "deleted",
    # Family-specific optional parameters
    trials: str | None = None,
    # ^ binomial: column name for number of trials
    n: str | None = None,
    # ^ beta: column name for survey sample size  (to compute phi)
    deff: str | None = None,
    # ^ beta: column name for design effect       (to compute phi)
    sampling_var: str | None = None,
    # ^ gaussian FH: column name for sampling variance D_i
    link: str | None = None,
    # ^ override default link for the family
    squeeze: bool = False,
    # ^ beta: apply Smithson-Verkuilen squeeze (y*(n-1)+0.5)/n before fitting;
    #   use only for datasets with boundary values y=0 or y=1
) -> BaseModel:
    """Build an unfitted model (single entry point); call `.fit()` to sample.

    Args:
        formula: R/lme4-style formula, e.g. `"y ~ x1 + x2 + (1|group)"`.
            If `group` is also supplied and `(1|group)` is not in the
            formula, it is appended automatically.
        family: One of `"gaussian"`, `"beta"`, `"binomial"`.
        data: Input `pandas.DataFrame`.
        config: `ModelConfig` (sampler settings). `None` uses `DEFAULT_CONFIG`.
        priors: Prior dict or `None` for Bambi auto-priors. Format:
            `{"x1": {"dist": "Normal", "mu": 0, "sigma": 1}}` or
            `{"x1": Prior("Normal", mu=0, sigma=1)}`.
        group: Grouping column for random effects.
        handle_missing: Only `"deleted"` is supported in v1.
        trials: Binomial — column with trial counts (required).
        n: Beta — survey sample-size column; use with `deff` (together they
            give phi = n/deff - 1).
        deff: Beta — design-effect column; use with `n` (together they give
            phi = n/deff - 1).
        sampling_var: Gaussian FH — column with known sampling variances D_i,
            enables the FH offset.
        link: Override default link.
        squeeze: Beta — apply Smithson-Verkuilen squeeze `(y*(n-1)+0.5)/n`.
            Only when data has y=0 or y=1 (requires n+deff).

    Returns:
        Unfitted `BaseModel` subclass.

    Raises:
        ModelRegistryError: Unknown `family`.
        TypeError: `data` is not a DataFrame, or `squeeze` is not bool.
        ValueError: Missing required family-specific parameter, or stray
            cross-family kwarg.
        PriorSpecError: Malformed entry in `priors`.

    Examples:
        >>> cfg = ModelConfig(draws=2000, chains=4)
        >>> m = create_model("y ~ x1 + x2", family="gaussian",
        ...                  data=df, group="area", config=cfg)
        >>> m.fit()
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"`data` must be a pandas DataFrame, got {type(data).__name__!r}"
        )

    resolved_config: ModelConfig = config if config is not None else DEFAULT_CONFIG

    if family in _PLANNED_FAMILIES:
        planned_version = _PLANNED_FAMILIES[family]
        raise ModelRegistryError(
            f"family={family!r} is planned for {planned_version} and is not "
            f"available in V1. Registered V1 families: {sorted(MODEL_REGISTRY)}.",
            family=family,
            registered=sorted(MODEL_REGISTRY),
        )

    if family not in MODEL_REGISTRY:
        raise ModelRegistryError(
            f"Unknown family {family!r}. "
            f"Registered families: {sorted(MODEL_REGISTRY)}.",
            family=family,
            registered=sorted(MODEL_REGISTRY),
        )

    # Cross-family argument validation (fail-fast on stray kwargs).
    # `squeeze` is a bool with safe default False; only squeeze=True counts as
    # user-provided. Other optional cols/link are "provided" iff not None.
    if not isinstance(squeeze, bool):
        raise TypeError(
            f"`squeeze` must be a bool, got {type(squeeze).__name__!r}."
        )

    _provided_optional: set[str] = {
        name for name, val in {
            "sampling_var": sampling_var,
            "n": n, "deff": deff,
            "trials": trials, "link": link,
        }.items()
        if val is not None
    }
    if squeeze is True:
        _provided_optional.add("squeeze")

    _validate_family_args(family, _provided_optional)

    # Family-specific parameter validation
    if family == "binomial" and trials is None:
        raise ValueError(
            "family='binomial' requires the `trials` parameter: "
            "the column name containing the number of trials per area."
        )

    if family == "beta" and (n is None) != (deff is None):
        raise ValueError(
            "family='beta': provide both `n` and `deff`, or neither. "
            "Both are needed to compute the precision parameter phi."
        )

    # Validate prior dicts early — before fit() so errors surface immediately.
    # Prior objects are already validated at construction; only raw dicts need
    # checking here.  Import is inline to keep bambi as a lazy dependency.
    if priors is not None:
        from hbsaemp.models._prior import Prior
        for pname, pspec in priors.items():
            if not isinstance(pspec, Prior):
                Prior.from_dict(pspec, param=pname)  # raises PriorSpecError if malformed

    # Auto-inject a random intercept for `group`, but only when the formula has
    # no random effect for it already (structural check via parse_formula — an
    # explicit `(1|g)` / `(1 | g)` / `(x1 | g)` is respected, never duplicated).
    resolved_formula = _resolve_group_formula(formula, group)
    if resolved_formula != formula:
        logger.debug("Auto-injected random effect: %r", resolved_formula)

    logger.info(
        "create_model: family=%r, n=%d, formula=%r",
        family, len(data), resolved_formula,
    )

    # Family-aware dispatch: only forward kwargs the chosen model accepts.
    user_kw: dict[str, object] = {
        "sampling_var": sampling_var,
        "n": n, "deff": deff, "squeeze": squeeze,
        "trials": trials, "link": link,
    }
    family_kw = {mp: user_kw[up] for mp, up in FAMILY_SPECS[family].user_params.items()}

    model_class = MODEL_REGISTRY[family]
    return model_class(
        resolved_formula,
        family,
        data,
        resolved_config,
        priors=priors,
        group=group,
        handle_missing=handle_missing,
        **family_kw,
    )


#: R-style alias — ``hbm(...)`` is identical to ``create_model(...)``.
#: Provided for users migrating from R ``hbsaems``.
hbm = create_model
