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


#: Accepted by every family, so it is not part of any family's `user_params`.
_UNIVERSAL_PARAMS: frozenset[str] = frozenset({"link"})


def _validate_family_args(family: str, provided: set[str]) -> None:
    """Reject user-facing kwargs not in the chosen family's spec.

    The allowed set is `FAMILY_SPECS[family].user_params.values()` plus the
    universal ones. Caller-side, `None`-valued args and `squeeze=False` are
    excluded from *provided* (treated as not supplied).

    Raises:
        ValueError: When `provided - allowed` is non-empty.
    """
    allowed = set(FAMILY_SPECS[family].user_params.values()) | _UNIVERSAL_PARAMS
    invalid = provided - allowed
    if invalid:
        raise ValueError(
            f"Argument(s) not valid for family={family!r}: {sorted(invalid)}. "
            f"Valid for {family!r}: {sorted(allowed)}."
        )


def _resolve_aux_args(
    family: str,
    aux_args: dict[str, object] | None,
    addition_var: str | None,
) -> dict[str, object]:
    """Turn the two family-agnostic channels into plain family kwargs.

    `aux_args` names the family's own parameters directly; `addition_var`
    names the addition-term column without the caller having to know what
    that family calls it. Both are checked here so a new family gets the same
    typo protection the built-in ones get from `create_model()`'s signature.

    Raises:
        ValueError: Unknown key in `aux_args`, `addition_var` on a family that
            takes none, or the same parameter set through both channels.
    """
    if aux_args is not None and not isinstance(aux_args, dict):
        # Before the dict() below, which would otherwise turn a list into an
        # opaque "dictionary update sequence element #0" message.
        raise ValueError(
            f"`aux_args` must be a dict of {{parameter: value}}, "
            f"got {type(aux_args).__name__!r}."
        )

    spec = FAMILY_SPECS[family]
    resolved: dict[str, object] = dict(aux_args or {})

    allowed = set(spec.user_params.values()) | _UNIVERSAL_PARAMS
    unknown = sorted(set(resolved) - allowed)
    if unknown:
        raise ValueError(
            f"aux_args key(s) not valid for family={family!r}: {unknown}. "
            f"Valid for {family!r}: {sorted(allowed)}."
        )

    if addition_var is not None:
        if spec.addition_field is None:
            raise ValueError(
                f"family={family!r} takes no addition variable, so "
                f"addition_var={addition_var!r} has nothing to fill. "
                f"Families with one: "
                f"{sorted(k for k, s in FAMILY_SPECS.items() if s.addition_field)}."
            )
        name = spec.user_params[spec.addition_field]
        if name in resolved:
            raise ValueError(
                f"{name!r} is set twice: once through addition_var and once "
                f"through aux_args[{name!r}]. Pass one or the other."
            )
        resolved[name] = addition_var

    return resolved


def _validate_fixed_params(
    family: str,
    fixed_params: dict[str, str | float] | None,
    supplied: set[str],
) -> None:
    """Check a caller-supplied `fixed_params` against the family's spec.

    Three ways it can be wrong, each caught here rather than deep in Bambi:
    a parameter the family cannot pin, a value that is neither a column name
    nor a number, and a parameter the family is *already* pinning from the
    survey-design columns the caller also passed.

    Raises:
        ValueError: On any of the three.
    """
    if not fixed_params:
        return
    if not isinstance(fixed_params, dict):
        raise ValueError(
            f"`fixed_params` must be a dict of {{parameter: column or value}}, "
            f"got {type(fixed_params).__name__!r}."
        )

    spec = FAMILY_SPECS[family]
    pinnable = spec.pinnable_params

    unknown = sorted(set(fixed_params) - set(pinnable))
    if unknown:
        raise ValueError(
            f"family={family!r} cannot pin {unknown}. "
            f"Pinnable parameter(s): {sorted(pinnable) or 'none'}."
        )

    for param, source in fixed_params.items():
        if not isinstance(source, str | int | float) or isinstance(source, bool):
            raise ValueError(
                f"fixed_params[{param!r}] must be a column name or a number, "
                f"got {type(source).__name__!r}."
            )

    # Both doors pin the same parameter: refuse rather than letting one win.
    for fp in spec.fixed_params:
        if fp.param not in fixed_params:
            continue
        sugar = [spec.user_params[f] for f in fp.source_fields]
        if any(name in supplied for name in sugar):
            raise ValueError(
                f"{fp.param!r} is pinned twice: once through {sugar} and once "
                f"through fixed_params[{fp.param!r}]. Pass one or the other."
            )


def _validate_family_arity(family: str, supplied: set[str]) -> None:
    """Enforce the spec's arity rules on the family-specific kwargs.

    Two rules, both read off `FAMILY_SPECS` rather than hardcoded per family:

    * `required_params` — the family cannot be fitted without them.
    * `FixedParam.source_fields` — a pin derived from several columns needs
      all of them or none; supplying half leaves the offset uncomputable.

    Raises:
        ValueError: On a missing required param, or a partially supplied pin.
    """
    spec = FAMILY_SPECS[family]

    missing = [p for p in spec.required_params if p not in supplied]
    if missing:
        raise ValueError(
            f"family={family!r} requires the {missing[0]!r} parameter: "
            f"pass the name of the column holding those values."
        )

    for fp in spec.fixed_params:
        if len(fp.source_fields) < 2:
            continue
        names = [spec.user_params[f] for f in fp.source_fields]
        given = [n for n in names if n in supplied]
        if given and len(given) != len(names):
            raise ValueError(
                f"family={family!r}: provide all of {names} or none of them. "
                f"Together they pin {fp.param!r}; {sorted(set(names) - set(given))} "
                f"{'is' if len(names) - len(given) == 1 else 'are'} missing."
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
    fixed_params: dict[str, str | float] | None = None,
    # ^ pin a distributional parameter to known values instead of sampling it
    addition_var: str | None = None,
    # ^ family-agnostic name for the addition-term column (binomial: trials)
    aux_args: dict[str, object] | None = None,
    # ^ family-agnostic channel for the family's own parameters; the only way
    #   a family whose kwarg names are not in the signature above can be used
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
        fixed_params: `{parameter: column name or number}` pinning a
            distributional parameter to known values instead of sampling it,
            e.g. `{"sigma": "sd_col"}` or `{"sigma": 2.0}`. The value is the
            parameter itself, not its linear predictor — hbsaemp applies the
            link. Only parameters the family declares as pinnable are
            accepted, and a parameter already pinned through the survey-design
            arguments (`sampling_var`, `n`+`deff`) is refused rather than
            silently overridden.
        addition_var: Column for the family's addition term, named without
            knowing the family's own vocabulary — `addition_var="n"` on a
            Binomial is `trials="n"`. Refused for a family that takes none.
        aux_args: `{parameter: value}` for the family's own parameters, using
            the same names as the keyword arguments above
            (`{"n": "n", "deff": "deff"}` is `n="n", deff="deff"`). The two
            forms are interchangeable for the families listed here; for a
            family whose parameters this signature does not name, `aux_args`
            is the way to pass them. Keys are checked against the family spec.

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

    # The v1 families' kwargs, spelled out so a typo is a Python TypeError.
    # A family whose parameters are NOT listed here reaches the same dict
    # through `aux_args` — that is what keeps adding one from editing this
    # signature. Merging before `_provided_optional` is computed is what puts
    # aux_args under the same cross-family and arity checks as the rest; a key
    # that never entered `user_kw` could never appear in `provided`, so those
    # checks silently did not apply to it.
    user_kw: dict[str, object] = {
        "sampling_var": sampling_var,
        "n": n, "deff": deff, "squeeze": squeeze,
        "trials": trials, "link": link,
    }
    user_kw.update(_resolve_aux_args(family, aux_args, addition_var))

    # After the merge, so the guard covers both doors. `squeeze` reaches the
    # preprocessor as a plain truth test, so a stray string would silently
    # enable the transform and relax the response-domain check with it.
    if not isinstance(user_kw["squeeze"], bool):
        raise TypeError(
            f"`squeeze` must be a bool, got "
            f"{type(user_kw['squeeze']).__name__!r}."
        )

    # `squeeze` is a bool with a safe default, so only `True` counts as
    # supplied; everything else is supplied iff not None.
    _provided_optional: set[str] = {
        name for name, val in user_kw.items()
        if val is not None and val is not False
    }

    # Cross-family argument validation (fail-fast on stray kwargs).
    _validate_family_args(family, _provided_optional)
    # Arity rules come from the spec: `required_params` and the multi-column
    # `FixedParam.source_fields`. No per-family branch lives here.
    _validate_family_arity(family, _provided_optional)
    _validate_fixed_params(family, fixed_params, _provided_optional)

    # Validate prior dicts early — before fit() so errors surface immediately.
    # Prior objects are already validated at construction; only raw dicts need
    # checking here.  Import is inline to keep bambi as a lazy dependency.
    from hbsaemp.models._prior import validate_priors

    validate_priors(priors)  # raises PriorSpecError if malformed

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
    # `link` is universal, so it is passed directly rather than through
    # `user_params` — see `_UNIVERSAL_PARAMS`.
    # `.get()`, not `[]`: a family whose parameters are not in the signature
    # above and that the caller left unset has no entry here, and "unset
    # optional family parameter" is exactly what None means. Subscripting
    # raised a bare `KeyError` naming only the kwarg, which made such a family
    # unusable with no hint why. Typos are caught by `_resolve_aux_args` and
    # `_validate_family_args` above, so nothing is masked by the fallback.
    family_kw = {
        mp: user_kw.get(up) for mp, up in FAMILY_SPECS[family].user_params.items()
    }

    model_class = MODEL_REGISTRY[family]
    return model_class(
        resolved_formula,
        family,
        data,
        resolved_config,
        priors=priors,
        group=group,
        handle_missing=handle_missing,
        link=link,
        fixed_params=fixed_params,
        **family_kw,
    )


#: R-style alias — ``hbm(...)`` is identical to ``create_model(...)``.
#: Provided for users migrating from R ``hbsaems``.
hbm = create_model
