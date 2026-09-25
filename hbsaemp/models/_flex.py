"""Tier 2 (intermediate interface) — `hbm_flex()`, which builds the formula.

User supplies `response` (str) and `auxiliary` (list of column names)
instead of a full R-style formula. The constructed formula is forwarded
to `create_model()`; this layer only validates the inputs and assembles
the formula string.

Family-agnostic is literal: no argument here names a family's own vocabulary.
A family's parameters arrive through `aux_args`, its addition term through
`addition_var`, and its pinned parameters through `fixed_params` — the same
three channels hbsaems' `hbm_flex()` uses. Adding a family therefore does not
touch this signature. Tier 1 (`_shortcuts.py`) is where the per-family names
live, so a typo there is still a Python `TypeError`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import pandas as pd

from hbsaemp._types import FamilyLiteral, MissingStrategyLiteral, PriorDict
from hbsaemp.models._base import BaseModel
from hbsaemp.models._config import ModelConfig
from hbsaemp.models._factory import create_model

__all__: list[str] = ["hbm_flex"]

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_]\w*$")


def _check_identifier(value: object, *, field: str) -> str:
    r"""Raise unless *value* is a bare column identifier (str matching ``\w+``)."""
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(
            f"{field}={value!r} must be a bare column identifier "
            f"(letters, digits, underscore; not starting with a digit). "
            f"Pre-compute transformations as DataFrame columns and pass the "
            f"new column name."
        )
    return value


def hbm_flex(
    response: str,
    auxiliary: Sequence[str],
    data: pd.DataFrame,
    family: FamilyLiteral,
    *,
    area_var: str | None = None,
    intercept: bool = True,
    priors: PriorDict | None = None,
    handle_missing: MissingStrategyLiteral = "deleted",
    config: ModelConfig | None = None,
    addition_var: str | None = None,
    aux_args: dict[str, object] | None = None,
    sampling_var: str | None = None,
    fixed_params: dict[str, str | float] | None = None,
    link: str | None = None,
) -> BaseModel:
    """Build an unfitted model from response + auxiliary list (no formula).

    Tier 2 (intermediate interface). Builds the formula
    ``"{response} ~ {' + '.join(auxiliary)}"`` and passes it to
    `create_model()`, which validates the family, adds the group term and
    checks the priors. This function checks only `response`, `auxiliary` and
    `area_var`.

    Args:
        response: Response column name.
        auxiliary: Non-empty sequence of predictor column names. For an
            intercept-only model, use `create_model()`.
        data: Input `pandas.DataFrame`.
        family: One of `"gaussian"`, `"beta"`, `"binomial"`.
        area_var: Grouping column for the random intercept ``(1|area_var)``;
            passed to `create_model()` as `group=`.
        intercept: `False` writes ``"y ~ 0 + ..."`` to drop the fixed
            intercept.
        priors: Optional prior dict or `Prior` instances.
        handle_missing: Missing-data strategy; only `"deleted"`.
        config: `ModelConfig` (sampler settings); `None` uses `DEFAULT_CONFIG`.
        addition_var: Column for the family's addition term, such as the
            trial counts for Binomial.
        aux_args: `{parameter: value}` for the family's own parameters, e.g.
            `{"n": "n", "deff": "deff"}` for Beta or `{"squeeze": True}`.
        sampling_var: Column of known sampling variances D_i, for the
            Fay-Herriot model. The same as `sampling_variance` in hbsaems.
        fixed_params: `{parameter: column name or number}` fixing a
            distributional parameter to known values.
        link: Override the default link.

    Returns:
        Unfitted `BaseModel` subclass.

    Raises:
        ValueError: Empty `auxiliary`, or non-identifier in any of
            `response` / `auxiliary` / `area_var`.
        TypeError: `auxiliary` is not a sequence of strings, or `data` is
            not a DataFrame (deferred to `create_model`).
    """
    _check_identifier(response, field="response")

    if isinstance(auxiliary, str) or not isinstance(auxiliary, Sequence):
        raise TypeError(
            f"`auxiliary` must be a sequence of column names (list/tuple), "
            f"got {type(auxiliary).__name__!r}. Wrap a single column in a list."
        )
    if len(auxiliary) == 0:
        raise ValueError(
            "`auxiliary` must be non-empty. Use create_model() directly "
            "for intercept-only or random-effects-only models."
        )
    for i, term in enumerate(auxiliary):
        _check_identifier(term, field=f"auxiliary[{i}]")

    if area_var is not None:
        _check_identifier(area_var, field="area_var")

    rhs_terms = list(auxiliary) if intercept else ["0", *auxiliary]
    formula = f"{response} ~ {' + '.join(rhs_terms)}"

    return create_model(
        formula,
        family,
        data,
        group=area_var,
        priors=priors,
        handle_missing=handle_missing,
        config=config,
        addition_var=addition_var,
        aux_args=aux_args,
        sampling_var=sampling_var,
        fixed_params=fixed_params,
        link=link,
    )
