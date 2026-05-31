"""Tier 2 — `hbm_flex()`: family-agnostic wrapper that builds the formula.

User supplies `response` (str) and `auxiliary` (list of column names)
instead of a full R-style formula. The constructed formula is forwarded
to `create_model()`; this layer only validates the inputs and assembles
the formula string.
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
    sampling_var: str | None = None,
    n: str | None = None,
    deff: str | None = None,
    trials: str | None = None,
    link: str | None = None,
    squeeze: bool = False,
) -> BaseModel:
    """Build an unfitted model from response + auxiliary list (no formula).

    Tier 2 of the 3-tier API. The formula
    ``"{response} ~ {' + '.join(auxiliary)}"`` is assembled internally and
    forwarded to `create_model()`. Family validation, group auto-injection,
    and prior checks all happen downstream in `create_model()` — this
    function only validates the shape of `response`, `auxiliary`, and
    `area_var`.

    Args:
        response: Response column name (bare identifier).
        auxiliary: Non-empty sequence of predictor column names (bare
            identifiers). Use `create_model()` directly for intercept-only
            or in-formula transforms.
        data: Input `pandas.DataFrame`.
        family: One of `"gaussian"`, `"beta"`, `"binomial"`.
        area_var: Grouping column for the random intercept ``(1|area_var)``;
            forwarded as `group=` to `create_model()` (which auto-injects).
        intercept: When `False`, emit ``"y ~ 0 + ..."`` to suppress the
            fixed intercept.
        priors: Optional prior dict or `Prior` instances; forwarded.
        handle_missing: Missing data strategy; only `"deleted"` in v1.
        config: `ModelConfig` (sampler settings); `None` uses `DEFAULT_CONFIG`.
        sampling_var: Gaussian FH — sampling variance column.
        n: Beta — survey sample size column (pair with `deff`).
        deff: Beta — design effect column (pair with `n`).
        trials: Binomial — column with trial counts.
        link: Override default link function.
        squeeze: Beta — Smithson-Verkuilen squeeze for boundary y values.

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
        sampling_var=sampling_var,
        n=n,
        deff=deff,
        trials=trials,
        link=link,
        squeeze=squeeze,
    )
