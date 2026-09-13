"""Tier 3 — family-specific shortcuts (`hbm_beta`, `hbm_gaussian`, ...).

Each function hardcodes `family=` and exposes only kwargs relevant to that
family, so a typo like ``hbm_gaussian(..., trials="n")`` is a Python
`TypeError` (unknown kwarg) instead of a downstream `ValueError`. All
shortcuts delegate to `hbm_flex()`.

That rule decides `fixed_params=` too: a shortcut takes it **iff** the
family declares something pinnable, i.e. ``FAMILY_SPECS[family].fixed_params``
is non-empty. Binomial pins nothing (``p`` is the estimand), so
``hbm_binomial(..., fixed_params=...)`` is a `TypeError` rather than an
argument that could only ever raise `ValueError` downstream.

This is also the layer that translates: tier 2 is family-agnostic, so a
shortcut's per-family arguments go down as ``aux_args`` (Beta's ``n``/``deff``/
``squeeze``) or ``addition_var`` (Binomial's ``trials``). The GUI reads these
signatures with ``inspect.signature``, so the names here are a cross-team
contract — keep them.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from hbsaemp._types import MissingStrategyLiteral, PriorDict
from hbsaemp.models._base import BaseModel
from hbsaemp.models._config import ModelConfig
from hbsaemp.models._flex import hbm_flex

__all__: list[str] = [
    "hbm_beta",
    "hbm_gaussian",
    "hbm_binomial",
]


def hbm_gaussian(
    response: str,
    auxiliary: Sequence[str],
    data: pd.DataFrame,
    *,
    sampling_var: str | None = None,
    area_var: str | None = None,
    intercept: bool = True,
    link: str | None = None,
    priors: PriorDict | None = None,
    handle_missing: MissingStrategyLiteral = "deleted",
    config: ModelConfig | None = None,
    fixed_params: dict[str, str | float] | None = None,
) -> BaseModel:
    """Gaussian-family shortcut. Pass `sampling_var=` to enable the FH offset.

    `fixed_params={"sigma": <column or number>}` pins the scale directly;
    it and `sampling_var=` pin the same parameter, so pass one or the other.
    """
    return hbm_flex(
        response, auxiliary, data, family="gaussian",
        area_var=area_var, intercept=intercept,
        sampling_var=sampling_var, link=link,
        priors=priors, handle_missing=handle_missing, config=config,
        fixed_params=fixed_params,
    )


def hbm_beta(
    response: str,
    auxiliary: Sequence[str],
    data: pd.DataFrame,
    *,
    n: str | None = None,
    deff: str | None = None,
    squeeze: bool = False,
    area_var: str | None = None,
    intercept: bool = True,
    link: str | None = None,
    priors: PriorDict | None = None,
    handle_missing: MissingStrategyLiteral = "deleted",
    config: ModelConfig | None = None,
    fixed_params: dict[str, str | float] | None = None,
) -> BaseModel:
    """Beta-family shortcut. Pass `n=`/`deff=` together to use the FH offset.

    Set `squeeze=True` for the Smithson-Verkuilen transform when the data
    contains boundary values y=0 or y=1.

    `fixed_params={"kappa": <column or number>}` pins the precision directly;
    it and `n=`/`deff=` pin the same parameter, so pass one or the other.
    """
    return hbm_flex(
        response, auxiliary, data, family="beta",
        area_var=area_var, intercept=intercept,
        aux_args={"n": n, "deff": deff, "squeeze": squeeze},
        link=link,
        priors=priors, handle_missing=handle_missing, config=config,
        fixed_params=fixed_params,
    )


def hbm_binomial(
    response: str,
    auxiliary: Sequence[str],
    data: pd.DataFrame,
    *,
    trials: str,
    area_var: str | None = None,
    intercept: bool = True,
    link: str | None = None,
    priors: PriorDict | None = None,
    handle_missing: MissingStrategyLiteral = "deleted",
    config: ModelConfig | None = None,
) -> BaseModel:
    """Binomial-family shortcut. `trials=` (column of trial counts) is required."""
    return hbm_flex(
        response, auxiliary, data, family="binomial",
        area_var=area_var, intercept=intercept,
        addition_var=trials, link=link,
        priors=priors, handle_missing=handle_missing, config=config,
    )
