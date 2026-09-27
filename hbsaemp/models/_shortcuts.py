"""Tier 1 (beginner interface) — family-specific shortcuts (`hbm_beta`, ...).

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

    Args:
        response: Response column name: the direct estimate of each area.
        auxiliary: Non-empty sequence of auxiliary (predictor) column names.
        data: Input `pandas.DataFrame`, one row per area.
        sampling_var: Column of known sampling variances D_i; enables the
            Fay-Herriot offset. The same as `sampling_variance` in hbsaems.
        area_var: Area identifier column; adds the random intercept
            ``(1|area_var)``.
        intercept: `False` drops the fixed intercept.
        link: Override the default link.
        priors: Optional prior dict or `Prior` instances, keyed by term.
        handle_missing: Missing-data strategy; only `"deleted"`.
        config: `ModelConfig` (sampler settings); `None` uses
            `DEFAULT_CONFIG`.
        fixed_params: `{"sigma": <column or number>}` pins the scale
            directly. It pins the same parameter as `sampling_var`, so pass
            one or the other.

    Returns:
        An unfitted model; call `fit()` to sample.
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

    Args:
        response: Response column name: the direct proportion of each area,
            strictly between 0 and 1 unless `squeeze=True`.
        auxiliary: Non-empty sequence of auxiliary (predictor) column names.
        data: Input `pandas.DataFrame`, one row per area.
        n: Survey sample-size column. Used with `deff` to fix the precision
            at phi = n/deff - 1.
        deff: Design-effect column. Used with `n`.
        squeeze: Apply the Smithson-Verkuilen transform `(y*(n-1)+0.5)/n`
            when the data contains boundary values y=0 or y=1. Requires `n`
            and `deff`.
        area_var: Area identifier column; adds the random intercept
            ``(1|area_var)``.
        intercept: `False` drops the fixed intercept.
        link: Override the default link.
        priors: Optional prior dict or `Prior` instances, keyed by term.
        handle_missing: Missing-data strategy; only `"deleted"`.
        config: `ModelConfig` (sampler settings); `None` uses
            `DEFAULT_CONFIG`.
        fixed_params: `{"kappa": <column or number>}` pins the precision
            directly. It pins the same parameter as `n`/`deff`, so pass one
            or the other.

    Returns:
        An unfitted model; call `fit()` to sample.
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
    """Binomial-family shortcut. `trials=` (column of trial counts) is required.

    Args:
        response: Response column name: the count of successes in each area,
            a non-negative integer not above `trials`.
        auxiliary: Non-empty sequence of auxiliary (predictor) column names.
        data: Input `pandas.DataFrame`, one row per area.
        trials: Column of trial counts, positive integers (required).
        area_var: Area identifier column; adds the random intercept
            ``(1|area_var)``.
        intercept: `False` drops the fixed intercept.
        link: Override the default link.
        priors: Optional prior dict or `Prior` instances, keyed by term.
        handle_missing: Missing-data strategy; only `"deleted"`.
        config: `ModelConfig` (sampler settings); `None` uses
            `DEFAULT_CONFIG`.

    Returns:
        An unfitted model; call `fit()` to sample.
    """
    return hbm_flex(
        response, auxiliary, data, family="binomial",
        area_var=area_var, intercept=intercept,
        addition_var=trials, link=link,
        priors=priors, handle_missing=handle_missing, config=config,
    )
