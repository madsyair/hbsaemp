"""`update_model` (alias `update_hbm`) — refit with new data, formula, priors or sampler.

This is a complete refit, not resumed sampling: all posterior draws in the
returned `ModelResult` are new, as with hbsaems, whose `update_hbm()` reruns
`brms::brm()`. The source model is updated in place so subsequent
`model.predict()` / `model.result` reflect the new fit.
"""
from __future__ import annotations

import dataclasses
import warnings
from typing import Any

import pandas as pd

from hbsaemp._exceptions import DataValidationError
from hbsaemp._logging import get_logger
from hbsaemp._types import FormulaStr, PriorDict
from hbsaemp.models._base import BaseModel, ModelResult
from hbsaemp.models._config import ModelConfig
from hbsaemp.models._family_spec import FAMILY_SPECS
from hbsaemp.models._prior import validate_priors
from hbsaemp.utils._formula import parse_formula, update_formula

logger = get_logger(__name__)
__all__: list[str] = ["update_model", "update_hbm"]


def _carry_design_columns(model: BaseModel, new_data: pd.DataFrame) -> pd.DataFrame:
    """Return *new_data* with any missing survey-design columns filled in.

    The design columns (`sampling_var`, `n`/`deff`) feed the fixed offset.
    When *new_data* lacks them but has as many rows as the current data, they
    are copied over by position with a warning — the treatment hbsaems gives
    its offset columns. The caller's frame is never modified.

    Raises:
        DataValidationError: If columns are missing and the row counts differ.
    """
    source_cols = [
        col
        for name in FAMILY_SPECS[model._family].offset_source_fields
        if (col := getattr(model, f"_{name}")) is not None
    ]
    missing = [c for c in source_cols if c not in new_data.columns]
    if not missing:
        return new_data
    if len(new_data) != len(model._data):
        raise DataValidationError(
            f"new_data is missing survey-design column(s) {missing} and has "
            f"{len(new_data)} row(s) instead of {len(model._data)}, so they "
            "cannot be copied from the current data. Add the column(s) to "
            "new_data, or build a fresh model with create_model().",
            column=missing[0],
            context={"missing_columns": missing},
        )
    warnings.warn(
        f"new_data is missing survey-design column(s) {missing}; copying "
        "them from the current data by position (row order is assumed "
        "unchanged). Add them to new_data if rows were reordered or filtered.",
        UserWarning,
        # stacklevel 3: warn() ← _carry_design_columns() ← update_model() ← caller
        stacklevel=3,
    )
    carried = new_data.copy()
    for col in missing:
        carried[col] = model._data[col].to_numpy()
    return carried


def _merge_priors(
    model: BaseModel, priors: PriorDict | None, *, removed_terms: set[str]
) -> PriorDict | None:
    """Combine *priors* with the model's user priors; new entries win.

    Mirrors `brms::update()`: user priors not mentioned are kept. Priors on
    fixed-effect terms the formula update removed are dropped — brms
    tolerates them, whereas Bambi raises a bare `KeyError` for a prior on a
    term the model does not have.
    """
    merged: dict[str, Any] = {**(model._priors or {}), **(priors or {})}
    dropped = sorted(removed_terms & merged.keys())
    if dropped:
        logger.warning(
            "update_model(): dropping prior(s) for %s — the updated formula "
            "no longer has these term(s).",
            dropped,
        )
        for name in dropped:
            del merged[name]
    return merged or None


def update_model(
    model: BaseModel,
    *,
    new_data: pd.DataFrame | None = None,
    formula: FormulaStr | None = None,
    priors: PriorDict | None = None,
    config: ModelConfig | None = None,
    draws: int | None = None,
    tune: int | None = None,
    chains: int | None = None,
    cores: int | None = None,
    target_accept: float | None = None,
    random_seed: int | None = None,
    max_treedepth: int | None = None,
    progressbar: bool | None = None,
    sampler_kwargs: dict[str, Any] | None = None,
) -> ModelResult:
    """Refit *model* with new data, formula, priors and/or sampler settings.

    Everything not passed is carried over: family, family-specific columns,
    link, grouping, missing-data strategy, the other priors and the other
    sampler settings. The refit samples from scratch. *model* is updated in
    place — `.result`, `.data`, `.config`, `.formula` and its priors reflect
    the new fit — so chained calls start from the current state. A failed
    refit leaves *model* unchanged.

    Sampler settings come either as a full replacement `config` or as
    individual overrides on the current config, not both.

    Args:
        model: Fitted `BaseModel`.
        new_data: Replacement DataFrame; `None` reuses the current data.
            Missing survey-design columns (`sampling_var`, `n`/`deff`) are
            copied from the current data, with a `UserWarning`, when the row
            count is unchanged.
        formula: New formula, or an update template in R `update.formula`
            style: `"."` stands for the current side, `+ term` adds and
            `- term` removes a term, e.g. `". ~ . + x3 - x1"`. Without
            `new_data`, added terms are read from the current data.
        priors: Priors to add or replace, merged into the current user
            priors (new entries win, as in `brms::update()`). Priors on terms
            the formula update removes are dropped.
        config: Full replacement `ModelConfig`.
        draws: Posterior draws per chain (hbsaems `iter` = `draws + tune`).
        tune: Warmup iterations per chain (hbsaems `warmup`).
        chains: Number of MCMC chains.
        cores: Number of parallel sampling cores.
        target_accept: NUTS target acceptance rate (hbsaems
            `control$adapt_delta`); raise toward 0.95-0.99 to clear
            divergences.
        random_seed: Random seed for a reproducible refit.
        max_treedepth: Maximum NUTS tree depth (hbsaems
            `control$max_treedepth`).
        progressbar: Show the sampling progress bar.
        sampler_kwargs: Extra keyword arguments for `pm.sample`; replaces the
            current dict.

    Returns:
        The new `ModelResult`, also stored on `model.result`.

    Raises:
        ModelNotFittedError: If `model` has not been fitted.
        TypeError: If `new_data` is supplied but not a DataFrame.
        ValueError: If `config` is combined with individual overrides, or an
            override fails `ModelConfig` validation.
        FormulaError: If `formula` is malformed or yields an invalid term.
        PriorSpecError: If an entry in `priors` is malformed.
        DataValidationError: If `new_data` lacks design columns that cannot
            be copied, or fails the family's validation.
    """
    # guard
    _ = model.result  # raises ModelNotFittedError if not fitted
    if new_data is not None and not isinstance(new_data, pd.DataFrame):
        raise TypeError(
            f"new_data must be a pandas DataFrame, got {type(new_data).__name__!r}."
        )

    # resolve config
    overrides: dict[str, Any] = {
        name: value
        for name, value in {
            "draws": draws, "tune": tune, "chains": chains, "cores": cores,
            "target_accept": target_accept, "random_seed": random_seed,
            "max_treedepth": max_treedepth, "progressbar": progressbar,
            "sampler_kwargs": sampler_kwargs,
        }.items()
        if value is not None
    }
    if config is not None and overrides:
        raise ValueError(
            "Pass either config= or individual sampler overrides, not both "
            f"(got config and {sorted(overrides)})."
        )
    # replace() copies the config and re-runs ModelConfig.__post_init__ validation.
    new_config: ModelConfig = dataclasses.replace(
        config if config is not None else model._config, **overrides
    )

    # resolve formula, grouping and priors
    new_formula: FormulaStr = (
        update_formula(model._formula, formula) if formula is not None else model._formula
    )
    new_parsed = parse_formula(new_formula)
    # The grouping column follows the formula: a template that removes the
    # random effect removes the grouping too.
    new_group = model._group if model._group in new_parsed["random_groups"] else None
    removed_terms = set(parse_formula(model._formula)["fixed"]) - set(new_parsed["fixed"])
    validate_priors(priors)
    new_priors = _merge_priors(model, priors, removed_terms=removed_terms)

    # resolve data
    target_data: pd.DataFrame = (
        _carry_design_columns(model, new_data) if new_data is not None else model._data
    )

    logger.info(
        "update_model(): family=%r, formula=%r, draws=%d, chains=%d, new_data=%s",
        model._family,
        new_formula,
        new_config.draws,
        new_config.chains,
        "None (reuse current)" if new_data is None else f"shape={new_data.shape}",
    )

    # Rebuild the same subclass, forwarding only its family attrs.
    # FAMILY_SPECS[family].user_params keys mirror `self._<key>` on the model.
    family_attrs = {
        mp: getattr(model, f"_{mp}")
        for mp in FAMILY_SPECS[model._family].user_params
    }
    new_model: BaseModel = type(model)(
        new_formula,
        model._family,
        target_data,
        new_config,
        priors=new_priors,
        group=new_group,
        handle_missing=model._handle_missing,
        **family_attrs,
    )

    # refit
    new_result: ModelResult = new_model.fit()

    # Update the original model in place. Sync ALL mutable state so chained
    # updates — and predict(), which re-reads the formula — start from the
    # current fit, not the construction values.
    model._result  = new_result
    model._data    = target_data
    model._config  = new_config
    model._formula = new_formula
    model._priors  = new_priors
    model._group   = new_group

    return new_result


#: R-style alias.
update_hbm = update_model
