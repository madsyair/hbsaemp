"""`update_model` (alias `update_hbm`) — full refit with new data/config.

This is a complete refit, not resumed sampling: all posterior draws in the
returned `ModelResult` are new. The source model is updated in place so
subsequent `model.predict()` / `model.result` reflect the new fit.
"""
from __future__ import annotations

import dataclasses

import pandas as pd

from hbsaemp._logging import get_logger
from hbsaemp.models._base import BaseModel, ModelResult
from hbsaemp.models._config import ModelConfig
from hbsaemp.models._family_spec import FAMILY_SPECS

logger = get_logger(__name__)
__all__: list[str] = ["update_model", "update_hbm"]


def update_model(
    model: BaseModel,
    *,
    new_data: pd.DataFrame | None = None,
    config: ModelConfig | None = None,
    draws: int | None = None,
    tune: int | None = None,
    chains: int | None = None,
    cores: int | None = None,
    target_accept: float | None = None,
    random_seed: int | None = None,
) -> ModelResult:
    """Full refit with optional new data and/or sampler overrides.

    Either pass a full replacement `config`, or individual overrides
    (`draws`, `tune`, `chains`, `cores`) applied via `dataclasses.replace`
    on the model's current config (ignored when `config` is supplied).
    Updates *model* in place — its `.result`/`.data`/`.config` reflect the new
    fit, so chained `update_model()` calls start from current state.

    Args:
        model: Fitted `BaseModel`.
        new_data: Optional new DataFrame; `None` reuses the original training
            data.
        config: Full replacement `ModelConfig`.
        draws: Override the number of posterior draws per chain.
        tune: Override the number of tuning (warmup) iterations.
        chains: Override the number of MCMC chains.
        cores: Override the number of parallel sampling cores.
        target_accept: Override the NUTS target acceptance rate (raise toward
            0.95-0.99 to clear divergences). Validated to ``(0, 1)`` by
            `ModelConfig.__post_init__` before refitting.
        random_seed: Override the random seed for a reproducible refit.

    Raises:
        ModelNotFittedError: If `model` has not been fitted.
        TypeError: If `new_data` is supplied but not a DataFrame.
    """
    # guard
    _ = model.result  # raises ModelNotFittedError if not fitted
    if new_data is not None and not isinstance(new_data, pd.DataFrame):
        raise TypeError(
            f"new_data must be a pandas DataFrame, got {type(new_data).__name__!r}."
        )

    # resolve config
    if config is not None:
        new_config: ModelConfig = config
    else:
        overrides: dict = {}
        if draws is not None:
            overrides["draws"] = draws
        if tune is not None:
            overrides["tune"] = tune
        if chains is not None:
            overrides["chains"] = chains
        if cores is not None:
            overrides["cores"] = cores
        if target_accept is not None:
            overrides["target_accept"] = target_accept
        if random_seed is not None:
            overrides["random_seed"] = random_seed
        new_config = dataclasses.replace(model._config, **overrides)

    # resolve data
    target_data: pd.DataFrame = (
        new_data if new_data is not None else model._data
    )

    logger.info(
        "update_model(): family=%r, draws=%d, chains=%d, new_data=%s",
        model._family,
        new_config.draws,
        new_config.chains,
        "None (reuse original)" if new_data is None else f"shape={new_data.shape}",
    )

    # Rebuild the same subclass, forwarding only its family attrs.
    # FAMILY_SPECS[family].user_params keys mirror `self._<key>` on the model.
    family_attrs = {
        mp: getattr(model, f"_{mp}")
        for mp in FAMILY_SPECS[model._family].user_params
    }
    new_model: BaseModel = type(model)(
        model._formula,
        model._family,
        target_data,
        new_config,
        priors=model._priors,
        group=model._group,
        handle_missing=model._handle_missing,
        **family_attrs,
    )

    # refit
    new_result: ModelResult = new_model.fit()

    # update original model in-place so model.predict() uses new result
    # Sync ALL mutable state so chained updates start from the current fit,
    # not the original construction values.
    model._result = new_result
    model._data   = target_data
    model._config = new_config

    return new_result


#: R-style alias.
update_hbm = update_model
