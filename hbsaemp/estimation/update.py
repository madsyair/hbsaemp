"""Update a fitted HBSAE model with new config or data.

Python equivalent of R hbsaems::update_hbm().

v0: update_model() stub.
v1: Refit from scratch with a new :class:`~hbsaemp.models._config.ModelConfig`
    and / or new data.  This is a full refit (not resumed sampling); see the
    :func:`update_model` docstring for rationale.
"""
from __future__ import annotations

import dataclasses

import pandas as pd

from hbsaemp._logging import get_logger
from hbsaemp.models._base import BaseModel, ModelResult
from hbsaemp.models._config import ModelConfig
from hbsaemp.models._factory import _FAMILY_PARAMS

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
) -> ModelResult:
    """Refit a model with a new config and / or new data.

    Python equivalent of ``update_hbm()`` in R hbsaems.

    .. note::
        **v1 semantics: full refit, not resumed sampling.**
        This function creates a fresh model with the updated settings and
        calls ``fit()`` from scratch.  All posterior draws in the returned
        :class:`~hbsaemp.models._base.ModelResult` are new.  The original
        *model* object's ``_result`` is updated in-place so that subsequent
        ``model.predict()`` and ``model.result`` calls reflect the new fit.

    Useful when you want to:

    * Increase draws for better convergence:
      ``update_model(m, draws=4000)``
    * Refit on a refreshed dataset:
      ``update_model(m, new_data=new_df)``
    * Combine both:
      ``update_model(m, new_data=new_df, draws=2000, chains=4)``

    Args:
        model: A fitted :class:`~hbsaemp.models._base.BaseModel`.
        new_data: Optional new :class:`pandas.DataFrame`.  When provided,
            the model is refit on *new_data*.  When ``None``, the original
            training data is reused.
        config: Full replacement :class:`~hbsaemp.models._config.ModelConfig`.
            When supplied, *draws*, *tune*, *chains*, *cores* are ignored.
        draws: Override draws per chain.
        tune: Override warmup steps.
        chains: Override number of chains.
        cores: Override number of cores.

    Returns:
        A fresh :class:`~hbsaemp.models._base.ModelResult` from the refit.
        The *model* object is also updated so that ``model.result`` and
        ``model.predict()`` reflect the new fit.

    Raises:
        ModelNotFittedError: If *model* has not been fitted.
        TypeError: If *new_data* is not a :class:`pandas.DataFrame`.

    Examples::

        # More draws for convergence
        result = update_model(model, draws=4000)

        # Refit on new data
        result = update_model(model, new_data=refreshed_df, draws=2000)
    """
    # ── guard ─────────────────────────────────────────────────────────────────
    _ = model.result  # raises ModelNotFittedError if not fitted
    if new_data is not None and not isinstance(new_data, pd.DataFrame):
        raise TypeError(
            f"new_data must be a pandas DataFrame, got {type(new_data).__name__!r}."
        )

    # ── resolve config ────────────────────────────────────────────────────────
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
        new_config = dataclasses.replace(model._config, **overrides)

    # ── resolve data ──────────────────────────────────────────────────────────
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

    # ── build new model of same type ──────────────────────────────────────────
    # Family-aware dispatch via _FAMILY_PARAMS (single source of truth in
    # _factory.py): only forward the attrs the original model actually stores.
    # Each key in _FAMILY_PARAMS[family] mirrors a `self._<key>` attribute on
    # the corresponding subclass — see _factory._FAMILY_PARAMS docstring.
    family_attrs = {
        mp: getattr(model, f"_{mp}")
        for mp in _FAMILY_PARAMS[model._family]
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

    # ── refit ─────────────────────────────────────────────────────────────────
    new_result: ModelResult = new_model.fit()

    # ── update original model in-place so model.predict() uses new result ────
    # Sync ALL mutable state so chained updates start from the current fit,
    # not the original construction values.
    model._result = new_result
    model._data   = target_data
    model._config = new_config

    return new_result


#: R-style alias.
update_hbm = update_model
