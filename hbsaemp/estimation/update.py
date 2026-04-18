"""Update a fitted HBSAE model with additional sampling.

Python equivalent of R hbsaems::update_hbm().

v0: update_model() stub.
v1: Continues sampling from an existing fitted model (bambi / PyMC resume).
"""
from __future__ import annotations
from hbsaemp._logging import get_logger
from hbsaemp.models._base import BaseModel, ModelResult
from hbsaemp.models._config import ModelConfig

logger = get_logger(__name__)
__all__: list[str] = ["update_model", "update_hbm"]


def update_model(
    model: BaseModel,
    *,
    new_data: object = None,
    config: ModelConfig | None = None,
    draws: int | None = None,
    tune: int | None = None,
    chains: int | None = None,
    cores: int | None = None,
) -> ModelResult:
    """Continue sampling from an existing fitted model.

    Python equivalent of ``update_hbm()`` in R hbsaems.

    Useful when convergence was not achieved in the initial fit:
    calling ``update_model(model, draws=4000)`` resumes sampling and
    effectively increases the total posterior draws.

    Args:
        model: A fitted :class:`~hbsaemp.models._base.BaseModel`.
        new_data: Optional new DataFrame. If provided, refits on new data.
        config: Full replacement :class:`~hbsaemp.models._config.ModelConfig`.
            When ``None``, individual override arguments are used.
        draws: Override draws per chain.
        tune: Override warmup steps.
        chains: Override number of chains.
        cores: Override number of cores.

    Returns:
        Updated :class:`~hbsaemp.models._base.ModelResult`.

    Raises:
        ModelNotFittedError: If *model* has not been fitted.
        NotImplementedError: In v0.
    """
    raise NotImplementedError(
        "update_model() requires a fitted model and bambi>=0.14 (v1)."
    )


#: R-style alias.
update_hbm = update_model
