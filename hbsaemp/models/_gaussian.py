"""Gaussian (normal) HBSAE model — v0 stub."""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from hbsaemp.models._base import BaseModel, ModelResult
from hbsaemp._logging import get_logger

logger = get_logger(__name__)
__all__: list[str] = ["GaussianModel"]


class GaussianModel(BaseModel):
    """Gaussian hierarchical Bayesian model for continuous area-level estimates.

    Suitable for Fay-Herriot style models.  Default link: ``identity``.

    In v1, implemented via ``bambi.Model(formula, data, family="gaussian")``.

    Args:
        *args: Forwarded to :class:`~hbsaemp.models._base.BaseModel`.
        link: Link function. Default ``"identity"``.
        **kwargs: Forwarded to :class:`~hbsaemp.models._base.BaseModel`.
    """

    def __init__(self, *args: Any, link: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._link = link or "identity"

    def fit(self) -> ModelResult:
        """v0 stub — raises NotImplementedError."""
        raise NotImplementedError(
            "GaussianModel.fit() requires bambi>=0.14 (v1)."
        )

    def predict(self, new_data: pd.DataFrame | None = None, **kwargs: Any) -> np.ndarray:
        """v0 stub — raises NotImplementedError."""
        raise NotImplementedError(
            "GaussianModel.predict() requires bambi>=0.14 (v1)."
        )
