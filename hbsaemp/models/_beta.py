"""Beta logit-normal HBSAE model — v0 stub."""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from hbsaemp.models._base import BaseModel, ModelResult
from hbsaemp._logging import get_logger

logger = get_logger(__name__)
__all__: list[str] = ["BetaModel"]


class BetaModel(BaseModel):
    """Beta distribution HBSAE model for proportional estimates in (0, 1).

    When ``n_col`` and ``deff_col`` are provided, the precision parameter is
    :math:`\\phi_i = n_i / \\text{deff}_i - 1` (not estimated).
    Default link: ``logit``.

    In v1, implemented via ``bambi.Model(formula, data, family="beta")``.

    Args:
        *args: Forwarded to :class:`~hbsaemp.models._base.BaseModel`.
        n_col: Column name for survey sample size (optional).
        deff_col: Column name for design effect (optional).
        link: Link function. Default ``"logit"``.
        **kwargs: Forwarded to :class:`~hbsaemp.models._base.BaseModel`.
    """

    def __init__(
        self, *args: Any,
        n_col: str | None = None,
        deff_col: str | None = None,
        link: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._n_col = n_col
        self._deff_col = deff_col
        self._link = link or "logit"

    def fit(self) -> ModelResult:
        """v0 stub — raises NotImplementedError."""
        raise NotImplementedError(
            "BetaModel.fit() requires bambi>=0.14 (v1)."
        )

    def predict(self, new_data: pd.DataFrame | None = None, **kwargs: Any) -> np.ndarray:
        """v0 stub — raises NotImplementedError."""
        raise NotImplementedError(
            "BetaModel.predict() requires bambi>=0.14 (v1)."
        )
