"""Binomial logit-normal HBSAE model — v0 stub."""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from hbsaemp.models._base import BaseModel, ModelResult
from hbsaemp._logging import get_logger

logger = get_logger(__name__)
__all__: list[str] = ["BinomialModel"]


class BinomialModel(BaseModel):
    """Binomial HBSAE model for count outcomes.

    Response is number of successes; ``trials_col`` gives the trial counts.
    Default link: ``logit``.

    In v1, implemented via ``bambi.Model(formula, data, family="binomial")``.

    Args:
        *args: Forwarded to :class:`~hbsaemp.models._base.BaseModel`.
        trials_col: Column name for number of trials (required).
        link: Link function. Default ``"logit"``.
        **kwargs: Forwarded to :class:`~hbsaemp.models._base.BaseModel`.
    """

    def __init__(
        self, *args: Any,
        trials_col: str | None = None,
        link: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._trials_col = trials_col
        self._link = link or "logit"

    def fit(self) -> ModelResult:
        """v0 stub — raises NotImplementedError."""
        raise NotImplementedError(
            "BinomialModel.fit() requires bambi>=0.14 (v1)."
        )

    def predict(self, new_data: pd.DataFrame | None = None, **kwargs: Any) -> np.ndarray:
        """v0 stub — raises NotImplementedError."""
        raise NotImplementedError(
            "BinomialModel.predict() requires bambi>=0.14 (v1)."
        )
