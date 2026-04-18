"""Data validation for hbsaemp.

v0: DataValidator stub.
v1: Concrete implementation using pandas/numpy checks.
"""
from __future__ import annotations
from typing import Any
import pandas as pd
from hbsaemp._logging import get_logger

logger = get_logger(__name__)
__all__: list[str] = ["DataValidator"]


class DataValidator:
    """Validate a DataFrame before model fitting.

    v0: All methods raise NotImplementedError.
    v1: Concrete checks — types, NaN, domain (y in (0,1) for beta, y>0 for lognormal, etc.).

    Args:
        handle_missing: Strategy for NaN values. ``"deleted"`` = dropna (v1).
    """

    def __init__(self, handle_missing: str = "deleted") -> None:
        self._handle_missing = handle_missing

    def validate(
        self,
        data: pd.DataFrame,
        response: str,
        predictors: list[str],
        *,
        family: str,
        group: str | None = None,
    ) -> pd.DataFrame:
        """Validate *data* and return a cleaned copy.

        Args:
            data: Input DataFrame.
            response: Response column name.
            predictors: Predictor column names.
            family: Distribution family for domain checks.
            group: Optional grouping column.

        Returns:
            Validated (and possibly cleaned) DataFrame.

        Raises:
            NotImplementedError: In v0.
            DataValidationError: When data fails any check (v1).
        """
        raise NotImplementedError("DataValidator.validate() requires v1.")

    def __repr__(self) -> str:
        return f"DataValidator(handle_missing={self._handle_missing!r}, status=stub)"
