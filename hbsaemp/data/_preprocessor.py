"""Data preprocessing for hbsaemp.

v0: DataPreprocessor stub.
v1: Concrete — missing value handling, type coercion, formula extraction.
"""
from __future__ import annotations
import pandas as pd
from hbsaemp._logging import get_logger

logger = get_logger(__name__)
__all__: list[str] = ["DataPreprocessor"]


class DataPreprocessor:
    """Preprocess a DataFrame before model fitting.

    v0: All methods raise NotImplementedError.
    v1: handle_missing="deleted" via pandas dropna; formula parsing via formulae.

    Args:
        handle_missing: ``"deleted"`` = complete-case analysis (v1).
    """

    def __init__(self, handle_missing: str = "deleted") -> None:
        self._handle_missing = handle_missing

    def process(
        self,
        data: pd.DataFrame,
        response: str,
        predictors: list[str],
        group: str | None = None,
    ) -> pd.DataFrame:
        """Preprocess *data* and return a cleaned copy.

        Args:
            data: Input DataFrame.
            response: Response column name.
            predictors: Predictor column names.
            group: Optional grouping column.

        Returns:
            Preprocessed DataFrame.

        Raises:
            NotImplementedError: In v0.
        """
        raise NotImplementedError("DataPreprocessor.process() requires v1.")

    def __repr__(self) -> str:
        return f"DataPreprocessor(handle_missing={self._handle_missing!r}, status=stub)"
