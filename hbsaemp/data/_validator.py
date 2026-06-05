"""Read-only data validation: structural checks + per-family domain checks.

`DataValidator.validate()` raises `DataValidationError` on failure and returns
None; it never mutates the frame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hbsaemp._exceptions import DataValidationError
from hbsaemp._logging import get_logger
from hbsaemp.models._family_spec import FAMILY_SPECS

logger = get_logger(__name__)
__all__: list[str] = ["DataValidator"]


class DataValidator:
    """Validate a DataFrame before model fitting.

    Concrete checks — types, NaN, and family-specific domains.

    Args:
        handle_missing: Strategy for NaN values. ``"deleted"`` = dropna (v1).
    """

    def __init__(self, handle_missing: str = "deleted") -> None:
        if handle_missing != "deleted":
            raise NotImplementedError(
                f"handle_missing={handle_missing!r} is not supported. "
                "Only 'deleted' is implemented in v1."
            )
        self._handle_missing = handle_missing

    # Public API
    def validate(
        self,
        data: pd.DataFrame,
        response: str,
        predictors: list[str],
        *,
        family: str,
        group: str | None = None,
        n_col: str | None = None,
        deff_col: str | None = None,
        sampling_var_col: str | None = None,
        trials_col: str | None = None,
        squeeze: bool = False,
    ) -> None:
        """Validate *data* in place; raises on failure, otherwise returns ``None``.

        Read-only — *data* is never mutated.  Caller owns any copies needed
        (the downstream :class:`DataPreprocessor` already takes a fresh copy).

        Checks (in order):

        1. *data* is a non-empty :class:`pandas.DataFrame`.
        2. *family* is supported.
        3. All required columns exist.
        4. Response, predictor, and auxiliary columns are numeric and finite.
        5. Family-specific domain constraints.

        Args:
            data: Input DataFrame.
            response: Response column name.
            predictors: Predictor column names.
            family: Distribution family for domain checks.
            group: Optional grouping column (non-numeric allowed).
            n_col: *Beta* — survey sample-size column.
            deff_col: *Beta* — design-effect column.
            sampling_var_col: *Gaussian FH* — sampling-variance column.
            trials_col: *Binomial* — number-of-trials column.
            squeeze: *Beta* — when ``True``, relax the response domain check
                to ``[0, 1]`` (closed) instead of the default ``(0, 1)``
                (strict), since the Smithson-Verkuilen transform will map
                boundary values into the open interval before fitting.

        Returns:
            ``None``.  All failures surface as :class:`DataValidationError`.

        Raises:
            DataValidationError: On any structural or domain failure.
        """
        if not isinstance(data, pd.DataFrame):
            raise DataValidationError(
                f"`data` must be a pandas DataFrame, got {type(data).__name__!r}."
            )
        if data.empty:
            raise DataValidationError("`data` is empty.")
        if family not in FAMILY_SPECS:
            raise DataValidationError(
                f"Unknown family {family!r}. Supported: {sorted(FAMILY_SPECS)}."
            )

        aux_cols = [
            c for c in (n_col, deff_col, sampling_var_col, trials_col)
            if c is not None
        ]
        all_cols = [response, *predictors, *aux_cols, *([group] if group else [])]

        self._check_columns_exist(data, all_cols)
        self._check_numeric(data, [response, *predictors, *aux_cols])

        # Family-specific domain check reads straight from the single source
        # (FAMILY_SPECS). None means the family declares no extra check.
        spec = FAMILY_SPECS[family]
        if spec.response_check is not None:
            spec.response_check(data, response, {
                "n_col": n_col,
                "deff_col": deff_col,
                "sampling_var_col": sampling_var_col,
                "trials_col": trials_col,
                "squeeze": squeeze,
            })

        logger.debug("DataValidator: passed (family=%r, n=%d)", family, len(data))

    # Structural & domain checks
    @staticmethod
    def _check_columns_exist(data: pd.DataFrame, cols: list[str]) -> None:
        missing = [c for c in cols if c not in data.columns]
        if missing:
            raise DataValidationError(
                f"Column(s) not found in data: {missing}. "
                f"Available: {list(data.columns)}.",
                context={"missing_columns": missing},
            )

    @staticmethod
    def _check_numeric(data: pd.DataFrame, cols: list[str]) -> None:
        non_numeric = [
            c for c in cols if not pd.api.types.is_numeric_dtype(data[c])
        ]
        if non_numeric:
            raise DataValidationError(
                f"Non-numeric column(s): {non_numeric}. "
                "Response, predictors, and auxiliary columns must be numeric.",
                context={"non_numeric_columns": non_numeric},
            )
        # np.inf passes is_numeric_dtype (it is a float) but causes NaN gradients
        # in MCMC samplers. Reject explicitly after the dtype check.
        inf_cols = [
            c for c in cols
            if not np.isfinite(data[c].dropna().to_numpy(dtype=float)).all()
        ]
        if inf_cols:
            raise DataValidationError(
                f"Infinite values (±inf) found in column(s): {inf_cols}. "
                "All numeric columns must contain finite values only.",
                context={"inf_columns": inf_cols},
            )

    def __repr__(self) -> str:
        return f"DataValidator(handle_missing={self._handle_missing!r})"
