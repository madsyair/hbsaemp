"""Data validation for hbsaemp.

v0: DataValidator stub.
v1: Concrete implementation using pandas/numpy checks.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from hbsaemp._exceptions import DataValidationError
from hbsaemp._logging import get_logger

logger = get_logger(__name__)
__all__: list[str] = ["DataValidator"]


class DataValidator:
    """Validate a DataFrame before model fitting.

    v0: All methods raise NotImplementedError.
    v1: Concrete checks — types, NaN, and family-specific domains.

    Args:
        handle_missing: Strategy for NaN values. ``"deleted"`` = dropna (v1).
    """

    _SUPPORTED_FAMILIES: frozenset[str] = frozenset(
        {"beta", "gaussian", "lognormal", "binomial"}
    )
    
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
        if family not in self._SUPPORTED_FAMILIES:
            raise DataValidationError(
                f"Unknown family {family!r}. "
                f"Supported: {sorted(self._SUPPORTED_FAMILIES)}."
            )

        aux_cols = [
            c for c in (n_col, deff_col, sampling_var_col, trials_col)
            if c is not None
        ]
        all_cols = [response, *predictors, *aux_cols, *([group] if group else [])]

        self._check_columns_exist(data, all_cols)
        self._check_numeric(data, [response, *predictors, *aux_cols])

        getattr(self, f"_check_{family}")(
            data, response,
            n_col=n_col,
            deff_col=deff_col,
            sampling_var_col=sampling_var_col,
            trials_col=trials_col,
            squeeze=squeeze,
        )

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
    
    # Family-specific checks
    def _check_beta(
        self,
        data: pd.DataFrame,
        response: str,
        *,
        n_col: str | None,
        deff_col: str | None,
        squeeze: bool = False,
        **_,
    ) -> None:
        y = data[response].dropna()
        if squeeze:
            # Smithson-Verkuilen will map [0,1] → (0,1); allow closed interval.
            n_bad = int(((y < 0) | (y > 1)).sum())
            domain_msg = "[0, 1]"
        else:
            # No squeeze: Bambi Beta requires strictly open (0, 1).
            n_bad = int(((y <= 0) | (y >= 1)).sum())
            domain_msg = "(0, 1)"
        if n_bad:
            raise DataValidationError(
                f"Beta family requires response in {domain_msg}. "
                f"Found {n_bad} value(s) outside this range in {response!r}. "
                + ("" if squeeze else "Pass squeeze=True to handle boundary values 0 and 1."),
                column=response,
                context={"n_boundary_values": n_bad},
            )

        if n_col is not None and deff_col is not None:
            phi_df = data[[n_col, deff_col]].dropna()
            if (phi_df[n_col] <= 0).any():
                raise DataValidationError(
                    f"{n_col!r}: sample sizes must be positive.", column=n_col
                )
            if (phi_df[deff_col] <= 0).any():
                raise DataValidationError(
                    f"{deff_col!r}: design effects must be positive.", column=deff_col
                )
            phi = phi_df[n_col].values / phi_df[deff_col].values - 1
            n_bad_phi = int((phi <= 0).sum())
            if n_bad_phi:
                raise DataValidationError(
                    f"Beta precision phi = n/deff - 1 must be > 0. "
                    f"Found {n_bad_phi} row(s) where n/deff ≤ 1.",
                    context={"n_invalid_phi": n_bad_phi},
                )

    def _check_gaussian(
        self,
        data: pd.DataFrame,
        response: str,
        *,
        sampling_var_col: str | None,
        **_,
    ) -> None:
        if sampling_var_col is not None:
            D = data[sampling_var_col].dropna()
            n_bad = int((D <= 0).sum())
            if n_bad:
                raise DataValidationError(
                    f"Gaussian FH: {sampling_var_col!r} must be positive. "
                    f"Found {n_bad} non-positive value(s).",
                    column=sampling_var_col,
                    context={"n_nonpositive": n_bad},
                )

    def _check_lognormal(self, data: pd.DataFrame, response: str, **_) -> None:
        """Validate positive original-scale responses for the planned V2 family."""
        y = data[response].dropna()
        n_bad = int((y <= 0).sum())
        if n_bad:
            raise DataValidationError(
                f"Lognormal family requires response > 0. "
                f"Found {n_bad} non-positive value(s) in {response!r}.",
                column=response,
                context={"n_nonpositive": n_bad},
            )

    def _check_binomial(
        self,
        data: pd.DataFrame,
        response: str,
        *,
        trials_col: str | None,
        **_,
    ) -> None:
        y = data[response].dropna()
        if (y < 0).any():
            raise DataValidationError(
                f"Binomial family: {response!r} must be non-negative.", column=response
            )
        if not (y % 1 == 0).all():
            raise DataValidationError(
                f"Binomial family: {response!r} must contain integers.", column=response
            )
        if trials_col is not None:
            check = data[[response, trials_col]].dropna()
            n = check[trials_col]
            if (n < 1).any() or not (n % 1 == 0).all():
                raise DataValidationError(
                    f"Binomial family: {trials_col!r} must contain positive integers.",
                    column=trials_col,
                )
            n_bad = int((check[response].values > check[trials_col].values).sum())
            if n_bad:
                raise DataValidationError(
                    f"Binomial family: {response!r} must not exceed {trials_col!r}. "
                    f"Found {n_bad} row(s) where y > n.",
                    column=response,
                    context={"n_exceeding_trials": n_bad},
                )

    def __repr__(self) -> str:
        return f"DataValidator(handle_missing={self._handle_missing!r})"
