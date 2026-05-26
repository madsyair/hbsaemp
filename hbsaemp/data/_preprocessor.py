"""Data preprocessing for hbsaemp.

v0: DataPreprocessor stub.
v1: Concrete — missing value handling, type coercion, formula extraction.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from hbsaemp._exceptions import DataValidationError
from hbsaemp._logging import get_logger

logger = get_logger(__name__)
__all__: list[str] = ["DataPreprocessor"]


class DataPreprocessor:
    """Preprocess a validated DataFrame before model fitting.

    Always operates on a copy — the original DataFrame is never modified.

    Transformations applied per family:

    * **Beta** — adds ``log_phi = log(n/deff - 1)`` column (when ``n_col``
      and ``deff_col`` are provided), used by Bambi's distributional formula
      ``"kappa ~ 0 + offset(log_phi)"``.  Smithson-Verkuilen squeeze
      ``(y*(n-1)+0.5)/n`` is applied **only** when ``squeeze=True`` (default
      ``False``); use it only for datasets with boundary values ``y=0``/``y=1``.

    * **Gaussian FH** — adds ``log_sqrt_D = 0.5 * log(D)`` column used by
      ``"sigma ~ 0 + offset(log_sqrt_D)"`` when ``sampling_var_col`` is
      provided.

    * **Lognormal / Binomial** — no additional transforms.

    Args:
        handle_missing: ``"deleted"`` (only supported value in v1) drops rows
            with NaN in any relevant column via
            :meth:`pandas.DataFrame.dropna`.
    """

    def __init__(self, handle_missing: str = "deleted") -> None:
        if handle_missing != "deleted":
            raise NotImplementedError(
                f"handle_missing={handle_missing!r} is not supported. "
                "Only 'deleted' is implemented in v1."
            )
        self._handle_missing = handle_missing

    # Public API
    def process(
        self,
        data: pd.DataFrame,
        response: str,
        predictors: list[str],
        *,
        group: str | None = None,
        family: str,
        n_col: str | None = None,
        deff_col: str | None = None,
        sampling_var_col: str | None = None,
        trials_col: str | None = None,
        squeeze: bool = False,
    ) -> pd.DataFrame:
        """Preprocess *data* and return a cleaned, transformed copy.

        Args:
            data: Validated input DataFrame (from :class:`DataValidator`).
            response: Response column name.
            predictors: Predictor column names.
            group: Grouping column for random effects (``None`` = no RE).
            family: Distribution family — determines which transforms apply.
            n_col: *Beta* — sample-size column for phi computation.
            deff_col: *Beta* — design-effect column for phi computation.
            sampling_var_col: *Gaussian FH* — sampling-variance column ``D``.
            trials_col: *Binomial* — number-of-trials column.
            squeeze: *Beta* — apply Smithson-Verkuilen squeeze
                ``(y*(n-1)+0.5)/n`` to the response before fitting.
                Default ``False``.  Set ``True`` only when the dataset
                contains boundary values ``y=0`` or ``y=1`` (requires
                ``n_col`` and ``deff_col`` to be provided).

        Returns:
            Preprocessed :class:`pandas.DataFrame` ready to pass to Bambi.
            Contains all original columns plus any added transform columns.
        """
        optional = [group, n_col, deff_col, sampling_var_col, trials_col]
        relevant_cols = [response, *predictors, *[c for c in optional if c is not None]]

        df = data.copy()

        n_before = len(df)
        df = df.dropna(subset=relevant_cols).reset_index(drop=True)
        n_dropped = n_before - len(df)
        if n_dropped:
            logger.info(
                "DataPreprocessor: dropped %d row(s) with NaN "
                "(handle_missing='deleted').",
                n_dropped,
            )
        if df.empty:
            raise DataValidationError(
                f"All {n_before} row(s) were dropped due to missing values "
                f"in the relevant columns: {relevant_cols}. "
                "Ensure the DataFrame has at least one complete row across "
                "the response, predictors, and any auxiliary columns.",
                context={"relevant_cols": relevant_cols, "n_original": n_before},
            )

        df = getattr(self, f"_transform_{family}")(
            df, response,
            n_col=n_col,
            deff_col=deff_col,
            sampling_var_col=sampling_var_col,
            trials_col=trials_col,
            squeeze=squeeze,
        )

        logger.debug(
            "DataPreprocessor: %d rows ready (family=%r).", len(df), family
        )
        return df
    
    # ── Family-specific transforms ────────────────────────────────────────────

    def _transform_beta(
        self,
        df: pd.DataFrame,
        response: str,
        *,
        n_col: str | None,
        deff_col: str | None,
        squeeze: bool = False,
        **_,
    ) -> pd.DataFrame:
        """Add ``log_phi`` column; optionally apply Smithson-Verkuilen squeeze.

        Only active when both *n_col* and *deff_col* are provided.

        ``log_phi = log(n/deff - 1)`` is always computed when the precision
        is pinned from data — it is required by the Bambi distributional
        formula ``"kappa ~ 0 + offset(log_phi)"``.

        Smithson-Verkuilen squeeze ``(y*(n-1)+0.5)/n`` is applied **only**
        when *squeeze* is ``True``.  Use this only when the dataset contains
        boundary values ``y=0`` or ``y=1``; the default (``False``) fits on
        the original direct-estimate proportions without distortion.
        """
        if n_col is not None and deff_col is not None:
            n_vals = df[n_col].to_numpy(dtype=float)
            deff_vals = df[deff_col].to_numpy(dtype=float)

            if squeeze:
                y = df[response].to_numpy(dtype=float)
                df[response] = (y * (n_vals - 1) + 0.5) / n_vals
                logger.debug(
                    "Beta: Smithson-Verkuilen squeeze applied (squeeze=True)."
                )

            df["log_phi"] = np.log(n_vals / deff_vals - 1)
            logger.debug(
                "Beta: log_phi in [%.3f, %.3f].",
                df["log_phi"].min(), df["log_phi"].max(),
            )
        return df

    def _transform_gaussian(
        self,
        df: pd.DataFrame,
        response: str,
        *,
        sampling_var_col: str | None,
        **_,
    ) -> pd.DataFrame:
        """Add ``log_sqrt_D`` column for Gaussian Fay-Herriot."""
        if sampling_var_col is not None:
            D = df[sampling_var_col].to_numpy(dtype=float)
            df["log_sqrt_D"] = 0.5 * np.log(D)
            logger.debug(
                "Gaussian FH: log_sqrt_D in [%.3f, %.3f].",
                df["log_sqrt_D"].min(), df["log_sqrt_D"].max(),
            )
        return df

    def _transform_lognormal(self, df: pd.DataFrame, response: str, **_) -> pd.DataFrame:
        """No-op when DataPreprocessor is called directly with ``family="lognormal"``.

        :class:`~hbsaemp.models._lognormal.LognormalModel` overrides
        ``_preproc_family`` to ``"gaussian"`` so the FH offset preprocessing
        (``log_sqrt_D``) runs via :meth:`_transform_gaussian`.  This method
        exists so the low-level public API ``DataPreprocessor().process(...,
        family="lognormal")`` does not fall through to ``getattr`` errors.
        """
        return df

    def _transform_binomial(self, df: pd.DataFrame, response: str, **_) -> pd.DataFrame:
        return df

    def __repr__(self) -> str:
        return f"DataPreprocessor(handle_missing={self._handle_missing!r})"
