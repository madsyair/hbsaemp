"""Data preprocessing for hbsaemp: returns a transformed copy of the frame.

`DataPreprocessor.process()` adds family-specific offset columns (e.g.
`log_phi`, `log_sqrt_D`) and applies the missing-data strategy; it never
mutates the input.
"""
from __future__ import annotations

import pandas as pd

from hbsaemp._exceptions import DataValidationError
from hbsaemp._logging import get_logger
from hbsaemp.models._family_spec import FAMILY_SPECS

logger = get_logger(__name__)
__all__: list[str] = ["DataPreprocessor"]


class DataPreprocessor:
    """Preprocess a validated DataFrame before model fitting.

    Always operates on a copy — the original DataFrame is never modified.

    Transformations applied per family:

    * **Beta** — adds ``log_phi = log(n/deff - 1)`` column (when ``n_col``
      and ``deff_col`` are provided), used by Bambi's distributional formula
      ``"kappa ~ 1 + offset(log_phi)"``.  Smithson-Verkuilen squeeze
      ``(y*(n-1)+0.5)/n`` is applied **only** when ``squeeze=True`` (default
      ``False``); use it only for datasets with boundary values ``y=0``/``y=1``.

    * **Gaussian FH** — adds ``log_sqrt_D = 0.5 * log(D)`` column used by
      ``"sigma ~ 1 + offset(log_sqrt_D)"`` when ``sampling_var_col`` is
      provided.

    * **Binomial** — no additional transforms.

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

        # Family-specific transform reads straight from the single source
        # (FAMILY_SPECS). None means the family needs no offset transform.
        spec = FAMILY_SPECS[family]
        if spec.preprocess is not None:
            df = spec.preprocess(df, response, {
                "n_col": n_col,
                "deff_col": deff_col,
                "sampling_var_col": sampling_var_col,
                "trials_col": trials_col,
                "squeeze": squeeze,
            })

        logger.debug(
            "DataPreprocessor: %d rows ready (family=%r).", len(df), family
        )
        return df

    def __repr__(self) -> str:
        return f"DataPreprocessor(handle_missing={self._handle_missing!r})"
