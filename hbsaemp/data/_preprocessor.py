"""Data preprocessing for hbsaemp: returns a transformed copy of the frame.

`DataPreprocessor.process()` adds family-specific offset columns (e.g.
`log_phi`, `log_sqrt_D`) and applies the missing-data strategy; it never
mutates the input.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from hbsaemp._exceptions import DataValidationError
from hbsaemp._logging import get_logger
from hbsaemp.models._family_spec import (
    FAMILY_SPECS,
    USER_FIXED_COL,
    apply_link,
    pin_source_columns,
)

logger = get_logger(__name__)
__all__: list[str] = ["DataPreprocessor"]


class DataPreprocessor:
    """Preprocess a validated DataFrame before model fitting.

    Works on a copy; the original DataFrame is never modified.

    Transformations per family:

    * **Beta**: adds ``log_phi = log(n/deff - 1)`` when ``n_col`` and
      ``deff_col`` are given, used by the distributional formula
      ``"kappa ~ 1 + offset(log_phi)"``. The Smithson-Verkuilen squeeze
      ``(y*(n-1)+0.5)/n`` is applied only when ``squeeze=True`` (default
      ``False``), for data with boundary values ``y=0`` or ``y=1``.

    * **Gaussian FH**: adds ``log_sqrt_D = 0.5 * log(D)`` when
      ``sampling_var_col`` is given, used by
      ``"sigma ~ 1 + offset(log_sqrt_D)"``.

    * **Binomial**: no transformation.

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
        **pipeline_fields: Any,
    ) -> pd.DataFrame:
        """Preprocess *data* and return a cleaned, transformed copy.

        Args:
            data: Validated input DataFrame (from :class:`DataValidator`).
            response: Response column name.
            predictors: Predictor column names.
            group: Grouping column for random effects (``None`` = no RE).
            family: Distribution family; selects the transformations.
            **pipeline_fields: The family's own fields, as declared in its
                :class:`~hbsaemp.FamilySpec`: ``n_col``, ``deff_col`` and
                ``squeeze`` (Beta), ``sampling_var_col`` (Gaussian FH),
                ``trials_col`` (Binomial). String values are column names and
                are included in the missing-value check; other values are
                settings.

        Returns:
            Preprocessed :class:`pandas.DataFrame` ready to pass to Bambi.
            Contains all original columns plus any added transform columns.
        """
        # String-valued pipeline fields name data columns; other values are
        # settings. Derived rather than enumerated so a family can add a field
        # in FAMILY_SPECS.pipeline_fields without editing this signature. A
        # caller pin naming a column joins them, so a NaN there drops the row
        # exactly as a NaN in a survey-design column does — otherwise it would
        # survive into the offset as log(NaN) and reach the sampler.
        optional = [
            group,
            *(v for v in pipeline_fields.values() if isinstance(v, str)),
            *pin_source_columns(pipeline_fields.get("fixed_params")),
        ]
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
            df = spec.preprocess(df, response, pipeline_fields)

        # Caller-supplied pins: materialise one offset column per parameter.
        # The family's own pins are computed by `preprocess` above; these are
        # values the caller gave directly, so all that is left is to put them
        # on the parameter's link scale.
        for param, source in (pipeline_fields.get("fixed_params") or {}).items():
            values = (
                df[source].to_numpy(dtype=float)
                if isinstance(source, str)
                else np.full(len(df), float(source))
            )
            df[USER_FIXED_COL.format(param=param)] = apply_link(
                values, spec.pinnable_params[param]
            )

        logger.debug(
            "DataPreprocessor: %d rows ready (family=%r).", len(df), family
        )
        return df

    def __repr__(self) -> str:
        return f"DataPreprocessor(handle_missing={self._handle_missing!r})"
