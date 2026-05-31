"""Small Area Estimation: per-area mean/SD/HDI/RSE/MSE from posterior draws.

`estimate_areas()` always reads the posterior of the latent mean parameter
(via `predict(kind="response_params")`) — never the posterior predictive,
which would re-add sampling variance and destroy the shrinkage estimate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from hbsaemp._logging import get_logger
from hbsaemp.models._base import BaseModel

logger = get_logger(__name__)
__all__: list[str] = ["AreaEstimatesResult", "estimate_areas", "hbsae"]


@dataclass
class AreaEstimatesResult:
    """Per-area SAE results.

    Attributes:
        result_table: DataFrame with columns `mean`, `sd`, `ci_lower`,
            `ci_upper`, `rse_pct`, `mse`, `rmse`. An optional `group` column
            is prepended when the source model has grouping info.
        mean_rse: Mean RSE (%) across all areas.
        mean_mse: Mean MSE across all areas.
    """

    result_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    mean_rse: float | None = None
    mean_mse: float | None = None

    def summary(self) -> str:
        """One-screen text summary."""
        if self.result_table.empty:
            return "AreaEstimatesResult [stub — call estimate_areas() with a fitted model]"
        return (
            f"AreaEstimatesResult\n"
            f"  Areas     : {len(self.result_table)}\n"
            f"  Mean RSE% : {self.mean_rse:.2f}\n"
            f"  Mean MSE  : {self.mean_mse:.4f}\n"
            f"  Columns   : {list(self.result_table.columns)}"
        )

    def __repr__(self) -> str:
        n = len(self.result_table)
        return f"AreaEstimatesResult(n_areas={n}, mean_rse={self.mean_rse})"


def estimate_areas(
    model: BaseModel,
    *,
    new_data: pd.DataFrame | None = None,
    ci_prob: float = 0.95,
) -> AreaEstimatesResult:
    """Compute per-area HB estimates from the posterior of the latent parameter.

    For each area, draws are taken via `predict(kind="response_params")` —
    posterior of mu (Gaussian/Beta) or p (Binomial) — never the
    posterior predictive (which would re-add sampling variance and destroy
    shrinkage; for Binomial it would also return counts not probabilities).

    Metrics per area (from `(total_draws, n_obs)` array):
    `mean`, `sd`, `ci_lower`/`ci_upper` (HDI at `ci_prob`), `rse_pct =
    sd / |mean| * 100`, `mse` (posterior variance), `rmse`.

    A `group` column is prepended when the model was fitted with grouping.

    Args:
        model: Fitted `BaseModel`.
        new_data: Optional out-of-sample DataFrame. `None` uses training data.
        ci_prob: HDI probability (default 0.95).

    Raises:
        ModelNotFittedError: If `model` has not been fitted.
        ImportError: If `arviz` is not installed.
    """
    try:
        import arviz as az
    except ImportError as exc:
        raise ImportError(
            "estimate_areas() requires arviz>=1.1. "
            "Install with: pip install 'hbsaemp[bambi]'"
        ) from exc

    # guard: model must be fitted
    result = model.result  # raises ModelNotFittedError if not fitted

    group_col: str | None = result.extra.get("group")

    logger.info(
        "estimate_areas(): family=%r, new_data=%s, ci_prob=%s",
        result.family,
        "None (in-sample)" if new_data is None else f"shape={new_data.shape}",
        ci_prob,
    )

    # posterior draws of the latent mean/probability parameter
    # kind="response_params" extracts idata.posterior[model._mean_param_key]:
    #   Gaussian/Beta -> "mu"  (shrinkage estimate of theta_i)
    #   Binomial      -> "p"   (area-level success probability)
    # Using the posterior predictive (kind="response") would re-add the local
    # sampling variance D_i, inflating SD/MSE/RMSE and destroying shrinkage.
    # draws shape: (total_draws, n_obs)
    draws: np.ndarray = model.predict(new_data=new_data, kind="response_params")
    n_obs: int = draws.shape[1]

    # group labels
    # Group labels must be positionally aligned with the draws array columns.
    # result.data is already NaN-dropped (preprocessed during fit()).
    # new_data is raw — it must be preprocessed to drop the same NaN rows that
    # predict() drops internally, otherwise labels and draws are misaligned when
    # new_data contains missing values.
    group_labels: pd.Series | None = None
    if group_col is not None:
        if new_data is not None:
            source_for_labels: pd.DataFrame = model._preprocess_new_data(new_data)
        else:
            source_for_labels = result.data
        if group_col in source_for_labels.columns:
            group_labels = source_for_labels[group_col].reset_index(drop=True)

    # per-area statistics (vectorised over axis=0 = sample dim)
    means = draws.mean(axis=0)
    sds   = draws.std(axis=0)
    mses  = draws.var(axis=0)
    rmses = np.sqrt(mses)
    with np.errstate(divide="ignore", invalid="ignore"):
        rse_pct = np.where(
            means != 0.0,
            sds / np.abs(means) * 100.0,
            np.nan,
        )

    # HDI per area. Loop is the safe path: az.hdi has no vectorised 2-D
    # variant in ArviZ 1.1 (calling with axis= aggregates incorrectly).
    # `prob` is the canonical kwarg since ArviZ ≥0.20 (was `hdi_prob` before);
    # lower-bound ≥1.1 makes runtime detection unnecessary.
    hdi_arr = np.empty((n_obs, 2), dtype=float)
    for i in range(n_obs):
        hdi_arr[i] = az.hdi(draws[:, i], prob=ci_prob)

    df_result = pd.DataFrame({
        "mean":     means,
        "sd":       sds,
        "ci_lower": hdi_arr[:, 0],
        "ci_upper": hdi_arr[:, 1],
        "rse_pct":  rse_pct,
        "mse":      mses,
        "rmse":     rmses,
    })

    # Prepend group column when available
    if group_labels is not None:
        df_result.insert(0, group_col, group_labels.values[:n_obs])

    logger.debug(
        "estimate_areas(): produced %d rows, columns=%s",
        len(df_result), list(df_result.columns),
    )

    mean_rse = float(df_result["rse_pct"].mean())
    mean_mse = float(df_result["mse"].mean())

    return AreaEstimatesResult(
        result_table=df_result,
        mean_rse=mean_rse,
        mean_mse=mean_mse,
    )


#: R-style alias — ``hbsae(model)`` equals ``estimate_areas(model)``.
hbsae = estimate_areas
