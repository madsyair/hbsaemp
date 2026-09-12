"""Small Area Estimation: per-area mean/SD/HDI/RSE/MSE from posterior draws.

`estimate_areas()` always reads the posterior of the latent mean parameter
(via `predict(kind="response_params")`) — never the posterior predictive,
which would re-add sampling variance and destroy the shrinkage estimate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from hbsaemp._exceptions import EstimationError, HBSAEError
from hbsaemp._logging import get_logger
from hbsaemp.models._base import BaseModel

logger = get_logger(__name__)
__all__: list[str] = ["AreaEstimatesResult", "estimate_areas", "hbsae"]


def _fmt(value: float | None, spec: str) -> str:
    """Format *value* with *spec*, or `"n/a"` when it is missing or NaN."""
    if value is None or not math.isfinite(value):
        return "n/a"
    return format(value, spec)


@dataclass
class AreaEstimatesResult:
    """Per-area SAE results.

    Attributes:
        result_table: DataFrame with columns `mean`, `sd`, `ci_lower`,
            `ci_upper`, `rse_pct`, `mse`, `rmse`. An optional `group` column
            is prepended when the source model has grouping info.
        mean_rse: Mean RSE (%) across areas. Areas whose `rse_pct` is NaN
            (posterior mean exactly 0) are excluded from the average.
        mean_mse: Mean MSE across all areas.
    """

    result_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    mean_rse: float | None = None
    mean_mse: float | None = None

    def summary(self) -> str:
        """One-screen text summary."""
        if self.result_table.empty:
            return (
                "AreaEstimatesResult [empty — not fitted; "
                "call estimate_areas() on a fitted model]"
            )
        return (
            f"AreaEstimatesResult\n"
            f"  Areas     : {len(self.result_table)}\n"
            f"  Mean RSE% : {_fmt(self.mean_rse, '.2f')}\n"
            f"  Mean MSE  : {_fmt(self.mean_mse, '.4f')}\n"
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

    Each row is one area (area-level data). Unit-level rows are summarised
    per observation, not aggregated per area; a warning is logged when group
    labels repeat. A `group` column is prepended when the model was fitted
    with grouping.

    Out-of-sample rows need only the predictor and group columns: the
    response and the survey-design columns (`sampling_var`, `n`/`deff`,
    `trials`) do not enter the mean parameter. A row whose group label was
    not seen during fitting is a non-sampled area. At each posterior draw its
    area effect is taken from a randomly chosen fitted area (Bambi
    `sample_new_groups=True`, equivalent to brms
    `sample_new_levels="uncertainty"`), so its interval is wider than a
    sampled area's. These draws are seeded by `config.random_seed`.

    Args:
        model: Fitted `BaseModel`.
        new_data: Optional out-of-sample DataFrame. `None` uses training data.
        ci_prob: HDI probability in (0, 1) (default 0.95).

    Raises:
        ValueError: If `ci_prob` is not in (0, 1).
        ModelNotFittedError: If `model` has not been fitted.
        ImportError: If `arviz` is not installed.
        DataValidationError: If `new_data` lacks a predictor or group column,
            or no row is complete in those columns.
        EstimationError: If the posterior draws, the HDI, or the result table
            cannot be computed.
    """
    # NaN fails both comparisons, so it is rejected here too.
    if not 0.0 < ci_prob < 1.0:
        raise ValueError(f"ci_prob must be in (0, 1); got {ci_prob!r}.")

    try:
        import arviz as az
    except ImportError as exc:
        raise ImportError(
            "estimate_areas() requires arviz>=1.2. "
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
    # Everything from here is the estimation proper. Failures inside it are
    # wrapped as EstimationError so callers (and the GUI) can distinguish
    # "this model/data combination cannot be estimated" from the guards above:
    # ModelNotFittedError (wrong call order) and ImportError (missing dep).
    try:
        draws: np.ndarray = model.predict(
            new_data=new_data, kind="response_params", sample_new_groups=True
        )
        n_obs: int = draws.shape[1]

        # Group labels must be positionally aligned with the draws columns.
        # In-sample, predict() reads result.data (the NaN-dropped fit frame).
        # For new_data, predict() runs _prepare_mean_data(); it is
        # deterministic, so re-running it here yields the same rows in the
        # same order. The length check turns any future drift into an error
        # instead of silently shifted labels.
        label_source: pd.DataFrame = (
            result.data if new_data is None else model._prepare_mean_data(new_data)
        )
        if len(label_source) != n_obs:
            raise EstimationError(
                f"estimate_areas(): {len(label_source)} label row(s) but "
                f"{n_obs} draw column(s); area labels cannot be aligned.",
                context={"n_labels": len(label_source), "n_obs": n_obs},
            )

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

        # HDI per area. az.hdi on a bare ndarray reduces over the LAST axis,
        # so passing the (draws, areas) matrix would summarise across areas
        # for each draw. The per-area loop keeps the reduction axis explicit.
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
        if group_col is not None and group_col in label_source.columns:
            labels = label_source[group_col]
            df_result.insert(0, group_col, labels.to_numpy())
            if labels.duplicated().any():
                logger.warning(
                    "estimate_areas(): labels in %r repeat — rows are "
                    "summarised per observation, not aggregated per area.",
                    group_col,
                )

    except HBSAEError:
        # Package errors already carry their own contract (e.g. the
        # DataValidationError _prepare_mean_data raises). Do not reclassify.
        raise
    except Exception as exc:
        raise EstimationError(
            f"estimate_areas() failed for family={result.family!r} "
            f"({'in-sample' if new_data is None else f'new_data shape={new_data.shape}'}, "
            f"ci_prob={ci_prob}): {exc}"
        ) from exc

    logger.debug(
        "estimate_areas(): produced %d rows, columns=%s",
        len(df_result), list(df_result.columns),
    )

    n_nan_rse = int(df_result["rse_pct"].isna().sum())
    if n_nan_rse:
        logger.warning(
            "estimate_areas(): %d area(s) have posterior mean 0, so rse_pct "
            "is NaN; they are excluded from mean_rse.",
            n_nan_rse,
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
