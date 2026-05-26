"""Small Area Estimation prediction and uncertainty quantification.

Python equivalent of R hbsaems::hbsae().

v0: estimate_areas() stub + AreaEstimatesResult dataclass.
v1: Concrete — model.predict() via bambi; RSE/MSE/RMSE/CI from posterior draws.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from hbsaemp._logging import get_logger
from hbsaemp.models._base import BaseModel

logger = get_logger(__name__)
__all__: list[str] = ["AreaEstimatesResult", "estimate_areas", "hbsae"]


@lru_cache(maxsize=1)
def _get_hdi_kwarg() -> str:
    """Return the active kwarg name for ``arviz.hdi`` probability.

    ArviZ ≥0.20 renamed ``hdi_prob`` → ``prob``.  Detect once per process
    via :func:`inspect.signature` and cache; preserves the lazy arviz
    import (signature inspection happens at first call, not module load).
    """
    from inspect import signature

    import arviz as az

    return "prob" if "prob" in signature(az.hdi).parameters else "hdi_prob"


@dataclass
class AreaEstimatesResult:
    """Container for small area estimation results.

    Attributes:
        result_table: DataFrame with per-area statistics.
            Columns: ``mean``, ``sd``, ``ci_lower``, ``ci_upper``,
            ``rse_pct``, ``mse``, ``rmse``.
            Optional ``group`` column when grouping info is available.
        mean_rse: Overall mean RSE (%) across all areas.
        mean_mse: Overall mean MSE across all areas.

    Example (v1 output)::

             mean        sd  ci_lower  ci_upper  rse_pct       mse      rmse
        0  5.1418  0.424869  4.308978  5.974702   8.2666  0.180594  0.424963
        1  4.2751  0.369247  3.551182  4.998952   8.6380  0.136343  0.369246
    """

    result_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    mean_rse: float | None = None
    mean_mse: float | None = None

    def summary(self) -> str:
        """Human-readable summary of SAE results."""
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
    """Generate hierarchical Bayesian small area estimates.

    Python equivalent of ``hbsae()`` in R hbsaems.

    Computes per-area statistics from the **posterior distribution of the
    latent mean/probability parameter** (not the posterior predictive):

    * Gaussian / Beta / Lognormal (log-scale): posterior of ``μ_i`` —
      the shrinkage estimate ``E[θ_i | y]``.
    * Binomial: posterior of ``p_i`` — the area-level success probability
      in ``(0, 1)``.

    Using the posterior of the latent parameter (rather than the posterior
    predictive ``y*``) is the correct choice for SAE because:

    1. The posterior predictive ``Var[y*|y] = Var[θ_i|y] + D_i`` re-adds
       the local sampling variance ``D_i``, destroying the shrinkage effect
       that is the core benefit of hierarchical Bayes models.
    2. For Binomial, the posterior predictive returns *counts* (``y* ∈
       {0, …, n_i}``), whereas the SAE target is the *proportion* ``p_i``.

    The posterior predictive is used only in
    :func:`~hbsaemp.diagnostics.comparison.compare_models` for posterior
    predictive checks (PPC), where distributing over new observations is
    the correct intent.

    Metrics computed per area from the posterior draws array of shape
    ``(total_draws, n_obs)``:

    * ``mean`` — posterior mean of the latent parameter
    * ``sd`` — posterior standard deviation
    * ``ci_lower``, ``ci_upper`` — ArviZ HDI at *ci_prob*
    * ``rse_pct`` — relative standard error: ``(sd / |mean|) × 100``
    * ``mse`` — posterior variance (= MSE of the HB estimator)
    * ``rmse`` — ``sqrt(mse)``

    A ``group`` column is appended when the model was fitted with group
    information (``model.result.extra["group"]`` is not ``None``).

    Args:
        model: A fitted :class:`~hbsaemp.models._base.BaseModel`.
        new_data: Optional out-of-sample DataFrame for prediction.
            When ``None``, uses the training data (in-sample estimates).
        ci_prob: Credible interval probability. Default 0.95 (95 % HDI).

    Returns:
        :class:`AreaEstimatesResult` with per-area table and overall metrics.

    Raises:
        ModelNotFittedError: If *model* has not been fitted.
        ImportError: If ``arviz`` is not installed.

    Example (v1)::

        result = estimate_areas(model)
        result.summary()
        result.result_table.to_csv("sae_estimates.csv")
    """
    try:
        import arviz as az
    except ImportError as exc:
        raise ImportError(
            "estimate_areas() requires arviz>=0.18. "
            "Install with: pip install 'hbsaemp[bambi]'"
        ) from exc

    # ── guard: model must be fitted ──────────────────────────────────────────
    result = model.result  # raises ModelNotFittedError if not fitted

    group_col: str | None = result.extra.get("group")

    logger.info(
        "estimate_areas(): family=%r, new_data=%s, ci_prob=%s",
        result.family,
        "None (in-sample)" if new_data is None else f"shape={new_data.shape}",
        ci_prob,
    )

    # ── posterior draws of the latent mean/probability parameter ─────────────
    # kind="response_params" extracts idata.posterior[_MEAN_PARAM_KEY]:
    #   Gaussian/Beta/Lognormal → "mu"  (shrinkage estimate of θ_i)
    #   Binomial                → "p"   (area-level success probability)
    # Using the posterior predictive (kind="response") would re-add the local
    # sampling variance D_i, inflating SD/MSE/RMSE and destroying shrinkage.
    # draws shape: (total_draws, n_obs)
    draws: np.ndarray = model.predict(new_data=new_data, kind="response_params")
    n_obs: int = draws.shape[1]

    # ── group labels ──────────────────────────────────────────────────────────
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

    # ── per-area statistics (vectorised over axis=0 = sample dim) ────────────
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

    # HDI per area. Loop here is the safe path across arviz versions —
    # numpy/xarray axis semantics for az.hdi shifted between releases. The
    # main perf win comes from vectorising mean/sd/var above and skipping
    # the list-of-dicts → DataFrame construction below.
    # The kwarg was renamed from ``hdi_prob`` to ``prob`` in ArviZ ≥0.20;
    # cached via ``_get_hdi_kwarg`` so signature inspection runs once per process.
    hdi_kwarg = _get_hdi_kwarg()
    hdi_arr = np.empty((n_obs, 2), dtype=float)
    for i in range(n_obs):
        hdi_arr[i] = az.hdi(draws[:, i], **{hdi_kwarg: ci_prob})

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
