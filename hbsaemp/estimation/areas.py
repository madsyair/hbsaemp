"""Small Area Estimation prediction and uncertainty quantification.

Python equivalent of R hbsaems::hbsae().

v0: estimate_areas() stub + AreaEstimatesResult dataclass.
v1: Concrete — model.predict() via bambi; RSE/MSE/RMSE/CI from posterior draws.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import pandas as pd
from hbsaemp._logging import get_logger
from hbsaemp.models._base import BaseModel

logger = get_logger(__name__)
__all__: list[str] = ["AreaEstimatesResult", "estimate_areas", "hbsae"]


@dataclass
class AreaEstimatesResult:
    """Container for small area estimation results.

    Attributes:
        result_table: DataFrame with per-area statistics.
            Columns: ``mean``, ``sd``, ``ci_lower``, ``ci_upper``,
            ``rse_pct``, ``mse``, ``rmse``.
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

    Computes per-area: posterior mean, SD, credible interval, RSE, MSE, RMSE.

    Args:
        model: A fitted :class:`~hbsaemp.models._base.BaseModel`.
        new_data: Optional out-of-sample DataFrame for prediction.
            If ``None``, uses the training data.
        ci_prob: Credible interval probability. Default 0.95 (95% HDI).

    Returns:
        :class:`AreaEstimatesResult` with per-area table and overall metrics.

    Raises:
        ModelNotFittedError: If *model* has not been fitted.
        NotImplementedError: In v0.

    Example (v1)::

        result = estimate_areas(model)
        result.summary()
        result.result_table.to_csv("sae_estimates.csv")
    """
    raise NotImplementedError(
        "estimate_areas() requires a fitted model and bambi>=0.14 (v1)."
    )


#: R-style alias — ``hbsae(model)`` equals ``estimate_areas(model)``.
hbsae = estimate_areas
