"""Model comparison and goodness-of-fit checks.

Python equivalent of R hbsaems::hbmc().

v0: compare_models() stub + ComparisonResult dataclass.
v1: LOO/WAIC via arviz; pp_check plot via arviz.
v2+: Bayes Factor via bridge sampling; prior sensitivity loop.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from hbsaemp._logging import get_logger
from hbsaemp.models._base import BaseModel

logger = get_logger(__name__)
__all__: list[str] = ["ComparisonResult", "compare_models", "hbmc"]


@dataclass
class ComparisonResult:
    """Container for model comparison results.

    Attributes:
        loo: LOO cross-validation result (arviz ELPDData in v1).
        waic: WAIC result (arviz ELPDData in v1).
        bayes_factor: Bayes Factor (v2+, None in v1).
        pp_check_plot: Posterior predictive check plot.
        params_plot: Marginal posterior distributions plot.
        prior_sensitivity: Prior sensitivity results (v2+).
        comparison_table: DataFrame comparing metrics across models (multi-model).
    """
    loo: Any = None
    waic: Any = None
    bayes_factor: Any = None
    pp_check_plot: Any = None
    params_plot: Any = None
    prior_sensitivity: Any = None
    comparison_table: Any = None

    def summary(self) -> str:
        return "ComparisonResult [stub]"

    def __repr__(self) -> str:
        return "ComparisonResult(status=stub)"


def compare_models(
    models: BaseModel | list[BaseModel],
    *,
    metrics: list[str] | None = None,
    n_draws_ppc: int = 100,
    run_prior_sensitivity: bool = False,
    sensitivity_vars: list[str] | None = None,
) -> ComparisonResult:
    """Compute goodness-of-fit metrics and compare models.

    Python equivalent of ``hbmc()`` in R hbsaems.

    Args:
        models: A single fitted model or a list of fitted models.
            When a list is provided, models are compared pairwise.
        metrics: Metrics to compute. v1: ``["loo", "waic"]``.
            v2+: also ``["bf"]`` (Bayes Factor). Default: all v1 metrics.
        n_draws_ppc: Number of draws for the posterior predictive check plot.
        run_prior_sensitivity: Whether to run prior sensitivity analysis (v2+).
        sensitivity_vars: Parameter names for sensitivity analysis (v2+).

    Returns:
        :class:`ComparisonResult` with metrics and plots.

    Raises:
        ModelNotFittedError: If any model in *models* has not been fitted.
        NotImplementedError: In v0.
    """
    raise NotImplementedError(
        "compare_models() requires arviz>=0.18 and fitted models (v1)."
    )


#: R-style alias.
hbmc = compare_models
