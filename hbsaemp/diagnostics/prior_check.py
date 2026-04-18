"""Prior predictive check for HBSAE models.

Python equivalent of R hbsaems::hbpc().

v0: check_prior() stub + PriorCheckResult dataclass.
v1: Concrete — bambi prior_predictive() + arviz.plot_ppc().
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from hbsaemp._logging import get_logger
from hbsaemp.models._base import BaseModel

logger = get_logger(__name__)
__all__: list[str] = ["PriorCheckResult", "check_prior", "hbpc"]


@dataclass
class PriorCheckResult:
    """Container for prior predictive check results.

    Attributes:
        prior_predictive_plot: PPC plot object (matplotlib Figure in v1).
        prior_summary: DataFrame summarising prior distributions.
        idata: arviz.InferenceData with prior samples (v1).
    """
    prior_predictive_plot: Any = None
    prior_summary: Any = None
    idata: Any = None

    def summary(self) -> str:
        return "PriorCheckResult [stub]"

    def __repr__(self) -> str:
        return "PriorCheckResult(status=stub)"


def check_prior(
    model: BaseModel,
    *,
    response_var: str | None = None,
    n_draws: int = 50,
) -> PriorCheckResult:
    """Perform a prior predictive check.

    Python equivalent of ``hbpc()`` in R hbsaems.

    The model must have been created with ``ModelConfig(sample_prior="only")``
    or will be re-sampled from the prior.

    Args:
        model: A :class:`~hbsaemp.models._base.BaseModel` (fitted or unfitted).
        response_var: Name of the response variable for the PPC plot.
            If ``None``, inferred from the formula.
        n_draws: Number of prior draws to plot. Default 50.

    Returns:
        :class:`PriorCheckResult` with plot and summary.

    Raises:
        NotImplementedError: In v0.
    """
    raise NotImplementedError(
        "check_prior() requires bambi>=0.14 and arviz>=0.18 (v1)."
    )


#: R-style alias.
hbpc = check_prior
