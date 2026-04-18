"""Convergence diagnostics for fitted HBSAE models.

Python equivalent of R hbsaems::hbcc().

v0: check_convergence() stub + ConvergenceResult dataclass.
v1: Concrete — Rhat/ESS via arviz; trace/dens/acf plots.
v2+: Geweke, Heidelberger-Welch, Raftery-Lewis via pymc-extras.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from hbsaemp._logging import get_logger
from hbsaemp.models._base import BaseModel

logger = get_logger(__name__)
__all__: list[str] = ["ConvergenceResult", "check_convergence", "hbcc"]


@dataclass
class ConvergenceResult:
    """Container for convergence diagnostic results.

    Attributes:
        rhat_ess: DataFrame with Rhat, Bulk_ESS, Tail_ESS per parameter.
        plots: Dict of diagnostic plots keyed by plot type name.
        geweke: Geweke Z-scores (v2+, None in v1).
        heidel: Heidelberger-Welch results (v2+, None in v1).
        raftery: Raftery-Lewis results (v2+, None in v1).
    """
    rhat_ess: Any = None
    plots: dict[str, Any] = field(default_factory=dict)
    geweke: Any = None
    heidel: Any = None
    raftery: Any = None

    def summary(self) -> str:
        """Human-readable summary of convergence diagnostics."""
        return "ConvergenceResult [stub — call check_convergence() with a fitted model]"

    def __repr__(self) -> str:
        return "ConvergenceResult(status=stub)"


def check_convergence(
    model: BaseModel,
    *,
    diag_tests: list[str] | None = None,
    plot_types: list[str] | None = None,
) -> ConvergenceResult:
    """Assess MCMC convergence of a fitted model.

    Python equivalent of ``hbcc()`` in R hbsaems.

    Args:
        model: A fitted :class:`~hbsaemp.models._base.BaseModel`.
        diag_tests: Tests to run. v1: ``["rhat", "ess"]``.
            v2+: also ``["geweke", "heidel", "raftery"]``.
            Default: ``["rhat", "ess"]``.
        plot_types: Plot types to generate. v1: ``["trace", "dens", "acf", "rhat", "neff"]``.
            v2+: also ``["nuts_energy"]``. Default: all v1 plots.

    Returns:
        :class:`ConvergenceResult` with diagnostics and plots.

    Raises:
        ModelNotFittedError: If *model* has not been fitted.
        NotImplementedError: In v0.
    """
    raise NotImplementedError(
        "check_convergence() requires arviz>=0.18 and a fitted model (v1)."
    )


#: R-style alias — ``hbcc(model)`` is equivalent to ``check_convergence(model)``.
hbcc = check_convergence
