"""hbsaemp.diagnostics — convergence, prior check, model comparison."""
from __future__ import annotations

from hbsaemp.diagnostics.convergence import ConvergenceResult, check_convergence, hbcc
from hbsaemp.diagnostics.prior_check import PriorCheckResult, check_prior, hbpc
from hbsaemp.diagnostics.comparison import ComparisonResult, compare_models, hbmc

__all__ = [
    "ConvergenceResult", "check_convergence", "hbcc",
    "PriorCheckResult", "check_prior", "hbpc",
    "ComparisonResult", "compare_models", "hbmc",
]
