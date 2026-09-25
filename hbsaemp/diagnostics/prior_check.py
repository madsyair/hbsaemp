"""Prior predictive check for HBSAE models.

Python equivalent of R hbsaems::hbpc().

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from hbsaemp._logging import get_logger
from hbsaemp.diagnostics._plot_utils import (
    _diagnostic_var_names,
    _ensure_headless_matplotlib,
    _fig_from_axes,
    _idata_groups,
)

if TYPE_CHECKING:
    from hbsaemp.models._base import BaseModel

logger = get_logger(__name__)
__all__: list[str] = ["PriorCheckResult", "check_prior", "hbpc"]


@dataclass
class PriorCheckResult:
    """Container for prior predictive check results.

    Attributes:
        prior_predictive_plot: Prior-predictive-vs-observed `matplotlib.Figure`,
            or `None` if it could not be rendered (see logged warning).
        prior_summary: `DataFrame` (mean/sd/hdi per scalar parameter) built
            from `idata.prior`, or an empty `DataFrame` if it could not be
            summarised.
        idata: `arviz.InferenceData` with `prior`, `prior_predictive` and
            `observed_data` groups, straight from `prior_predictive_idata()`.
    """
    prior_predictive_plot: Any = None
    prior_summary: Any = None
    idata: Any = None

    def summary(self) -> str:
        """One-screen text summary."""
        if self.idata is None:
            return "PriorCheckResult [not computed — call check_prior()]"
        n_params = 0 if self.prior_summary is None else len(self.prior_summary)
        return (
            f"PriorCheckResult\n"
            f"  Parameters : {n_params}\n"
            f"  Plot       : {'available' if self.prior_predictive_plot is not None else 'unavailable'}"
        )

    def __repr__(self) -> str:
        n = 0 if self.prior_summary is None else len(self.prior_summary)
        return f"PriorCheckResult(n_params={n}, has_plot={self.prior_predictive_plot is not None})"


def check_prior(
    model: BaseModel,
    *,
    response_var: str | None = None,
    n_draws: int = 50,
) -> PriorCheckResult:
    """Perform a prior predictive check without fitting the model.

    Python equivalent of ``hbpc()`` in R hbsaems.

    Runs the model's build pipeline and samples from the priors only. The
    model is not fitted: ``model.is_fitted`` stays ``False``.

    Args:
        model: A :class:`~hbsaemp.BaseModel`, fitted or not.
        response_var: Response label for the plot title. Default:
            ``model.response_name``.
        n_draws: Number of prior predictive draws. Default 50, enough for a
            visual check.

    Returns:
        :class:`PriorCheckResult` with ``prior_predictive_plot`` (``None`` if
        the plot failed; the reason is logged), ``prior_summary`` (empty if
        the draws could not be summarised) and ``idata``.

    Raises:
        ImportError: If ``bambi`` is not installed.
        DataValidationError: If the data fails validation.
    """
    _ensure_headless_matplotlib()

    idata = model.prior_predictive_idata(draws=n_draws)
    response = response_var or model.response_name

    logger.info(
        "check_prior(): response=%r, n_draws=%d", response, n_draws,
    )

    prior_summary: pd.DataFrame = pd.DataFrame()
    try:
        import arviz as az

        var_names = _diagnostic_var_names(idata.prior, policy="scalar_and_group")
        if var_names:
            prior_summary = az.summary(idata, group="prior", var_names=var_names)
    except Exception as exc:  
        logger.warning("check_prior(): could not summarise prior draws: %s", exc)

    prior_predictive_plot: Any = None
    try:
        import arviz_plots as azp

        has_observed = "observed_data" in _idata_groups(idata)
        pc = azp.plot_ppc_dist(
            idata,
            group="prior_predictive",
            visuals={"observed_dist": has_observed},
        )
        fig = _fig_from_axes(pc)
        if fig is not None:
            fig.suptitle(
                f"Prior Predictive: {response}", fontsize=11, fontweight="bold"
            )
        prior_predictive_plot = fig
    except Exception as exc:  
        logger.warning("check_prior(): could not render prior predictive plot: %s", exc)

    logger.info(
        "check_prior(): complete — %d prior param(s) summarised, plot=%s",
        len(prior_summary), "ok" if prior_predictive_plot is not None else "unavailable",
    )

    return PriorCheckResult(
        prior_predictive_plot=prior_predictive_plot,
        prior_summary=prior_summary,
        idata=idata,
    )


#: R-style alias.
hbpc = check_prior
