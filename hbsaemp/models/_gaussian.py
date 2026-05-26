"""Gaussian (Fay-Herriot) HBSAE model.

v0: stubs raising NotImplementedError.
v1: Bambi distributional model — pins sigma_i = sqrt(D_i) via
    ``sigma ~ 0 + offset(log_sqrt_D)`` when ``sampling_var_col`` is provided.

The fit/predict pipeline lives in :class:`~hbsaemp.models._base.BaseModel`;
this module only declares the Gaussian-specific hooks
(:meth:`_build_formula_and_link`, :meth:`_extra_result_dict`,
:meth:`_extra_pipeline_kwargs`).
"""
from __future__ import annotations

from typing import Any, ClassVar

from hbsaemp.models._base import BaseModel

__all__: list[str] = ["GaussianModel"]


class GaussianModel(BaseModel):
    """Gaussian hierarchical Bayesian model for continuous area-level estimates.

    Suitable for Fay-Herriot style models.

    When ``sampling_var_col`` is provided, the residual standard deviation is
    pinned to the known direct-estimate error:

    .. math::

        \\sigma_i = \\sqrt{D_i}

    implemented via Bambi's distributional formula::

        sigma ~ 0 + offset(log_sqrt_D)

    where ``log_sqrt_D = 0.5 * log(D)`` is added by
    :class:`~hbsaemp.data._preprocessor.DataPreprocessor`.

    When ``sampling_var_col`` is absent, Bambi estimates ``sigma`` from
    the data (standard Gaussian regression).

    Default link for ``mu``: ``identity``.

    Args:
        *args: Forwarded to :class:`~hbsaemp.models._base.BaseModel`.
        sampling_var_col: Column name for known sampling variances :math:`D_i`
            (optional).  Enables the Fay-Herriot distributional formula.
        link: Link function for ``mu``. Default ``"identity"``.
        **kwargs: Forwarded to :class:`~hbsaemp.models._base.BaseModel`.
    """

    #: Bambi's Gaussian family uses ``"mu"`` as the mean parameter name.
    _MEAN_PARAM_KEY: ClassVar[str] = "mu"
    _EXTRA_FIELD_NAMES: ClassVar[tuple[str, ...]] = ("sampling_var_col",)

    def __init__(
        self,
        *args: Any,
        sampling_var_col: str | None = None,
        link: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._sampling_var_col = sampling_var_col
        self._link = link or "identity"

    # ── hooks ────────────────────────────────────────────────────────────────

    def _build_formula_and_link(
        self,
        bmb_module: Any,
        response: str,
    ) -> tuple[Any, Any]:
        # When D_i known (FH): sigma ~ 1 + offset(log_sqrt_D)
        # The "1 +" is a Bambi 0.14+ workaround — "sigma ~ 0 + offset(...)" leaves
        # the common design matrix empty so DistributionalComponent.predict()
        # returns a raw numpy offset whose .transpose(*str_dims) raises TypeError.
        # The Intercept is clamped to ~0 via ``_workaround_priors`` so
        # log(sigma) ≈ log(sqrt(D)) → sigma ≈ sqrt(D).
        if self._sampling_var_col is not None:
            formula = bmb_module.Formula(
                self._formula, "sigma ~ 1 + offset(log_sqrt_D)"
            )
            # Both mu and sigma links — partial dict drops sigma and causes
            # KeyError inside Bambi's backend build step.
            link: Any = {"mu": self._link, "sigma": "log"}
        else:
            formula = self._formula
            link = self._link
        return formula, link

    def _workaround_priors(self, bmb_module: Any) -> dict[str, Any]:
        # Pin sigma intercept tightly to 0 so sigma ≈ sqrt(D) from the offset.
        if self._sampling_var_col is not None:
            return {
                "sigma_Intercept": bmb_module.Prior("Normal", mu=0.0, sigma=1e-3),
            }
        return {}

