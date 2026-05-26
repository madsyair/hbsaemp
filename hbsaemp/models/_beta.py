"""Beta logit-normal HBSAE model.

v0: stubs raising NotImplementedError.
v1: Bambi distributional model — pins precision phi_i = n_i/deff_i - 1
    via ``kappa ~ 0 + offset(log_phi)`` when n_col and deff_col are provided.

The fit/predict pipeline lives in :class:`~hbsaemp.models._base.BaseModel`;
this module only declares the Beta-specific hooks
(:meth:`_build_formula_and_link`, :meth:`_extra_result_dict`,
:meth:`_extra_pipeline_kwargs`).
"""
from __future__ import annotations

from typing import Any, ClassVar

from hbsaemp.models._base import BaseModel

__all__: list[str] = ["BetaModel"]


class BetaModel(BaseModel):
    """Beta distribution HBSAE model for proportional estimates in (0, 1).

    When ``n_col`` and ``deff_col`` are both provided, precision is pinned:

    .. math::

        \\phi_i = n_i / \\text{deff}_i - 1

    implemented via Bambi's distributional formula::

        kappa ~ 0 + offset(log_phi)

    where ``log_phi`` is added to the preprocessed DataFrame by
    :class:`~hbsaemp.data._preprocessor.DataPreprocessor`.

    When ``n_col`` / ``deff_col`` are absent, Bambi estimates ``kappa``
    from the data (standard Beta regression).

    Default link for ``mu``: ``logit``.

    Args:
        *args: Forwarded to :class:`~hbsaemp.models._base.BaseModel`.
        n_col: Column name for survey sample size (optional).
        deff_col: Column name for design effect (optional).
        link: Link function for ``mu``. Default ``"logit"``.
        squeeze: Apply Smithson-Verkuilen squeeze ``(y*(n-1)+0.5)/n`` to
            the response before fitting.  Default ``False``.  Set ``True``
            only when the dataset contains boundary values ``y=0`` or
            ``y=1`` (requires *n_col* and *deff_col* to be provided).
            When ``True``, the validator relaxes the domain check to
            ``[0, 1]`` (closed interval).
        **kwargs: Forwarded to :class:`~hbsaemp.models._base.BaseModel`.
    """

    #: Bambi's Beta family uses ``"mu"`` as the mean parameter name.
    _MEAN_PARAM_KEY: ClassVar[str] = "mu"
    _EXTRA_FIELD_NAMES: ClassVar[tuple[str, ...]] = ("n_col", "deff_col", "squeeze")

    def __init__(
        self,
        *args: Any,
        n_col: str | None = None,
        deff_col: str | None = None,
        link: str | None = None,
        squeeze: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._n_col = n_col
        self._deff_col = deff_col
        self._link = link or "logit"
        self._squeeze = squeeze

    # ── hooks ────────────────────────────────────────────────────────────────

    def _build_formula_and_link(
        self,
        bmb_module: Any,
        response: str,
    ) -> tuple[Any, Any]:
        # When phi is pinned from data: distributional model
        #   kappa ~ 1 + offset(log_phi)   (with tight Intercept prior at 0)
        # The "1 +" is a Bambi 0.14+ workaround — "kappa ~ 0 + offset(log_phi)"
        # produces an empty common design matrix, so DistributionalComponent.
        # predict() returns a raw numpy offset whose .transpose(*str_dims)
        # raises TypeError. Adding the intercept (clamped near 0 via
        # ``_workaround_priors``) keeps the design non-empty so
        # log(kappa) ≈ log(phi) → kappa ≈ phi.
        if self._n_col is not None and self._deff_col is not None:
            formula = bmb_module.Formula(
                self._formula, "kappa ~ 1 + offset(log_phi)"
            )
            # Both mu and kappa links — partial dict drops kappa and causes
            # KeyError inside Bambi's backend build step.
            link: Any = {"mu": self._link, "kappa": "log"}
        else:
            formula = self._formula
            link = self._link
        return formula, link

    def _workaround_priors(self, bmb_module: Any) -> dict[str, Any]:
        # Pin the kappa intercept tightly to 0 so kappa ≈ phi from the offset
        # (see _build_formula_and_link comment).
        if self._n_col is not None and self._deff_col is not None:
            return {
                "kappa_Intercept": bmb_module.Prior("Normal", mu=0.0, sigma=1e-3),
            }
        return {}

