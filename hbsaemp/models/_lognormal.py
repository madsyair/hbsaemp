"""Lognormal log-normal HBSAE model.

v0: stubs raising NotImplementedError.
v1: Bambi Gaussian model on the log-scale response.

Design note
-----------
The lognormal HBSAE model treats the *already log-transformed* response
(e.g. ``y_log_obs``) as a Gaussian variable.  Users must supply the
log-transformed column as the response in the formula::

    create_model("y_log_obs ~ x1 + x2 + (1|group)", family="lognormal", ...)

Internally, ``LognormalModel`` calls Bambi with ``family="gaussian"`` (via
:meth:`_bambi_family`) and ``link="identity"``.  This is equivalent to the
R hbsaems area-level lognormal FH model.  ``ModelResult.family`` keeps the
user-facing label ``"lognormal"``.

Fay-Herriot variant
-------------------
When ``sampling_var_col`` is provided (column of known sampling variances
:math:`\\psi_i`), the residual sigma is pinned::

    sigma ~ 0 + offset(log_sqrt_D)   where log_sqrt_D = 0.5 * log(psi_i)

added by :class:`~hbsaemp.data._preprocessor.DataPreprocessor`.

The fit/predict pipeline lives in :class:`~hbsaemp.models._base.BaseModel`;
this module only declares the lognormal-specific hooks
(:attr:`_preproc_family`, :meth:`_bambi_family`,
:meth:`_build_formula_and_link`, :meth:`_extra_result_dict`,
:meth:`_extra_pipeline_kwargs`).
"""
from __future__ import annotations

from typing import Any, ClassVar

from hbsaemp.models._base import BaseModel

__all__: list[str] = ["LognormalModel"]


class LognormalModel(BaseModel):
    """Lognormal HBSAE model — Gaussian fit on the log-scale response.

    The response column in *formula* must already be log-transformed (e.g.
    ``y_log_obs`` from :func:`~hbsaemp.data.datasets.load_dataset`).

    When ``sampling_var_col`` is provided (Fay-Herriot variant), sigma is
    pinned via::

        sigma ~ 0 + offset(log_sqrt_D)

    Default link for ``mu``: ``identity`` (response is already on log scale).

    .. note::
        :meth:`predict` returns samples on the **log scale**.  Apply
        :func:`numpy.exp` to recover the original-scale response.

    Args:
        *args: Forwarded to :class:`~hbsaemp.models._base.BaseModel`.
        sampling_var_col: Column name for known sampling variances
            :math:`\\psi_i` (optional).
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

    @property
    def _preproc_family(self) -> str:
        """Override: validator + preprocessor see this as ``"gaussian"``.

        The log-scale response can be negative, so the lognormal positivity
        check in :class:`~hbsaemp.data._validator.DataValidator` must be
        skipped.  Using ``"gaussian"`` also triggers
        :meth:`~hbsaemp.data._preprocessor.DataPreprocessor._transform_gaussian`
        which adds the ``log_sqrt_D`` offset column.
        """
        return "gaussian"

    def _bambi_family(self) -> str:
        """Override: Bambi sees ``"gaussian"`` since we fit on log-scale y.

        ``ModelResult.family`` retains the user-facing ``"lognormal"`` label
        because :meth:`~hbsaemp.models._base.BaseModel.fit` uses
        ``self._family`` rather than this method when constructing the result.
        """
        return "gaussian"

    def _build_formula_and_link(
        self,
        bmb_module: Any,
        response: str,
    ) -> tuple[Any, Any]:
        # Same Fay-Herriot offset pattern as GaussianModel when psi_i is known.
        # "1 +" instead of "0 +" works around the Bambi 0.14+ TypeError on
        # empty-design distributional components — see GaussianModel docs.
        if self._sampling_var_col is not None:
            formula = bmb_module.Formula(
                self._formula, "sigma ~ 1 + offset(log_sqrt_D)"
            )
            link: Any = {"mu": self._link, "sigma": "log"}
        else:
            formula = self._formula
            link = self._link
        return formula, link

    def _workaround_priors(self, bmb_module: Any) -> dict[str, Any]:
        # Pin sigma intercept tightly to 0 so sigma ≈ sqrt(psi) from the offset.
        if self._sampling_var_col is not None:
            return {
                "sigma_Intercept": bmb_module.Prior("Normal", mu=0.0, sigma=1e-3),
            }
        return {}

