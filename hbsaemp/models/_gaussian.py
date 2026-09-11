"""Gaussian (Fay-Herriot) HBSAE model.

When `sampling_var_col` is given, sigma is pinned to sqrt(D) via the
distributional formula `sigma ~ 1 + offset(log_sqrt_D)`. Otherwise Bambi
estimates sigma from the data.

Pipeline lives in `BaseModel`; this module supplies only family-specific hooks.
"""
from __future__ import annotations

from typing import Any

from hbsaemp.models._base import BaseModel

__all__: list[str] = ["GaussianModel"]


class GaussianModel(BaseModel):
    """Gaussian hierarchical model for continuous area-level estimates.

    Args:
        *args: Forwarded to `BaseModel`.
        sampling_var_col: Column with known D_i (optional). When set,
            enables the FH distributional formula `sigma ~ 1 + offset(log_sqrt_D)`
            (preprocessor adds the `log_sqrt_D` column).
        link: Link for mu (default `"identity"`).
        **kwargs: Forwarded to `BaseModel`.
    """

    def __init__(
        self,
        *args: Any,
        sampling_var_col: str | None = None,
        link: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._sampling_var_col = sampling_var_col
        self._link = link or self._default_link

    # hooks

    def _build_formula_and_link(
        self,
        bmb_module: Any,
        response: str,
    ) -> tuple[Any, Any]:
        # Delegate to the shared FH offset helper in BaseModel.
        # When D_i known (FH): distributional sigma ~ 1 + offset(log_sqrt_D).
        # When absent: plain gaussian regression.
        return self._build_distributional_formula(
            bmb_module,
            self._formula,
            param="sigma",
            offset_col=self._spec.offset_col if self._sampling_var_col is not None else None,
            mu_link=self._link,
        )

    def _workaround_priors(self, bmb_module: Any) -> dict[str, Any]:
        # Pin sigma intercept to ~0 so sigma ≈ sqrt(D) from the offset.
        return self._pin_intercept_prior(
            bmb_module, param="sigma", active=self._sampling_var_col is not None
        )
