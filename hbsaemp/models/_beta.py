"""Beta logit-normal HBSAE model.

When `n_col` and `deff_col` are given, precision is pinned to phi = n/deff - 1
via `kappa ~ 1 + offset(log_phi)`. Otherwise Bambi estimates kappa.

Pipeline lives in `BaseModel`; this module supplies only family-specific hooks.
"""
from __future__ import annotations

from typing import Any

from hbsaemp.models._base import BaseModel

__all__: list[str] = ["BetaModel"]


class BetaModel(BaseModel):
    """Beta distribution model for proportional estimates in (0, 1).

    Args:
        *args: Forwarded to `BaseModel`.
        n_col, deff_col: Survey size and design effect columns. Both or
            neither. When both are given, the FH precision offset is enabled
            (preprocessor adds the `log_phi` column).
        link: Link for mu (default `"logit"`).
        squeeze: Smithson-Verkuilen squeeze `(y*(n-1)+0.5)/n` applied before
            fit. Default False. Set True only for boundary `y=0` / `y=1`
            cases (requires `n_col`+`deff_col`). When True the validator
            relaxes the domain to `[0, 1]`.
        **kwargs: Forwarded to `BaseModel`.
    """

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
        self._link = link or self._default_link
        self._squeeze = squeeze

    # hooks

    def _pre_fit_checks(self) -> None:
        """Guard squeeze/offset combo, then validate the link (via super).

        `squeeze=True` only takes effect when the precision offset is active
        (the Smithson-Verkuilen transform lives in that branch of the
        preprocessor). Without `n`/`deff`, the validator would relax the
        domain to `[0, 1]` while the boundary values stay un-squeezed and
        break the Beta likelihood — so reject the combination up front.
        """
        if self._squeeze and (self._n_col is None or self._deff_col is None):
            raise ValueError(
                "squeeze=True for family='beta' requires both n and deff "
                "(the Smithson-Verkuilen transform is only applied when the "
                "precision offset is active)."
            )
        super()._pre_fit_checks()

    def _build_formula_and_link(
        self,
        bmb_module: Any,
        response: str,
    ) -> tuple[Any, Any]:
        # Delegate to the shared FH offset helper in BaseModel.
        # When phi is pinned: distributional kappa ~ 1 + offset(log_phi).
        # When absent: plain Beta regression.
        return self._build_distributional_formula(
            bmb_module,
            self._formula,
            param="kappa",
            offset_col=self._spec.offset_col if (
                self._n_col is not None and self._deff_col is not None
            ) else None,
            mu_link=self._link,
        )

    def _workaround_priors(self, bmb_module: Any) -> dict[str, Any]:
        # Pin kappa intercept to ~0 so kappa ≈ phi from the offset.
        return self._pin_intercept_prior(
            bmb_module,
            param="kappa",
            active=self._n_col is not None and self._deff_col is not None,
        )
