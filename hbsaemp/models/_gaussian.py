"""Gaussian (Fay-Herriot) HBSAE model.

When `sampling_var_col` is given, sigma is pinned to sqrt(D) via the
distributional formula `sigma ~ 1 + offset(log_sqrt_D)`. Otherwise Bambi
estimates sigma from the data.

Neither branch is written here: the pin is declared in
`FAMILY_SPECS["gaussian"].fixed_params` and assembled by `BaseModel`. This
class exists only to hold the design column that decides whether it applies.
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
        **kwargs: Forwarded to `BaseModel` (including `link`).
    """

    def __init__(
        self,
        *args: Any,
        sampling_var_col: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._sampling_var_col = sampling_var_col
