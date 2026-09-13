"""Binomial logit-normal HBSAE model.

User formula `"y ~ rhs"` is rewritten to Bambi 0.18+ proportion syntax
`"p(y, n) ~ rhs"`, where `n` is `trials_col`. Neither the rewrite nor the
matching posterior-predictive key is written here: both come from
`FAMILY_SPECS["binomial"].addition_template` via `BaseModel._addition_lhs()`.
What remains is the one rule a spec cannot state — `trials_col` is required,
and a subclass built directly bypasses the factory's arity check.
"""
from __future__ import annotations

from typing import Any

from hbsaemp.models._base import BaseModel

__all__: list[str] = ["BinomialModel"]


class BinomialModel(BaseModel):
    """Binomial model for count outcomes (successes out of `trials_col`).

    `predict(kind="response")` returns posterior predictive *counts*
    `y* ~ Binomial(n_i, p_i)`. Use `kind="response_params"` for area-level
    success probabilities (extracts `idata.posterior["p"]`). `new_data`
    must include `trials_col`.

    Args:
        *args: Forwarded to `BaseModel`.
        trials_col: Column with trial counts n_i (required).
        **kwargs: Forwarded to `BaseModel` (including `link`).
    """

    def __init__(
        self,
        *args: Any,
        trials_col: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._trials_col = trials_col

    # hooks

    def _pre_fit_checks(self) -> None:
        """Fail fast on missing trials_col, then validate the link (via super).

        `required_params` already makes `create_model()` refuse without it;
        this guards the direct-construction path, where the formula would
        otherwise be rewritten to the literal ``p(y, None)``.
        """
        if self._trials_col is None:
            raise ValueError(
                "BinomialModel requires trials_col. "
                "Pass trials= to create_model()."
            )
        super()._pre_fit_checks()
