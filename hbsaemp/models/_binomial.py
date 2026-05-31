"""Binomial logit-normal HBSAE model.

User formula `"y ~ rhs"` is rewritten to Bambi 0.18+ proportion syntax
`"p(y, n) ~ rhs"`, where `n` is `trials_col`. The legacy
`"y | trials(n) ~ rhs"` form was dropped by `formulae` >= 0.5.

Pipeline lives in `BaseModel`; this module supplies only family-specific hooks.
"""
from __future__ import annotations

from typing import Any

from hbsaemp._logging import get_logger
from hbsaemp.models._base import BaseModel

logger = get_logger(__name__)
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
        link: Link for p (default `"logit"`).
        **kwargs: Forwarded to `BaseModel`.
    """

    def __init__(
        self,
        *args: Any,
        trials_col: str | None = None,
        link: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._trials_col = trials_col
        self._link = link or self._default_link

    # hooks

    def _pre_fit_checks(self) -> None:
        """Fail fast on missing trials_col, then validate the link (via super)."""
        if self._trials_col is None:
            raise ValueError(
                "BinomialModel requires trials_col. "
                "Pass trials= to create_model()."
            )
        super()._pre_fit_checks()

    def _response_pp_key(self, response: str) -> str:
        # Bambi 0.18+ stores binomial PP under the wrapped LHS literal
        # ``p(y, trials_col)`` (matching the formula passed to bmb.Model),
        # not the bare ``response`` column.
        return f"p({response}, {self._trials_col})"

    def _build_formula_and_link(
        self,
        bmb_module: Any,
        response: str,
    ) -> tuple[Any, Any]:
        # Rewrite "y ~ rhs" → "p(y, n) ~ rhs" (Bambi 0.18+ binomial syntax).
        # The legacy "y | trials(n) ~ rhs" form was removed when formulae>=0.5
        # repurposed "|" on the LHS, raising
        # "response term must be of class Term, not Model".
        _, rhs = self._formula.split("~", 1)
        formula = f"p({response}, {self._trials_col}) ~ {rhs.strip()}"
        logger.debug("BinomialModel: rewritten formula = %r", formula)
        return formula, self._link

