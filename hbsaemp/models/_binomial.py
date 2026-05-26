"""Binomial logit-normal HBSAE model.

v0: stubs raising NotImplementedError.
v1: Bambi binomial model with ``p(y, n)`` proportion-syntax (Bambi 0.14+).

Formula rewrite
---------------
User supplies a plain formula::

    "y ~ x1 + x2 + (1|group)"

:meth:`BinomialModel._build_formula_and_link` rewrites the LHS to the
Bambi 0.14+ binomial syntax::

    "p(y, n) ~ x1 + x2 + (1|group)"

where ``n`` is the column specified by ``trials_col``.  The older
``"y | trials(n) ~ rhs"`` form is rejected by ``formulae`` >= 0.5
because ``|`` on the LHS is no longer a trials specification.

The fit/predict pipeline lives in :class:`~hbsaemp.models._base.BaseModel`;
this module only declares the Binomial-specific hooks
(:meth:`_pre_fit_checks`, :meth:`_build_formula_and_link`,
:meth:`_extra_result_dict`, :meth:`_extra_pipeline_kwargs`).
"""
from __future__ import annotations

from typing import Any, ClassVar

from hbsaemp._logging import get_logger
from hbsaemp.models._base import BaseModel

logger = get_logger(__name__)
__all__: list[str] = ["BinomialModel"]


class BinomialModel(BaseModel):
    """Binomial HBSAE model for count outcomes (successes out of trials).

    ``trials_col`` is **required** — it names the column containing trial
    counts :math:`n_i`.  The response column contains success counts
    :math:`y_i`.

    Internally rewrites the user formula::

        "y ~ x1 + (1|group)"
        →  "p(y, n) ~ x1 + (1|group)"

    and calls Bambi with ``family="binomial"``.

    Default link for ``mu``: ``logit``.

    .. note::
        :meth:`predict` with ``kind="response"`` returns posterior
        predictive **counts** :math:`y^* \\sim \\text{Binomial}(n_i, p_i)`.
        For the area-level success probability use
        ``kind="response_params"`` which extracts ``idata.posterior["p"]``.
        ``new_data`` for prediction must include ``trials_col``.

    Args:
        *args: Forwarded to :class:`~hbsaemp.models._base.BaseModel`.
        trials_col: Column name for the number of trials :math:`n_i`.
        link: Link function for ``mu``. Default ``"logit"``.
        **kwargs: Forwarded to :class:`~hbsaemp.models._base.BaseModel`.
    """

    #: Bambi's Binomial family uses ``"p"`` as the success-probability parameter.
    _MEAN_PARAM_KEY: ClassVar[str] = "p"
    _EXTRA_FIELD_NAMES: ClassVar[tuple[str, ...]] = ("trials_col",)

    def __init__(
        self,
        *args: Any,
        trials_col: str | None = None,
        link: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._trials_col = trials_col
        self._link = link or "logit"

    # ── hooks ────────────────────────────────────────────────────────────────

    def _pre_fit_checks(self) -> None:
        """Fail fast on missing ``trials_col`` — before the ``bambi`` import."""
        if self._trials_col is None:
            raise ValueError(
                "BinomialModel requires trials_col. "
                "Pass trials= to create_model()."
            )

    def _response_pp_key(self, response: str) -> str:
        # Bambi 0.14+ stores binomial PP under the wrapped LHS literal
        # ``p(y, trials_col)`` (matching the formula passed to bmb.Model),
        # not the bare ``response`` column.
        return f"p({response}, {self._trials_col})"

    def _build_formula_and_link(
        self,
        bmb_module: Any,
        response: str,
    ) -> tuple[Any, Any]:
        # Rewrite "y ~ rhs" → "p(y, n) ~ rhs" (Bambi 0.14+ binomial syntax).
        # The legacy "y | trials(n) ~ rhs" form was removed when formulae>=0.5
        # repurposed "|" on the LHS, raising
        # "response term must be of class Term, not Model".
        _, rhs = self._formula.split("~", 1)
        formula = f"p({response}, {self._trials_col}) ~ {rhs.strip()}"
        logger.debug("BinomialModel: rewritten formula = %r", formula)
        return formula, self._link

