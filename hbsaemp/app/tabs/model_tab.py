"""Tab 3 — Model specification and fitting.

Equivalent to the "Modeling" tab in R hbsaems Shiny app.

v0: stub class.
v1: implemented with Panel widgets + hbm() backend.

Features (v1)
--------------
* **Model configuration**:
  - Response variable selector.
  - Predictor variable multi-select.
  - Group variable selector (random effects).
  - Family / distribution selector (gaussian / beta / binomial / lognormal).
  - Link function selector (auto-populated based on family).
* **Sampler configuration**: draws, tune, chains, cores, target_accept,
  random_seed — all backed by :class:`~hbsaemp.models._config.ModelConfig`.
* **Prior specification**: per-term prior distribution with name + parameters.
  "Summarize Priors" button shows ``model.prior_predictive()``.
* **Prior predictive check**: plot before fitting.
* **Fit Model** button: calls ``hbm(...).fit()`` in a background thread;
  shows progress indicator.

Mapping from R hbsaems
-----------------------
.. code-block:: text

    R Shiny                             Panel v1 equivalent
    ─────────────────────────────────── ──────────────────────────────────
    selectInput("response_var", …)      pn.widgets.Select
    pickerInput("auxiliary_vars", …)    pn.widgets.MultiChoice
    selectInput("group_var", …)         pn.widgets.Select
    selectInput("distribution_type", …) pn.widgets.Select (family)
    selectInput("hb_link", …)           pn.widgets.Select (link)
    numericInput("num_chains", …)       pn.widgets.IntInput
    sliderInput("adapt_delta", …)       pn.widgets.FloatSlider
    actionButton("fit_model", …)        pn.widgets.Button
    withProgress(…)                     pn.indicators.LoadingSpinner
"""

from __future__ import annotations

from typing import Any

__all__: list[str] = ["ModelTab"]


class ModelTab:
    """Model specification and fitting tab.

    v0: All methods raise :exc:`NotImplementedError`.
    v1: Implemented with ``panel`` + :func:`~hbsaemp.models._factory.hbm`.

    Args:
        app_state: Shared application state.  Reads ``app_state["data"]``;
            writes ``app_state["model"]`` after fitting.
    """

    def __init__(self, app_state: dict[str, Any]) -> None:
        self._state = app_state

    def panel(self) -> Any:
        """Return a Panel layout object for this tab.

        Returns:
            A ``panel`` layout object.

        Raises:
            NotImplementedError: In v0.
        """
        raise NotImplementedError(
            "ModelTab.panel() requires panel>=1.3 and bambi>=0.14 (v1)."
        )

    def get_fitted_model(self) -> Any:
        """Return the fitted :class:`~hbsaemp.models._base.BaseModel`, or ``None``.

        Raises:
            NotImplementedError: In v0.
        """
        raise NotImplementedError("ModelTab.get_fitted_model() requires v1.")

    def __repr__(self) -> str:
        return "ModelTab(status=stub)"
