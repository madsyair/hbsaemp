"""Tab 4 — Results: diagnostics, SAE estimates, and export.

Equivalent to the "Results" tab in R hbsaems Shiny app.

v0: stub class.
v1: implemented with Panel + ArviZ plots + pandas tables.

Sub-tabs (v1)
--------------
1. **Model Summary** — posterior coefficient table (mean, sd, HDI, Rhat, ESS).
2. **Convergence diagnostics** — trace, density, ACF, Rhat plot, neff plot
   (all via ``arviz``).
3. **Model checking** — LOO, WAIC table; posterior predictive check plot.
4. **SAE estimates** — area-level table (mean, RSE, MSE, RMSE, CI lower/upper);
   bar chart with error bands.  Spatial choropleth map in v2+.
5. **Save output** — download buttons:
   - Estimates as CSV.
   - Diagnostic plots as PDF.
   - Fitted model as pickle.

Mapping from R hbsaems
-----------------------
.. code-block:: text

    R Shiny output                          Panel v1 equivalent
    ─────────────────────────────────────── ──────────────────────────────────
    verbatimTextOutput("model_summary")     pn.pane.DataFrame + pn.pane.Str
    verbatimTextOutput("model_prior")       pn.pane.Str
    plotOutput("diag_plots")                pn.pane.Matplotlib (az.plot_*)
    verbatimTextOutput("diag_numerical")    pn.pane.DataFrame
    plotOutput("ppc_plot")                  pn.pane.Matplotlib (az.plot_ppc)
    DT::dataTableOutput("sae_table")        pn.widgets.Tabulator
    plotOutput("prediction_plot")           pn.pane.Matplotlib (bar + CI)
    downloadButton("download_estimates")    pn.widgets.FileDownload
    downloadButton("download_plots")        pn.widgets.FileDownload
"""

from __future__ import annotations

from typing import Any

__all__: list[str] = ["ResultsTab"]


class ResultsTab:
    """Results display tab with convergence, SAE estimates, and export.

    v0: All methods raise :exc:`NotImplementedError`.
    v1: Implemented with ``panel`` + ``arviz`` + ``matplotlib``.

    Args:
        app_state: Shared application state.  Reads ``app_state["model"]``
            (set by :class:`~hbsaemp.app.tabs.model_tab.ModelTab`).
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
            "ResultsTab.panel() requires panel>=1.3, arviz>=0.18, "
            "and a fitted model (v1)."
        )

    def __repr__(self) -> str:
        return "ResultsTab(status=stub)"
