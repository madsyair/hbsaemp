"""Tab 2 — Data exploration and EDA.

Equivalent to the "Data Exploration" tab in R hbsaems Shiny app.

v0: stub class.
v1: implemented with Panel + Matplotlib/hvplot.

Features (v1)
--------------
* **Summary statistics**: describe() table per variable.
* **Histogram**: frequency + density curve (matplotlib).
* **Boxplot**: median, IQR, outliers (matplotlib).
* **Scatter plot + correlation**: with axis transform options (none/log/logit).
  Correlation coefficients computed: Pearson, Spearman, Chatterjee Xi, dCor.

Mapping from R hbsaems
-----------------------
.. code-block:: text

    R Shiny                         Panel v1 equivalent
    ─────────────────────────────── ──────────────────────────────
    selectInput("explore_var_…")    pn.widgets.Select
    verbatimTextOutput("numeric_…") pn.pane.DataFrame (describe())
    plotOutput("histogram_plot")    pn.pane.Matplotlib
    plotOutput("boxplot_plot")      pn.pane.Matplotlib
    plotOutput("scatter_plot")      pn.pane.Matplotlib
    renderUI("correlation_results") pn.pane.HTML (formatted table)
    XICOR::xicor                    xicor package (v1 optional dep)
    energy::dcor.test               dcor package (v1 optional dep)
"""

from __future__ import annotations

from typing import Any

__all__: list[str] = ["ExploreTab"]


class ExploreTab:
    """Data exploration / EDA tab.

    v0: All methods raise :exc:`NotImplementedError`.
    v1: Implemented with ``panel`` + ``matplotlib``.

    Args:
        app_state: Shared application state.  Reads ``app_state["data"]``
            (set by :class:`~hbsaemp.app.tabs.data_tab.DataTab`).
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
            "ExploreTab.panel() requires panel>=1.3 (v1)."
        )

    def __repr__(self) -> str:
        return "ExploreTab(status=stub)"
