"""Main application class for the hbsaemp web dashboard.

:class:`App` assembles the four tabs into a single Panel dashboard and
manages the shared application state (uploaded data, fitted model).

v0: stub — all rendering methods raise :exc:`NotImplementedError`.
v1: implemented with ``panel.template.FastListTemplate``.

Architecture
------------
.. code-block:: text

    App (v0 stub / v1 Panel dashboard)
    ├── shared state dict  {"data": DataFrame|None, "model": BaseModel|None}
    ├── DataTab    — tab 1: upload + preview
    ├── ExploreTab — tab 2: EDA (histogram, boxplot, scatter, corr)
    ├── ModelTab   — tab 3: hbm() config + fit
    └── ResultsTab — tab 4: convergence + SAE + save

Mapping from R hbsaems
-----------------------
.. code-block:: text

    R Shiny structure                  Panel v1 equivalent
    ────────────────────────────────── ────────────────────────────────────
    dashboardPage(header, sidebar, …)  pn.template.FastListTemplate(...)
    dashboardSidebar(sidebarMenu(…))   sidebar= parameter of template
    dashboardBody(tabItems(…))         main= parameter of template
    reactiveVal(NULL)                  Python dict (app_state)
    observe({ if model_fit… })         pn.param.watch callback
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hbsaemp._logging import get_logger
from hbsaemp.app._config import AppConfig, DEFAULT_APP_CONFIG
from hbsaemp.app.tabs import DataTab, ExploreTab, ModelTab, ResultsTab

logger = get_logger(__name__)

pn.extension("tabulator", sizing_mode="stretch_width")

__all__: list[str] = ["App", "AppState"]

class AppState(param.Parameterized):
    """Shared reactive state passed to every tab.

    Tabs read/write these parameters and use ``param.watch`` to react to
    changes made by upstream tabs (e.g. :class:`~hbsaemp.app.tabs.data_tab.DataTab`
    writing ``data`` triggers refreshes in
    :class:`~hbsaemp.app.tabs.explore_tab.ExploreTab` and
    :class:`~hbsaemp.app.tabs.model_tab.ModelTab`).
    """

    data         : object | None = param.Parameter(default=None)
    model        : object | None = param.Parameter(default=None)
    idata        : object | None = param.Parameter(default=None)
    y_vals       : object | None = param.Parameter(default=None)
    pred_names   : list          = param.Parameter(default=[])
    response_col : str | None    = param.Parameter(default=None)

class App:
    """hbsaemp web dashboard.
 
    Assembles :class:`~hbsaemp.app.tabs.data_tab.DataTab`,
    :class:`~hbsaemp.app.tabs.explore_tab.ExploreTab`,
    :class:`~hbsaemp.app.tabs.model_tab.ModelTab`, and
    :class:`~hbsaemp.app.tabs.results_tab.ResultsTab` into one
    ``panel.template.FastListTemplate`` dashboard, all sharing a single
    :class:`~hbsaemp.app._state.AppState`.
 
    Args:
        app_config: Appearance and server configuration.
            Defaults to :data:`~hbsaemp.app._config.DEFAULT_APP_CONFIG`.
    """

    def __init__(self, app_config: AppConfig | None = None) -> None:
        self._config: AppConfig = app_config or DEFAULT_APP_CONFIG

        self._state = AppState()

        # Instantiate the existing tab controllers around the shared state.
        self._data_tab    = DataTab(state=self._state)
        self._explore_tab = ExploreTab(state=self._state)
        self._model_tab   = ModelTab(state=self._state)
        self._results_tab = ResultsTab(state=self._state)

        logger.debug("App created: title=%r, port=%d", self._config.title, self._config.port)

    @property
    def config(self) -> AppConfig:
        """The :class:`AppConfig` for this application instance."""
        return self._config

    @property
    def state(self) -> AppState:
        """Shared :class:`~hbsaemp.app._state.AppState` (read-only view)"""
        return self._state

    def view(self) -> pn.template.FastListTemplate:
        """Assemble and return the Panel dashboard object (not yet served).

        Useful for embedding the app in a Jupyter notebook, or calling
        ``.servable()`` on the result to serve it with ``panel serve``.

        Returns:
            A ``panel.template.FastListTemplate`` instance.
        """
        tabs = pn.Tabs(
            ("Data Upload",      self._data_tab.panel()),
            ("Data Exploration", self._explore_tab.panel()),
            ("Modeling",         self._model_tab.panel()),
            ("Results",          self._results_tab.panel()),
            sizing_mode="stretch_width",
        )
        sidebar = pn.Column(
            pn.pane.Markdown(
                "**HBSAEMP** is a dashboard for Hierarchical Bayesian "
                "Small Area Estimation using Bambi, PyMC, and ArviZ."
            ),
            pn.layout.Divider(),
            pn.pane.Markdown(
                "**Workflow**\n"
                "1. **Data Upload** — upload a CSV file.\n"
                "2. **Data Exploration** — inspect summary stats, "
                "distributions, and correlations.\n"
                "3. **Modeling** — select variables, choose a family, "
                "run prior/posterior predictive checks, and fit the model.\n"
                "4. **Results** — review convergence diagnostics and "
                "download the SAE estimation results."
            ),
            sizing_mode="stretch_width",
        )
        return pn.template.FastListTemplate(
            title=self._config.title,
            sidebar=[sidebar],
            main=[tabs],
            accent=self._config.accent,
        )

    def build(self) -> pn.template.FastListTemplate:
        """Alias for :meth:`view`, kept for backward compatibility."""
        return self.view()

    def serve(self) -> None:
        """Build and serve the dashboard in the browser.

        Equivalent to R's ``shiny::runApp()`` / ``run_sae_app()``.

        """
        pn.serve(
            self.view(),
            port=self._config.port,
            show=self._config.open_browser,
        )

    def __repr__(self) -> str:
        return (
            f"App(title={self._config.title!r}, port={self._config.port}, "
            f"data_loaded={self._state['data'] is not None}, "
            f"model_fitted={self._state['model'] is not None})"
        )
