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
from hbsaemp.app._state import AppState
from hbsaemp.app.tabs.data_tab import DataTab
from hbsaemp.app.tabs.explore_tab import ExploreTab
from hbsaemp.app.tabs.model_tab import ModelTab
from hbsaemp.app.tabs.results_tab import ResultsTab

logger = get_logger(__name__)

pn.extension("tabulator", sizing_mode="stretch_width")

__all__: list[str] = ["App"]


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

        # Shared mutable state — tabs read/write to this dict.
        # Equivalent to Shiny reactive values (reactiveVal).
        self._state: AppState = AppState()

        # Instantiate tab controllers (stub objects in v0).
        self._data_tab    = DataTab(self._state)
        self._explore_tab = ExploreTab(self._state)
        self._model_tab   = ModelTab(self._state)
        self._results_tab = ResultsTab(self._state)

        logger.debug("App created: title=%r, port=%d", self._config.title, self._config.port)

    @property
    def config(self) -> AppConfig:
        """The :class:`AppConfig` for this application instance."""
        return self._config

    @property
    def state(self) -> AppState:
        """Shared :class:`~hbsaemp.app._state.AppState` (read-only view)"""
        return self._state

    def build(self) -> pn.template.FastListTemplate:
        """Assemble and return the Panel dashboard object (not yet served).
        Useful for embedding the app in a Jupyter notebook, or for calling
        ``.servable()`` on it inside a script launched with
        ``panel serve script.py``, without starting a server via
        :meth:`serve`.
 
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
        return pn.template.FastListTemplate(
            title=self._config.title,
            main=[tabs],
            accent="#A01346",
        )

    def serve(self) -> None:
        """Build and serve the dashboard in the browser.

        Equivalent to R's ``shiny::runApp()`` / ``run_sae_app()``.

        Raises:
            NotImplementedError: In v0.
        """
        pn.serve(
            self.build(),
            port=self._config.port,
            show=self._config.open_browser,
        )

    def __repr__(self) -> str:
        return (
            f"App(title={self._config.title!r}, port={self._config.port}, "
            f"data_loaded={self._state['data'] is not None}, "
            f"model_fitted={self._state['model'] is not None})"
        )
