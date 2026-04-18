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
from hbsaemp.app.tabs.data_tab import DataTab
from hbsaemp.app.tabs.explore_tab import ExploreTab
from hbsaemp.app.tabs.model_tab import ModelTab
from hbsaemp.app.tabs.results_tab import ResultsTab

logger = get_logger(__name__)

__all__: list[str] = ["App"]


class App:
    """hbsaemp web dashboard.

    v0: constructor works (creates stub tabs), but :meth:`serve` raises
    :exc:`NotImplementedError` because ``panel`` is not installed.

    v1: :meth:`serve` launches a Panel dashboard in the browser.

    Args:
        app_config: Appearance and server configuration.
            Defaults to :data:`~hbsaemp.app._config.DEFAULT_APP_CONFIG`.
    """

    def __init__(self, app_config: AppConfig | None = None) -> None:
        self._config: AppConfig = app_config or DEFAULT_APP_CONFIG

        # Shared mutable state — tabs read/write to this dict.
        # Equivalent to Shiny reactive values (reactiveVal).
        self._state: dict[str, Any] = {
            "data":  None,   # pd.DataFrame after upload
            "model": None,   # BaseModel after fitting
        }

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
    def state(self) -> dict[str, Any]:
        """Shared state dict (read-only view)."""
        return self._state

    def build(self) -> Any:
        """Assemble and return the Panel dashboard object (not yet served).

        In v1, returns a ``panel.template.FastListTemplate`` instance.
        Useful for embedding the app in a Jupyter notebook without launching
        a server.

        Returns:
            A ``panel`` servable object.

        Raises:
            NotImplementedError: In v0.
        """
        raise NotImplementedError(
            "App.build() requires panel>=1.3 (v1)."
        )

    def serve(self) -> None:
        """Build and serve the dashboard in the browser.

        Equivalent to R's ``shiny::runApp()`` / ``run_sae_app()``.

        Raises:
            NotImplementedError: In v0.
        """
        raise NotImplementedError(
            "App.serve() requires panel>=1.3 (v1)."
        )

    def __repr__(self) -> str:
        return (
            f"App(title={self._config.title!r}, port={self._config.port}, "
            f"data_loaded={self._state['data'] is not None}, "
            f"model_fitted={self._state['model'] is not None})"
        )
