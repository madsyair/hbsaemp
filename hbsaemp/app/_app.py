"""Main application class for the hbsaemp web dashboard.
Mapping from R hbsaems
-----------------------
.. code-block:: text

    R Shiny structure                  Panel v1 equivalent
    ─────────────────────────────────  ────────────────────────────────────
    dashboardPage(header, sidebar, …)  pn.template.FastListTemplate(...)
    dashboardSidebar(sidebarMenu(…))   sidebar= parameter of template
    dashboardBody(tabItems(…))         main= parameter of template
    reactiveValues(data=NULL, …)       AppState(param.Parameterized)
    observe({ if model_fit… })         pn.param.watch callback
"""

from __future__ import annotations

import panel as pn
import param

from hbsaemp._logging import configure_logging, get_logger
from hbsaemp.app._config import DEFAULT_APP_CONFIG, AppConfig
from hbsaemp.app.tabs import DataTab, ExploreTab, ModelTab, ResultsTab, UpdateModelTab

logger = get_logger(__name__)

pn.extension("tabulator", sizing_mode="stretch_width")

__all__: list[str] = ["App", "AppState"]

_THEME_MAP: dict[str, type] = {}


def _resolve_theme(name: str) -> type:
    """Panel ``Theme`` class for *name* (`"dark"` -> DarkTheme, else default)."""
    from panel.theme import DarkTheme, DefaultTheme

    if not _THEME_MAP:
        _THEME_MAP.update({"dark": DarkTheme, "bootstrap": DefaultTheme})
    return _THEME_MAP.get(name, DefaultTheme)


class AppState(param.Parameterized):
    """Shared reactive state passed to every tab.

    Tabs read/write these parameters and use ``param.watch`` to react to
    changes made by upstream tabs (e.g. :class:`~hbsaemp.app.tabs.data_tab.DataTab`
    writing ``data`` triggers refreshes in
    :class:`~hbsaemp.app.tabs.explore_tab.ExploreTab` and
    :class:`~hbsaemp.app.tabs.model_tab.ModelTab`).

    Attributes:
        data: Uploaded/loaded :class:`pandas.DataFrame`, or ``None``.
        model: A :class:`~hbsaemp.models._base.BaseModel` written by
            :class:`~hbsaemp.app.tabs.model_tab.ModelTab` after a successful
            :meth:`~hbsaemp.models._base.BaseModel.fit`. ``model.is_fitted``
            tells :class:`~hbsaemp.app.tabs.results_tab.ResultsTab` whether
            results are ready; ``model.result`` raises
            :class:`~hbsaemp._exceptions.ModelNotFittedError` otherwise.
    """

    data:  object | None = param.Parameter(default=None)
    model: object | None = param.Parameter(default=None)


class App:
    """hbsaemp web dashboard.

    Assembles :class:`~hbsaemp.app.tabs.data_tab.DataTab`,
    :class:`~hbsaemp.app.tabs.explore_tab.ExploreTab`,
    :class:`~hbsaemp.app.tabs.model_tab.ModelTab`, and
    :class:`~hbsaemp.app.tabs.results_tab.ResultsTab` around one shared
    :class:`AppState` instance, and exposes a servable Panel dashboard via
    :meth:`view`.

    Args:
        app_config: Appearance and server configuration.
            Defaults to :data:`~hbsaemp.app._config.DEFAULT_APP_CONFIG`.
    """

    def __init__(self, app_config: AppConfig | None = None) -> None:
        self._config: AppConfig = app_config or DEFAULT_APP_CONFIG

        configure_logging(level=self._config.log_level)

        self._state = AppState()

        self._data_tab    = DataTab(state=self._state, app_config=self._config)
        self._explore_tab = ExploreTab(state=self._state)
        self._model_tab   = ModelTab(state=self._state)
        self._results_tab = ResultsTab(state=self._state)
        self._update_tab  = UpdateModelTab(state=self._state)

        logger.debug(
            "App created: title=%r, port=%d", self._config.title, self._config.port
        )

    @property
    def config(self) -> AppConfig:
        """The :class:`AppConfig` for this application instance."""
        return self._config

    @property
    def state(self) -> AppState:
        """Shared :class:`AppState` instance (read-only view)."""
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
            ("Update Model",     self._update_tab.panel()),
            sizing_mode="stretch_width",
        )
        sidebar = [
            pn.Column(
                pn.pane.Markdown(
                    "The `launch_app()` function in the `hbsaemp` package provides an interactive dashboard "
                    "for Hierarchical Bayesian Small Area Estimation (HBSAE) in Python. The application offers " \
                    "a user-friendly graphical interface that allows users to upload data, specify models, and obtain " \
                    "estimation results without requiring extensive Python programming."
                ),
                pn.layout.Divider(),
                pn.pane.Markdown(
                    "**Workflow**\n"
                    "1. **Data Upload**: Upload a CSV file or load a built-in dataset.\n"
                    "2. **Data Exploration**: View summary statistics, data distributions, and correlations.\n"
                    "3. **Modeling**: Select variables, choose the model family, check priors and posterior predictions, and fit the model.\n"
                    "4. **Results**: Review convergence diagnostics and download the SAE estimates.\n"
                    "5. **Update Model**: Refit with different sampler settings or replacement data, without starting over."
                ),
                sizing_mode="stretch_width",
            )
        ] if self._config.show_sidebar else []

        return pn.template.FastListTemplate(
            title=self._config.title,
            sidebar=sidebar,
            main=[tabs],
            accent=self._config.accent,
            theme=_resolve_theme(self._config.theme),
        )

    def build(self) -> pn.template.FastListTemplate:
        """Alias for :meth:`view`, kept for backward compatibility."""
        return self.view()

    def serve(self) -> None:
        pn.serve(
            self.view,
            port=self._config.port,
            show=self._config.open_browser,
        )

    def __repr__(self) -> str:
        return (
            f"App(title={self._config.title!r}, port={self._config.port}, "
            f"data_loaded={self._state.data is not None}, "
            f"model_fitted={getattr(self._state.model, 'is_fitted', False)})"
        )
