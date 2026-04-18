"""Tab 1 — Data upload and preview.

Equivalent to the "Data Upload" tab in R hbsaems Shiny app.

v0: stub class, raises NotImplementedError.
v1: implemented with Panel FileInput + Tabulator.

Features (v1)
--------------
* Upload a CSV or Excel file via drag-and-drop or file browser.
* Select one of the five built-in hbsaemp datasets.
* Preview data in an interactive sortable/filterable table.
* Report missing values per column.
* Report basic data types and shape.

Mapping from R hbsaems
-----------------------
.. code-block:: text

    R Shiny widget / output         Panel v1 equivalent
    ─────────────────────────────── ──────────────────────────────
    fileInput("data_file", …)       pn.widgets.FileInput
    DT::DTOutput("data_preview")    pn.widgets.Tabulator
    verbatimTextOutput("na_report") pn.pane.Str
    selectInput("builtin", …)       pn.widgets.Select (builtin datasets)
"""

from __future__ import annotations

from typing import Any

__all__: list[str] = ["DataTab"]


class DataTab:
    """Data upload and preview tab.

    v0: All methods raise :exc:`NotImplementedError`.
    v1: Implemented with ``panel`` library.

    Args:
        app_state: Shared mutable application state dict passed by
            :class:`~hbsaemp.app._app.App`.  The ``"data"`` key is
            written here and read by downstream tabs.
    """

    def __init__(self, app_state: dict[str, Any]) -> None:
        self._state = app_state

    def panel(self) -> Any:
        """Return a Panel layout object for this tab.

        Returns:
            A ``panel`` layout object (``pn.Column`` or similar).

        Raises:
            NotImplementedError: In v0.  Requires ``panel>=1.3`` (v1).
        """
        raise NotImplementedError(
            "DataTab.panel() requires panel>=1.3 (v1)."
        )

    def get_dataframe(self) -> Any:
        """Return the currently loaded :class:`pandas.DataFrame`, or ``None``.

        Raises:
            NotImplementedError: In v0.
        """
        raise NotImplementedError("DataTab.get_dataframe() requires v1.")

    def __repr__(self) -> str:
        return "DataTab(status=stub)"
