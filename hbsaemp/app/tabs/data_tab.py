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

import io
 
import pandas as pd
import panel as pn
import param
 
from hbsaemp.app._helpers import success_box
from hbsaemp.app._state import AppState

__all__: list[str] = ["DataTab"]


class DataTab:
    """Data upload and preview tab.
 
    Args:
        state: Shared :class:`~hbsaemp.app._state.AppState`.  Writes
            ``state.data`` after a successful upload; read by downstream tabs.
    """
 
    state: AppState = param.Parameter()

    def __init__(self, state: AppState, **params) -> None:
        super().__init__(state=state, **params)
 
        self._file_input = pn.widgets.FileInput(
            accept=".csv", name="Upload CSV File"
        )
        self._file_input.param.watch(self._on_upload, "value")
 
        self._summary       = pn.pane.Markdown("*No data uploaded yet.*")
        self._var_badges    = pn.pane.HTML("")
        self._preview       = pn.widgets.Tabulator(
            pd.DataFrame(), pagination="remote", page_size=10, show_index=False
        )
        self._upload_status = pn.pane.HTML("")

    def _on_upload(self, event) -> None:
        if not self._file_input.value:
            return
        fname = self._file_input.filename
        raw   = io.BytesIO(self._file_input.value)
        df    = pd.read_csv(raw) 
 
        self._refresh_preview(df)
        self.state.data = df
        self._upload_status.object = success_box(
            f"✔ Data <b>{fname}</b> successfully loaded. "
            "Moving on to the tab <b>Data Exploration</b>."
        )
 
    def _refresh_preview(self, df: pd.DataFrame) -> None:
        n_rows, n_cols = df.shape
        n_miss = int(df.isna().sum().sum())
        self._summary.object = (
            f"**Total Rows:** {n_rows} &nbsp;|&nbsp; "
            f"**Total Columns:** {n_cols} &nbsp;|&nbsp; "
            f"**Missing Values:** {n_miss}"
        )
        badges = " ".join(
            f'<span style="background:#0072B2;color:white;padding:4px 10px;'
            f'border-radius:20px;margin:3px;display:inline-block">{c}</span>'
            for c in df.columns
        )
        self._var_badges.object = badges
        self._preview.value = df
 
    def panel(self) -> pn.Column:
        """Return the Panel layout for this tab."""
        guidelines = pn.pane.Markdown("""
                                      **Accepted formats:** `.csv`

                                      **Dataset Structure:**
                                      - Tabular format; each row = one observation/area.
                                      - First row must contain column names (headers).
                                      - Include a column for direct estimates.
                                      - Include one or more auxiliary variables (predictors).
                                      - Optionally include a unique identifier column per area.

                                      **File format notes:**
                                      - `.csv`: comma (`,`) separator, period (`.`) decimal.
                                      
                                      **Missing Values:** Rows with missing values ​​in the columns used will be automatically removed before modeling.""")
 
        return pn.Column(
            pn.Card(guidelines,                            title="Data Requirements & Format Guidelines", margin=10),
            pn.Card(self._file_input, self._upload_status, title="Upload File",                           margin=10),
            pn.Card(self._summary,                          title="Summary Dataset",                       margin=10),
            pn.Card(self._var_badges,                       title="Variable List",                         margin=10),
            pn.Card(self._preview,                          title="Data Preview",                          margin=10),
        )

    def __repr__(self) -> str:
        return "DataTab(data_loaded={self.state.data is not None})"
