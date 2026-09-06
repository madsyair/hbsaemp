"""Tab 1 — Data upload and preview.
Mapping from R hbsaems
-----------------------
.. code-block:: text

    R Shiny widget / output         Panel v1 equivalent
    ─────────────────────────────── ──────────────────────────────
    fileInput("data_file", …)       pn.widgets.FileInput
    selectInput("builtin_data", …)  pn.widgets.Select + load_dataset()
    DT::DTOutput("data_preview")    pn.widgets.Tabulator
    verbatimTextOutput("na_report") pn.pane.Markdown
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

import pandas as pd
import panel as pn
import param

from hbsaemp.data.datasets import AVAILABLE_DATASETS, load_dataset

if TYPE_CHECKING:
    from hbsaemp.app._app import AppState
    from hbsaemp.app._config import AppConfig

__all__: list[str] = ["DataTab"]


def _success_box(body: str) -> str:
    """Render a green success/notification banner as raw HTML."""
    return (
        f'<div style="background:#2ca02c;color:white;padding:10px 16px;'
        f'border-radius:8px;margin-top:8px">{body}</div>'
    )


def _error_box(title: str, body: str = "") -> str:
    inner = f"<b>{title}</b><br>{body}" if body else title
    return (
        f'<div style="background:#d62728;color:white;padding:10px 16px;'
        f'border-radius:8px;margin-top:8px">{inner}</div>'
    )


class DataTab(param.Parameterized):
    state: AppState = param.Parameter()

    def __init__(
        self,
        state: AppState,
        app_config: AppConfig | None = None,
        **params: Any,
    ) -> None:
        super().__init__(state=state, **params)
        self._app_config = app_config

        self._file_input = pn.widgets.FileInput(
            accept=".csv", name="Upload CSV File"
        )
        self._file_input.param.watch(self._on_upload, "value")

        self._dataset_sel = pn.widgets.Select(
            name="Or load a built-in dataset",
            options=[None, *AVAILABLE_DATASETS],
            value=None,
        )
        self._load_dataset_btn = pn.widgets.Button(
            name="Load Dataset", button_type="primary", max_width=160,
        )
        self._load_dataset_btn.on_click(self._on_load_dataset)

        self._summary       = pn.pane.Markdown("*No data uploaded yet.*")
        self._var_badges    = pn.pane.HTML("")
        self._preview       = pn.widgets.Tabulator(
            pd.DataFrame(), pagination="remote", page_size=10, show_index=False
        )
        self._upload_status = pn.pane.HTML("")

    def _max_upload_bytes(self) -> int | None:
        if self._app_config is None:
            return None
        return self._app_config.max_upload_mb * 1024 * 1024

    def _on_upload(self, event: param.parameterized.Event) -> None:
        if not self._file_input.value:
            return

        limit = self._max_upload_bytes()
        size = len(self._file_input.value)
        if limit is not None and size > limit:
            self._upload_status.object = _error_box(
                "File too large",
                f"{size / (1024 * 1024):.1f} MB exceeds the "
                f"{self._app_config.max_upload_mb} MB limit.",
            )
            return

        fname = self._file_input.filename
        try:
            raw = io.BytesIO(self._file_input.value)
            df = pd.read_csv(raw)
        except Exception as exc:
            self._upload_status.object = _error_box(
                "Could not read CSV file", str(exc)
            )
            return

        self._refresh_preview(df)
        self.state.data = df
        self._upload_status.object = _success_box(
            f"Data <b>{fname}</b> successfully loaded. "
            "Moving on to the tab <b>Data Exploration</b>."
        )

    def _on_load_dataset(self, event: Any) -> None:
        name = self._dataset_sel.value
        if not name:
            self._upload_status.object = _error_box(
                "Please select a built-in dataset first."
            )
            return
        try:
            df = load_dataset(name)
        except Exception as exc:
            self._upload_status.object = _error_box(
                "Could not load dataset", str(exc)
            )
            return

        self._refresh_preview(df)
        self.state.data = df
        self._upload_status.object = _success_box(
            f"Built-in dataset <b>{name}</b> successfully loaded. "
            "Moving on to the tab <b>Data Exploration</b>."
        )

    def _refresh_preview(self, df: pd.DataFrame) -> None:
        n_rows, n_cols = df.shape
        n_miss         = int(df.isna().sum().sum())
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
        self._preview.value     = df

    def panel(self) -> pn.Column:
        """Return the Panel layout for this tab."""
        guidelines = pn.pane.Markdown("""
                                      **Accepted formats:** `.csv`, or one of the built-in datasets below.

                                      **Dataset Structure:**
                                      - Tabular format; each row = one observation/area.
                                      - First row must contain column names (headers).
                                      - Include a column for direct estimates.
                                      - Include one or more auxiliary variables (predictors).
                                      - Optionally include a unique identifier column per area.

                                      **File format notes:**
                                      - `.csv`: comma (`,`) separator, period (`.`) decimal.

                                      **Missing Values:** Rows with missing values in the columns used will be automatically removed before modeling (`model.check_data()` in the Modeling tab).""")

        return pn.Column(
            pn.Card(guidelines,                            title="Data Requirements & Format Guidelines", margin=10),
            pn.Card(
                self._file_input, self._upload_status,
                pn.layout.Divider(),
                pn.Row(self._dataset_sel, self._load_dataset_btn),
                title="Upload File / Load Built-in Dataset", margin=10,
            ),
            pn.Card(self._summary,                          title="Summary Dataset",                       margin=10),
            pn.Card(self._var_badges,                        title="Variable List",                         margin=10),
            pn.Card(self._preview,                           title="Data Preview",                          margin=10),
        )

    def get_dataframe(self) -> pd.DataFrame | None:
        """Return the currently loaded :class:`pandas.DataFrame`, or ``None``."""
        return self.state.data

    def __repr__(self) -> str:
        loaded = self.state.data is not None if self.state is not None else False
        return f"DataTab(data_loaded={loaded})"
