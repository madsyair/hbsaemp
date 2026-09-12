"""Tab 2 — Data exploration and EDA.
Mapping from R hbsaems
-----------------------
.. code-block:: text

    R Shiny                         Panel v1 equivalent
    ─────────────────────────────── ──────────────────────────────
    selectInput("explore_var_…")    pn.widgets.Select
    verbatimTextOutput("numeric_…") pn.widgets.Tabulator (describe())
    plotOutput("histogram_plot")    pn.pane.Matplotlib
    plotOutput("boxplot_plot")      pn.pane.Matplotlib
    plotOutput("scatter_plot")      pn.pane.Matplotlib
    renderUI("correlation_results") pn.widgets.Tabulator (formatted table)
    XICOR::xicor                    scipy.stats.chatterjeexi
    energy::dcor.test               dcor package
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import dcor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import panel as pn
import param
import seaborn as sns
from scipy.stats import chatterjeexi, pearsonr, spearmanr

if TYPE_CHECKING:
    from hbsaemp.app._app import AppState

__all__: list[str] = ["ExploreTab"]


class ExploreTab(param.Parameterized):
    state: AppState = param.Parameter()

    def __init__(self, state: AppState, **params: Any) -> None:
        super().__init__(state=state, **params)

        self._hist_var = pn.widgets.Select(name="Variable (Histogram)", options=[])
        self._n_bins   = pn.widgets.IntSlider(name="Number of bins", start=1, end=100, value=20)
        self._box_var  = pn.widgets.Select(name="Variable (Boxplot)", options=[])
        self._x_var    = pn.widgets.Select(name="X-axis Variable", options=[])
        self._y_var    = pn.widgets.Select(name="Y-axis Variable", options=[])
        self._x_trans  = pn.widgets.Select(
            name="X Transformation",
            options={"None": "none", "Log": "log", "Z-score": "zscore"},
            value="none",
        )
        self._y_trans  = pn.widgets.Select(
            name="Y Transformation",
            options={"None": "none", "Log": "log", "Logit": "logit", "Z-score": "zscore"},
            value="none",
        )

        self._summary_table = pn.widgets.Tabulator(pd.DataFrame(), show_index=False)
        self._hist_pane     = pn.pane.Matplotlib(sizing_mode="stretch_width", tight=True)
        self._box_pane      = pn.pane.Matplotlib(sizing_mode="stretch_width", tight=True)
        self._corr_title    = pn.pane.Markdown("")
        self._corr_table    = pn.widgets.Tabulator(pd.DataFrame(), show_index=False)
        self._scatter_pane  = pn.pane.Matplotlib(sizing_mode="stretch_width", tight=True)

        self.state.param.watch(self._on_data_change, "data")

        self._hist_var.param.watch(self._update_histogram, "value")
        self._n_bins.param.watch(self._update_histogram, "value")
        self._box_var.param.watch(self._update_boxplot, "value")
        for w in [self._x_var, self._y_var, self._x_trans, self._y_trans]:
            w.param.watch(self._update_scatter_corr, "value")

    def _on_data_change(self, event: param.parameterized.Event) -> None:
        df = self.state.data
        if df is None:
            return
        cols  = df.select_dtypes(include="number").columns.tolist()
        first = cols[0] if cols else None
        for w in [self._hist_var, self._box_var, self._x_var, self._y_var]:
            w.options = cols
            w.value   = first
        self._update_summary()
        self._update_histogram()
        self._update_boxplot()
        self._update_scatter_corr()

    def _numeric_df(self) -> pd.DataFrame | None:
        df = self.state.data
        return df.select_dtypes(include="number") if df is not None else None

    def _update_summary(self) -> None:
        df = self._numeric_df()
        if df is None:
            return
        self._summary_table.value = (
            df.describe().T.reset_index()
            .rename(columns={
                "index": "Variable", "min": "Min", "25%": "1st Qu.",
                "50%": "Median", "mean": "Mean", "75%": "3rd Qu.", "max": "Max",
            })
            .round(3)
        )

    def _update_histogram(self, *_: Any) -> None:
        df  = self._numeric_df()
        var = self._hist_var.value
        if df is None or not var:
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df[var].dropna(), bins=self._n_bins.value, kde=True, ax=ax)
        ax.set_title(f"Histogram of {var}")
        ax.set_xlabel(var)
        ax.set_ylabel("Density")
        plt.tight_layout()
        self._hist_pane.object = fig
        plt.close(fig)

    def _update_boxplot(self, *_: Any) -> None:
        df  = self._numeric_df()
        var = self._box_var.value
        if df is None or not var:
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(df[var].dropna(), ax=ax)
        ax.set_title(f"Boxplot of {var}")
        ax.set_xlabel(var)
        ax.set_ylabel("Value")
        plt.tight_layout()
        self._box_pane.object = fig
        plt.close(fig)

    def _transformed_xy(self) -> tuple[pd.Series, pd.Series, str, str] | None:
        df    = self._numeric_df()
        x_var = self._x_var.value
        y_var = self._y_var.value
        if df is None or not x_var or not y_var:
            return None
        x, y           = df[x_var].copy(), df[y_var].copy()
        x_name, y_name = x_var, y_var

        if self._x_trans.value == "log":
            x = np.log(np.maximum(x, 1e-6))
            x_name = f"{x_var}_log"
        elif self._x_trans.value == "zscore":
            x = (x - x.mean()) / x.std()
            x_name = f"{x_var}_zscore"

        if self._y_trans.value == "log":
            y = np.log(np.maximum(y, 1e-6))
            y_name = f"{y_var}_log"
        elif self._y_trans.value == "logit":
            y_clip = np.clip(y, 1e-6, 1 - 1e-6)
            y = np.log(y_clip / (1 - y_clip))
            y_name = f"{y_var}_logit"
        elif self._y_trans.value == "zscore":
            y = (y - y.mean()) / y.std()
            y_name = f"{y_var}_zscore"

        return x, y, x_name, y_name

    def _update_scatter_corr(self, *_: Any) -> None:
        res = self._transformed_xy()
        if res is None:
            return
        x, y, x_name, y_name = res

        mask   = x.notna() & y.notna()
        xn     = x[mask].to_numpy()
        yn     = y[mask].to_numpy()
        xi_res = chatterjeexi(xn, yn)

        rows = [
            ("Pearson's r",          *pearsonr(xn, yn)),
            ("Spearman's rho",       *spearmanr(xn, yn)),
            ("Chatterjee's Xi",      xi_res.statistic, xi_res.pvalue),
            ("Distance Correlation", dcor.distance_correlation(xn, yn), None),
        ]
        self._corr_title.object = f"### Correlation: **{x_name}** vs **{y_name}**"
        self._corr_table.value  = pd.DataFrame([
            {"Metric": name, "Value": round(val, 4),
             "p-value": round(pval, 4) if pval is not None else "N/A"}
            for name, val, pval in rows
        ])

        xc, yc = xn, yn
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(xc, yc, alpha=0.7)
        m, b   = np.polyfit(xc, yc, 1)
        x_line = np.linspace(xc.min(), xc.max(), 200)
        ax.plot(x_line, m * x_line + b, linestyle="--", color="red",
                linewidth=1.5, label="Trendline")
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.set_title(f"{x_name} vs {y_name}")
        ax.legend()
        plt.tight_layout()
        self._scatter_pane.object = fig
        plt.close(fig)

    def panel(self) -> pn.Column:
        """Return the Panel layout for this tab."""
        return pn.Column(
            pn.pane.Markdown("Explore dataset characteristics before modelling. Moving on to the tab <b>Modeling</b>."),
            pn.Tabs(
                (
                    "Summary Statistics",
                    pn.Card(self._summary_table, title="Summary Statistics", margin=10),
                ),
                (
                    "Visualize Distribution",
                    pn.FlexBox(
                        pn.Card(
                            pn.Column(self._hist_var, self._n_bins, self._hist_pane),
                            title="Histogram", margin=10, min_width=320,
                        ),
                        pn.Card(
                            pn.Column(self._box_var, self._box_pane),
                            title="Boxplot", margin=10, min_width=320,
                        ),
                    ),
                ),
                (
                    "Scatter & Correlation",
                    pn.Column(
                        pn.FlexBox(self._x_var, self._y_var),
                        pn.FlexBox(self._x_trans, self._y_trans),
                        self._corr_title,
                        self._corr_table,
                        self._scatter_pane,
                    ),
                ),
            ),
        )

    def __repr__(self) -> str:
        return "ExploreTab()"
