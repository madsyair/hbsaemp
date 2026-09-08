"""Tab 4 — Results: diagnostics, SAE estimates, and export.
Mapping from R hbsaems
-----------------------
.. code-block:: text

    R Shiny output                          Panel v1 equivalent
    ─────────────────────────────────────── ──────────────────────────────────
    verbatimTextOutput("diag_numerical")    pn.widgets.Tabulator (rhat_ess)
    plotOutput("diag_plots")                pn.pane.Matplotlib (ConvergenceResult.plots)
    DT::dataTableOutput("sae_table")        pn.widgets.Tabulator (AreaEstimatesResult.result_table)
    downloadButton("download_estimates")    pn.widgets.FileDownload
"""

from __future__ import annotations

import asyncio
import io
import warnings
from typing import TYPE_CHECKING, Any

import pandas as pd
import panel as pn
import param

from hbsaemp import (
    ConvergenceResult,
    ConvergenceWarning,
    DataValidationError,
    EstimationError,
    HBSAEError,
    ModelNotFittedError,
    check_convergence,
    estimate_areas,
)
from hbsaemp._logging import get_logger

if TYPE_CHECKING:
    from hbsaemp.app._app import AppState
    from hbsaemp.estimation.areas import AreaEstimatesResult

logger = get_logger(__name__)

__all__: list[str] = ["ResultsTab"]

_PLOT_TITLES: dict[str, str] = {
    "trace":  "Trace Plot",
    "dens":   "Density Plot",
    "acf":    "Autocorrelation Plot",
    "rhat":   "R-hat Forest Plot",
    "neff":   "Effective Sample Size Plot",
    "energy": "NUTS Energy / BFMI Plot",
}
_PLOT_ORDER: tuple[str, ...] = ("trace", "dens", "acf", "rhat", "neff", "energy")


def _error_box(title: str, body: str = "") -> str:
    inner = f"<b>{title}</b><br>{body}" if body else title
    return (
        f'<div style="background:#d62728;color:white;padding:10px 16px;'
        f'border-radius:8px;margin-top:8px">{inner}</div>'
    )


def _success_box(body: str) -> str:
    return (
        f'<div style="background:#2ca02c;color:white;padding:10px 16px;'
        f'border-radius:8px;margin-top:8px">{body}</div>'
    )


def _warn_box(body: str) -> str:
    return (
        f'<div style="background:#f0ad4e;color:#3a2e00;padding:10px 16px;'
        f'border-radius:8px;margin-top:8px">{body}</div>'
    )


def _info_box(body: str) -> str:
    return (
        f'<div style="background:#0072B2;color:white;padding:10px 16px;'
        f'border-radius:8px;margin-top:8px">{body}</div>'
    )


def _describe_error(exc: Exception) -> tuple[str, str]:
    """Map a backend exception to a (title, body) pair for the UI.

    Mirrors :func:`hbsaemp.app.tabs.model_tab._describe_error` — kept as a
    small local copy rather than a shared cross-tab helper module (see
    ``app/`` package layout).
    """
    if isinstance(exc, ModelNotFittedError):
        return "Model has not been fitted", "Please fit the model in the Modeling tab first."
    if isinstance(exc, DataValidationError):
        return "Data does not meet the family's requirements", str(exc)
    if isinstance(exc, EstimationError):
        return "Estimation failed", str(exc)
    if isinstance(exc, ImportError):
        return "Missing dependency", str(exc)
    if isinstance(exc, HBSAEError):
        return "Backend error", str(exc)
    return "Unexpected error", str(exc)


class ResultsTab(param.Parameterized):
    """Results display tab with convergence diagnostics and SAE estimates.

    Args:
        state: Shared :class:`~hbsaemp.app._app.AppState` instance. Reads
            ``state.model`` — a fitted
            :class:`~hbsaemp.models._base.BaseModel` written by
            :class:`~hbsaemp.app.tabs.model_tab.ModelTab` after fitting.
    """

    state: AppState = param.Parameter()

    def __init__(self, state: AppState, **params: Any) -> None:
        super().__init__(state=state, **params)

        self._conv_run_btn   = pn.widgets.Button(
            name="Load Convergence Diagnostics", button_type="primary", max_width=280,
        )
        self._conv_status    = pn.pane.HTML("")
        self._rhat_ess_table = pn.widgets.Tabulator(
            pd.DataFrame(), show_index=False, pagination="remote", page_size=15,
        )
        self._plots_pane     = pn.Column()
        self._conv_run_btn.on_click(self._on_load_convergence)

        self._sae_run_btn      = pn.widgets.Button(
            name="Run SAE Estimation", button_type="primary", max_width=280,
        )
        self._sae_status       = pn.pane.HTML("")
        self._sae_table        = pn.widgets.Tabulator(
            pd.DataFrame(), show_index=False, pagination="remote", page_size=20,
        )
        self._sae_download_btn = pn.widgets.FileDownload(
            label="Download CSV of SAE Results",
            filename="sae_estimation.csv",
            callback=self._sae_csv_callback,
            button_type="success",
            max_width=280,
            disabled=True,
        )
        self._sae_run_btn.on_click(self._on_run_sae_estimation)

    async def _on_load_convergence(self, event: Any) -> None:
        if self._conv_run_btn.loading:  
            return
        model = getattr(self.state, "model", None)
        if model is None or not model.is_fitted:
            self._conv_status.object = _error_box(
                "Model has not been fitted",
                "Please click <b>Fit Model</b> in the <b>Modeling</b> tab first.",
            )
            return

        self._conv_run_btn.loading = self._conv_run_btn.disabled = True
        self._conv_status.object = _info_box("Computing convergence diagnostics…")
        try:
            loop = asyncio.get_running_loop()
            result, caught_warnings = await loop.run_in_executor(
                None, self._convergence_blocking, model
            )
        except Exception as exc:
            logger.exception("ResultsTab convergence check failed")
            title, body = _describe_error(exc)
            self._conv_status.object = _error_box(title, body)
            return
        finally:
            self._conv_run_btn.loading = self._conv_run_btn.disabled = False

        self._render_convergence(result, caught_warnings)

    @staticmethod
    def _convergence_blocking(
        model: Any,
    ) -> tuple[ConvergenceResult, list[str]]:
        """Synchronous, Panel-free — runs in the executor thread.

        ``ConvergenceWarning`` is a real warning from ``check_convergence()``
        (R-hat/ESS out of threshold) and must reach the UI, not be silently
        dropped — caught here (task #22) and returned alongside the result.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            result = check_convergence(model)
        messages = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
        return result, messages

    def _render_convergence(self, result: ConvergenceResult, warning_messages: list[str]) -> None:
        if result.rhat_ess is not None and not result.rhat_ess.empty:
            self._rhat_ess_table.value = result.rhat_ess.reset_index().rename(
                columns={"index": "Parameter"}
            )

        sections: list[tuple[str, Any]] = []
        for ptype in _PLOT_ORDER:
            title = _PLOT_TITLES.get(ptype, ptype)
            if ptype in result.plots:
                content = pn.pane.Matplotlib(
                    result.plots[ptype], sizing_mode="stretch_width", tight=True, max_width=900,
                )
            elif ptype in result.plot_errors:
                content = pn.pane.HTML(
                    _error_box(f"Could not render the {title.lower()}", result.plot_errors[ptype])
                )
            else:
                continue
            sections.append((title, pn.Column(content, sizing_mode="stretch_width")))
        self._plots_pane.objects = [
            pn.Accordion(*sections, active=[], sizing_mode="stretch_width")
        ] if sections else [pn.pane.Markdown("*No plots were generated.*")]

        if warning_messages:
            self._conv_status.object = _warn_box(
                "<b>Convergence issues detected:</b><br>"
                + "<br>".join(f"• {m}" for m in warning_messages)
                + "<br><br>Consider reviewing the model specification or the data."
            )
        else:
            self._conv_status.object = _success_box(
                "Diagnostics computed — no convergence warnings raised."
            )

    async def _on_run_sae_estimation(self, event: Any) -> None:
        if self._sae_run_btn.loading: 
            return
        model = getattr(self.state, "model", None)
        if model is None or not model.is_fitted:
            self._sae_status.object = _error_box(
                "Model has not been fitted",
                "Please fit the model in the <b>Modeling</b> tab first.",
            )
            return

        self._sae_run_btn.loading = self._sae_run_btn.disabled = True
        self._sae_status.object = _info_box("Computing SAE estimates…")
        try:
            loop = asyncio.get_running_loop()
            result: AreaEstimatesResult = await loop.run_in_executor(
                None, self._sae_blocking, model
            )
        except Exception as exc:
            logger.exception("ResultsTab SAE estimation failed")
            title, body = _describe_error(exc)
            self._sae_status.object = _error_box(title, body)
            return
        finally:
            self._sae_run_btn.loading = self._sae_run_btn.disabled = False

        self._sae_table.value = result.result_table
        self._sae_download_btn.disabled = False
        self._sae_status.object = _success_box(
            f"SAE estimation complete — mean RSE: <b>{result.mean_rse:.2f}%</b>, "
            f"mean MSE: <b>{result.mean_mse:.4f}</b>."
        )

    @staticmethod
    def _sae_blocking(model: Any) -> AreaEstimatesResult:
        """Synchronous, Panel-free — runs in the executor thread."""
        return estimate_areas(model, ci_prob=0.95)

    def _sae_csv_callback(self) -> io.StringIO:
        buf = io.StringIO()
        self._sae_table.value.to_csv(buf, index=False)
        buf.seek(0)
        return buf

    def panel(self) -> pn.Tabs:
        """Return the Panel layout for this tab."""
        convergenceevaluation_card = pn.Card(
            pn.Column(
                pn.pane.Markdown(
                    "MCMC convergence evaluation via `check_convergence()`: R-hat, "
                    "Effective Sample Size (ESS), trace, autocorrelation, and density "
                    "plots.",
                    margin=(4, 0, 8, 0),
                ),
                self._conv_run_btn,
                self._conv_status,
                pn.layout.Divider(),
                pn.Accordion(
                    (
                        "R-hat and ESS",
                        pn.Column(
                            pn.pane.Markdown(
                                "R-hat close to 1 indicates the chains have converged. "
                                "Values above 1.01, or an ESS below 400, trigger a "
                                "`ConvergenceWarning` from the backend.",
                                margin=(4, 0, 8, 0),
                            ),
                            self._rhat_ess_table,
                            sizing_mode="stretch_width",
                        ),
                    ),
                    active=[], sizing_mode="stretch_width", margin=(8, 0),
                ),
                self._plots_pane,
            ),
            title="MCMC Convergence Evaluation",
            margin=10,
        )

        saeestimation_card = pn.Card(
            pn.Column(
                self._sae_run_btn,
                self._sae_status,
                pn.layout.Divider(),
                self._sae_table,
                pn.Row(self._sae_download_btn),
            ),
            title="SAE Estimation",
            margin=10,
        )

        return pn.Tabs(
            ("Convergence Evaluation", convergenceevaluation_card),
            ("SAE Estimation",         saeestimation_card),
            sizing_mode="stretch_width",
        )

    def __repr__(self) -> str:
        model = getattr(self.state, "model", None)
        has_results = bool(model is not None and getattr(model, "is_fitted", False))
        return f"ResultsTab(has_results={has_results})"
