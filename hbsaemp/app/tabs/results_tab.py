"""Tab 4 — Results: diagnostics, model comparison, refit, and export.
Sub-tabs (in workflow order):

1. **Convergence Evaluation** — R-hat/ESS table and diagnostic plots from
   `check_convergence()`, shown as-is. The GUI does not judge or gate on
   convergence anywhere — it's up to whoever is reading these numbers to
   decide whether a fit is usable.
2. **Model Comparison** — compare models saved from the Modeling tab's
   "Save Model" button, via `compare_models()`.
3. **Update Model** — refit the model currently in use (`state.model`)
   via `update_model()`, without rebuilding it from scratch. Embeds
   :class:`~hbsaemp.app.tabs.update_tab.UpdateModelTab`.
4. **SAE Estimation** — small-area estimates from `estimate_areas()`,
   including out-of-sample prediction for unsampled areas.
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
    ComparisonResult,
    ConvergenceResult,
    ConvergenceWarning,
    DataValidationError,
    EstimationError,
    HBSAEError,
    ModelNotFittedError,
    check_convergence,
    compare_models,
    estimate_areas,
)
from hbsaemp._logging import get_logger
from hbsaemp.app.tabs.update_tab import UpdateModelTab

if TYPE_CHECKING:
    from hbsaemp.app._app import AppState
    from hbsaemp.estimation.areas import AreaEstimatesResult

logger = get_logger(__name__)

__all__: list[str] = ["ResultsTab"]

_PLOT_TITLES: dict[str, str] = {
    "trace":  "Trace Plot",
    "dens":   "Density Plot",
    "acf":    "Autocorrelation Plot",
    "rhat":   "R-hat Distribution Plot",
    "neff":   "Effective Sample Size Plot",
    "energy": "NUTS Energy / BFMI Plot",
}
_PLOT_ORDER: tuple[str, ...] = ("trace", "dens", "acf", "rhat", "neff", "energy")

# Display-only rounding for the R-hat/ESS table. check_convergence() itself
# returns full-precision floats (as it should, for the internal >1.01
# comparison) — this only affects what the Tabulator shows.
_RHAT_ESS_DISPLAY_ROUND: dict[str, int] = {
    "mean": 3, "sd": 3, "r_hat": 3, "ess_bulk": 0, "ess_tail": 0,
}


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
    if isinstance(exc, ValueError):
        return "Invalid setting", str(exc)
    if isinstance(exc, ImportError):
        return "Missing dependency", str(exc)
    if isinstance(exc, HBSAEError):
        return "Backend error", str(exc)
    return "Unexpected error", str(exc)


def _matplotlib_pane(fig: Any, *, max_width: int = 850) -> pn.pane.Matplotlib:
    """A ``Matplotlib`` pane sized to *fig*'s own aspect ratio.

    ``sizing_mode="stretch_width"`` alone leaves height to Panel's default,
    which does not track a wide/short figure's actual proportions.
    Deriving both `width` and `height` directly from
    ``fig.get_size_inches()`` guarantees the pane's box matches the image
    exactly, with no leftover space.
    """
    fig_w, fig_h = fig.get_size_inches()
    width = min(max_width, int(fig_w * 100))
    height = int(width * (fig_h / fig_w))
    return pn.pane.Matplotlib(fig, width=width, height=height, tight=True)


def _plot_errors_section(plot_errors: dict[str, str], titles: dict[str, str]) -> list[Any]:
    return [
        pn.pane.HTML(_error_box(f"Could not render the {titles.get(k, k).lower()}", msg))
        for k, msg in plot_errors.items()
    ]


class ResultsTab(param.Parameterized):
    """Results tab: convergence, model comparison, refit, and SAE estimates.

    Args:
        state: Shared :class:`~hbsaemp.app._app.AppState` instance. Reads
            ``state.model`` (written by
            :class:`~hbsaemp.app.tabs.model_tab.ModelTab` after fitting)
            and ``state.saved_models`` (written by that tab's "Save Model"
            button).
    """

    state: AppState = param.Parameter()

    def __init__(self, state: AppState, **params: Any) -> None:
        super().__init__(state=state, **params)

        # Update Model lives here as an embedded sub-tab, not a separate
        # top-level tab — it, Comparison, Convergence, and SAE Estimation
        # are all "things you do with the model currently in use".
        self._update_tab = UpdateModelTab(state=state)

        self._build_convergence_widgets()
        self._build_comparison_widgets()
        self._build_sae_widgets()

        # S1 (estimation-frontend.md): a refit anywhere (Update Model) must
        # not leave stale convergence/SAE numbers on screen.
        self.state.param.watch(self._on_model_change, "model")

    # State change handling

    def _on_model_change(self, event: param.parameterized.Event) -> None:
        self._sae_table.value = pd.DataFrame()
        self._sae_notes.object = ""
        self._rhat_ess_table.value = pd.DataFrame()
        self._plots_pane.objects, self._last_plots = [], {}
        self._sae_download_btn.disabled = True
        self._plots_download_btn.disabled = True
        msg = _info_box("Model changed — run again to refresh the results.")
        self._sae_status.object = msg
        self._conv_status.object = msg

    # Convergence Evaluation

    def _build_convergence_widgets(self) -> None:
        self._conv_run_btn = pn.widgets.Button(
            name="Load Convergence Diagnostics", button_type="primary", max_width=280,
        )
        self._conv_status = pn.pane.HTML("")
        self._rhat_ess_desc = pn.pane.Markdown("", margin=(4, 0, 8, 0))
        self._rhat_ess_table = pn.widgets.Tabulator(
            pd.DataFrame(), show_index=False, pagination="remote", page_size=15,
        )
        self._plots_pane = pn.Column()
        self._last_plots: dict[str, Any] = {}
        self._plots_download_btn = pn.widgets.FileDownload(
            label="Download Plots (PDF)",
            filename="convergence_plots.pdf",
            callback=self._plots_pdf_callback,
            button_type="success",
            max_width=280,
            disabled=True,
        )
        self._conv_run_btn.on_click(self._on_load_convergence)

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
            result = await loop.run_in_executor(
                None, self._convergence_blocking, model
            )
        except Exception as exc:
            logger.exception("ResultsTab convergence check failed")
            title, body = _describe_error(exc)
            self._conv_status.object = _error_box(title, body)
            return
        finally:
            self._conv_run_btn.loading = self._conv_run_btn.disabled = False

        self._render_convergence(result)

    @staticmethod
    def _convergence_blocking(model: Any) -> ConvergenceResult:
        """Synchronous, Panel-free — runs in the executor thread.

        `check_convergence()` raises `ConvergenceWarning` through Python's
        own warnings machinery; caught here only so it doesn't print to
        the server's console — not used to render a verdict. The GUI
        shows the R-hat/ESS/divergence numbers as-is and leaves the
        judgment of "did this converge" to whoever is reading them.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            return check_convergence(model)

    def _render_convergence(self, result: ConvergenceResult) -> None:
        # diagnostics-frontend.md S2: thresholds come from the result, not
        # hardcoded text (R-hat rank-normalized threshold is 1.01; ESS
        # threshold scales with chain count). Stated so the reader can
        # judge convergence themselves — the GUI does not.
        self._rhat_ess_desc.object = (
            "The **R-hat (Gelman-Rubin)** diagnostic compares within-chain to "
            "between-chain variance. Values noticeably above `1` suggest the "
            f"chains have not mixed adequately. Commonly used thresholds: "
            f"**R-hat ≤ 1.01**, **ESS ≥ {result.ess_threshold}** "
            "(100 × number of chains) — interpret against your own study's "
            "requirements."
        )

        if result.rhat_ess is not None and not result.rhat_ess.empty:
            # diagnostics-frontend.md S1: full precision internally, rounded
            # only for display.
            display_cols = {
                c: n for c, n in _RHAT_ESS_DISPLAY_ROUND.items() if c in result.rhat_ess.columns
            }
            table = result.rhat_ess.round(display_cols) if display_cols else result.rhat_ess
            self._rhat_ess_table.value = table.reset_index().rename(columns={"index": "Parameter"})

        self._last_plots = dict(result.plots)
        self._plots_download_btn.disabled = not self._last_plots

        sections: list[tuple[str, Any]] = []
        for ptype in _PLOT_ORDER:
            title = _PLOT_TITLES.get(ptype, ptype)
            if ptype in result.plots:
                content = _matplotlib_pane(result.plots[ptype])
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

        self._conv_status.object = _success_box("Diagnostics computed.")

    def _plots_pdf_callback(self) -> io.BytesIO:
        from matplotlib.backends.backend_pdf import PdfPages

        buf = io.BytesIO()
        with PdfPages(buf) as pdf:
            for ptype in _PLOT_ORDER:
                fig = self._last_plots.get(ptype)
                if fig is not None:
                    pdf.savefig(fig)
        buf.seek(0)
        return buf

    # Model Comparison

    def _build_comparison_widgets(self) -> None:
        self._compare_table = pn.widgets.Tabulator(
        # `columns=["Model"]` alone gives an empty column typed float64
        # (pandas' default for an empty column), which makes Tabulator.js
        # treat it as a numeric column — a later string value then
        # renders as NaN instead of the model's name. Force the dtype
        # explicitly so it's a text column from the start.
        pd.DataFrame({"Model": pd.Series(dtype="object")}),
        show_index=False,
        selectable="checkbox",
        disabled=True,
)
        self._compare_bf_cb = pn.widgets.Checkbox(
            name="Include Bayes Factor (Savage-Dickey, per coefficient)", value=False,
        )
        self._compare_psense_cb = pn.widgets.Checkbox(
            name="Include Prior Sensitivity (power-scaling, per parameter)", value=False,
        )
        self._compare_run_btn = pn.widgets.Button(
            name="Compare Selected", button_type="primary", max_width=220,
        )
        self._compare_status = pn.pane.HTML("")
        self._compare_result_pane = pn.Column()
        self._use_model_sel = pn.widgets.Select(name="Use this model", options=[], max_width=220)
        self._use_model_btn = pn.widgets.Button(
            name="Use Selected Model", button_type="success", max_width=220, disabled=True,
        )
        self._use_model_status = pn.pane.HTML("")

        self._compare_run_btn.on_click(self._on_compare_click)
        self._use_model_btn.on_click(self._on_use_model_click)
        self.state.param.watch(self._on_saved_models_change, "saved_models")
        self._refresh_saved_models_table()

    def _on_saved_models_change(self, event: param.parameterized.Event) -> None:
        self._refresh_saved_models_table()

    def _refresh_saved_models_table(self) -> None:
        saved = self.state.saved_models or {}
        # dtype="object" for the same reason as the initial construction —
        # pd.DataFrame({"Model": []}) on an empty dict would default to
        # float64 and make Tabulator.js render later names as NaN.
        self._compare_table.value = pd.DataFrame(
        {"Model": pd.Series(list(saved.keys()), dtype="object")}
        )
        self._use_model_sel.options = [None, *saved.keys()]

    async def _on_compare_click(self, event: Any) -> None:
        if self._compare_run_btn.loading:
            return
        saved = self.state.saved_models or {}
        selected_rows = self._compare_table.value.iloc[self._compare_table.selection]
        names = selected_rows["Model"].tolist() if not selected_rows.empty else []

        if not names:
            self._compare_status.object = _error_box(
                "Select a model",
                "Check one or more models above, then click Compare Selected. "
                "A single model shows its own diagnostics (LOO, Bayes Factor if "
                "enabled); two or more also add a side-by-side ranking.",
            )
            return

        models = [saved[n].model for n in names]
        with_bf = self._compare_bf_cb.value
        with_psense = self._compare_psense_cb.value

        self._compare_run_btn.loading = self._compare_run_btn.disabled = True
        self._compare_status.object = _info_box(f"Comparing {len(models)} model(s)…")
        try:
            loop = asyncio.get_running_loop()
            result: ComparisonResult = await loop.run_in_executor(
                None, self._compare_blocking, models, with_bf, with_psense
            )
        except Exception as exc:
            logger.exception("ResultsTab model comparison failed")
            title, body = _describe_error(exc)
            self._compare_status.object = _error_box(title, body)
            return
        finally:
            self._compare_run_btn.loading = self._compare_run_btn.disabled = False

        self._render_comparison(result, names)

    @staticmethod
    def _compare_blocking(
        models: list[Any], with_bf: bool = False, with_psense: bool = False,
    ) -> ComparisonResult:
        """Synchronous, Panel-free — runs in the executor thread.

        ``metrics=["loo", "bf"]`` and ``run_prior_sensitivity=True`` are
        both opt-in (checkboxes default off) rather than always-on. Not
        because either is unsafe — the backend's ``_bf_frame()`` correctly
        handles the ``xarray.Dataset`` shape ``az.bayes_factor()`` returns
        since arviz-stats 1.2.0, which is this project's pinned floor
        (``arviz>=1.2.0`` in pyproject.toml) — but because they're extra
        statistics most comparisons don't need, and each costs another
        `prior_predictive_idata()`-adjacent sampling pass per model on
        top of plain LOO.
        """
        metrics = ["loo", "bf"] if with_bf else None
        return compare_models(models, metrics=metrics, run_prior_sensitivity=with_psense)

    def _render_comparison(self, result: ComparisonResult, names: list[str]) -> None:
        content: list[Any] = []

        # compare_models() always labels models "model_0", "model_1", ... by
        # list position (see comparison.py — there's no way to pass custom
        # names into it). `names` is that same list, in that same order, so
        # this mapping is exact — not a guess.
        label_map = {f"model_{i}": n for i, n in enumerate(names)}
        mapping_line = " · ".join(f"<b>{k}</b> = {v}" for k, v in label_map.items())

        if result.comparison_table is not None:
            table_df = result.comparison_table.reset_index().rename(columns={"index": "Model"})
            table_df["Model"] = table_df["Model"].replace(label_map)
            table = pn.widgets.Tabulator(table_df, show_index=False)
            content.append(pn.Card(
                pn.Column(
                    pn.pane.Markdown(
                        "**LOO** estimates out-of-sample predictive accuracy by "
                        "systematically leaving out one observation at a time, "
                        "then computing the log-predictive density for the "
                        "omitted point. It is computed efficiently using "
                        "Pareto-smoothed importance sampling (PSIS). An "
                        "important diagnostic in LOO is the **Pareto k "
                        "diagnostic**, which measures the reliability of the "
                        "importance sampling approximation. Values of "
                        "**k > 0.7** indicate that the approximation may be "
                        "unstable for certain observations — treat the ranking "
                        "with caution when this appears.",
                        margin=(4, 0, 8, 0),
                    ),
                    table,
                ),
                title="Ranking (LOO / ELPD)", margin=10,
            ))

            # A Pareto-k warning belongs in front of the ranking it qualifies.
            summary_text = result.summary()
            if "unreliable" in summary_text.lower() or "pareto" in summary_text.lower():
                for line in summary_text.splitlines():
                    if "pareto" in line.lower():
                        for generic, real in label_map.items():
                            line = line.replace(generic, real)
                        content.insert(0, pn.pane.HTML(_warn_box(line.strip())))
                        break

        plot_specs = (
            (result.compare_plot, "Model Ranking Plot"),
            (result.pp_check_plot, "Posterior Predictive Check (first model)"),
            (result.params_plot, "Parameter Distributions (first model)"),
        )
        any_plot = any(fig is not None for fig, _ in plot_specs)
        if any_plot:
            # The plots themselves are rendered by compare_models() with
            # "model_0"/"model_1" baked into the image as text — relabeling
            # the image would mean redrawing it ourselves, which is exactly
            # the "don't reimplement backend plotting" line we don't cross.
            # A caption mapping the generic labels to your names is the
            # closest fix that stays entirely on the GUI side.
            content.append(pn.pane.HTML(
                _info_box(f"In the plots below, {mapping_line} (by save order, not rank).")
            ))
        for fig, title in plot_specs:
            if fig is not None:
                content.append(pn.Card(_matplotlib_pane(fig), title=title, margin=10))

        if result.bayes_factor:
            bf_tables = []
            for key, name in label_map.items():
                bf_df = result.bayes_factor.get(key)
                if bf_df is not None and not bf_df.empty:
                    bf_tables.append(pn.Column(
                        pn.pane.Markdown(f"**{name}**", margin=(4, 0, 2, 0)),
                        pn.widgets.Tabulator(bf_df.reset_index().rename(columns={"index": "Term"}),
                                              show_index=False),
                    ))
            if bf_tables:
                content.append(pn.Card(
                    pn.Column(
                        pn.pane.Markdown(
                            "Savage-Dickey Bayes factor per coefficient (H0: coefficient = 0). "
                            "**BF10 > 1** favors including the term; **BF01 > 1** favors excluding "
                            "it. This is *not* a model-vs-model Bayes factor.",
                            margin=(4, 0, 8, 0),
                        ),
                        *bf_tables,
                    ),
                    title="Bayes Factor", margin=10,
                ))

        if result.prior_sensitivity:
            psense_tables = []
            for key, name in label_map.items():
                psense_df = result.prior_sensitivity.get(key)
                if psense_df is not None and not psense_df.empty:
                    psense_tables.append(pn.Column(
                        pn.pane.Markdown(f"**{name}**", margin=(4, 0, 2, 0)),
                        pn.widgets.Tabulator(
                            psense_df.reset_index().rename(columns={"index": "Parameter"}),
                            show_index=False,
                        ),
                    ))
            if psense_tables:
                content.append(pn.Card(
                    pn.Column(
                        pn.pane.Markdown(
                            "Power-scaling prior/likelihood sensitivity per parameter. Higher "
                            "**prior** or **likelihood** values mean the posterior shifts more "
                            "when that side is scaled up/down; **diagnosis** flags a parameter "
                            "whose posterior is driven mainly by the prior rather than the data "
                            "(e.g. \"potential strong prior / weak likelihood\") — worth a second "
                            "look before trusting that parameter's estimate.",
                            margin=(4, 0, 8, 0),
                        ),
                        *psense_tables,
                    ),
                    title="Prior Sensitivity", margin=10,
                ))

        for err_pane in _plot_errors_section(result.plot_errors or {}, {
            "compare_plot": "model ranking plot",
            "pp_check_plot": "posterior predictive check",
            "params_plot": "parameter distribution plot",
        }):
            content.append(err_pane)

        self._compare_result_pane.objects = content
        self._use_model_btn.disabled = False
        self._compare_status.object = _success_box(
            f"Compared {len(names)} model(s): {', '.join(names)}. "
            "Select a model below and click <b>Use Selected Model</b> to use it for estimation."
        )

    def _on_use_model_click(self, event: Any) -> None:
        import copy

        name = self._use_model_sel.value
        saved = self.state.saved_models or {}
        if not name or name not in saved:
            self._use_model_status.object = _error_box(
                "No model selected", "Choose a model from the dropdown first."
            )
            return
        # A fresh shallow copy: Update Model mutates state.model's attributes
        # in place, and must not be able to reach back into (and quietly
        # change) the frozen comparison snapshot.
        self.state.model = copy.copy(saved[name].model)
        self.state.param.trigger("model")
        self._use_model_status.object = _success_box(
            f"<b>{name}</b> is now the active model. Check Convergence Evaluation, "
            "Update Model, or SAE Estimation above/below."
        )

    # SAE Estimation

    def _build_sae_widgets(self) -> None:
        self._ci_prob_in = pn.widgets.FloatInput(
            name="Credible interval (ci_prob)", value=0.95, start=0.5, end=0.99,
            step=0.01, max_width=220,
        )
        self._sae_new_data_cb = pn.widgets.Checkbox(
            name="Estimate for new/unsampled areas (replaces the training data "
            "for this run only)",
            value=False,
        )
        self._sae_new_data_file = pn.widgets.FileInput(
            accept=".csv", name="Upload new-area CSV (predictors + area column, no response needed)",
            disabled=True,
        )
        self._sae_new_data_cb.param.watch(
            lambda e: setattr(self._sae_new_data_file, "disabled", not e.new), "value"
        )

        self._sae_run_btn = pn.widgets.Button(
            name="Run SAE Estimation", button_type="primary", max_width=280,
        )
        self._sae_status = pn.pane.HTML("")
        self._sae_notes = pn.pane.HTML("")
        self._sae_table = pn.widgets.Tabulator(
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

        new_data = None
        if self._sae_new_data_cb.value:
            if not self._sae_new_data_file.value:
                self._sae_status.object = _error_box(
                    "No file uploaded",
                    "Upload a CSV of new areas, or uncheck 'Estimate for new/unsampled areas'.",
                )
                return
            try:
                new_data = pd.read_csv(io.BytesIO(self._sae_new_data_file.value))
            except Exception as exc:
                self._sae_status.object = _error_box("Could not read CSV file", str(exc))
                return

        ci_prob = float(self._ci_prob_in.value)

        self._sae_run_btn.loading = self._sae_run_btn.disabled = True
        self._sae_status.object = _info_box("Computing SAE estimates…")
        try:
            loop = asyncio.get_running_loop()
            result: AreaEstimatesResult = await loop.run_in_executor(
                None, self._sae_blocking, model, ci_prob, new_data
            )
        except Exception as exc:
            logger.exception("ResultsTab SAE estimation failed")
            title, body = _describe_error(exc)
            self._sae_status.object = _error_box(title, body)
            return
        finally:
            self._sae_run_btn.loading = self._sae_run_btn.disabled = False

        df = result.result_table.copy()

        # estimation-frontend.md S4: mark which rows are unsampled areas
        # (only meaningful when new_data was supplied and a group column
        # exists on the model).
        group_col = (model.result.extra or {}).get("group")
        if new_data is not None and group_col and group_col in df.columns:
            sampled = set(model.data[group_col])
            df["area_type"] = df[group_col].map(
                lambda g: "sampled" if g in sampled else "non-sampled"
            )

        self._sae_table.value = df
        self._sae_download_btn.disabled = False

        # estimation-frontend.md S3: surface notes the backend only logs.
        notes: list[str] = []
        if group_col and group_col in df.columns and df[group_col].duplicated().any():
            notes.append(
                "Area labels repeat: rows are per observation, not aggregated per area."
            )
        if (n_nan := int(df["rse_pct"].isna().sum())) > 0:
            notes.append(
                f"{n_nan} area(s) have a mean of 0, so RSE is undefined there and "
                "excluded from Mean RSE."
            )
        if new_data is not None and "area_type" in df.columns and (df["area_type"] == "non-sampled").any():
            notes.append(
                "Credible intervals for non-sampled areas are wider because there is "
                "no survey data for them — their area effect is drawn from the "
                "distribution of other areas rather than estimated directly."
            )
        self._sae_notes.object = _warn_box("<br>".join(f"• {n}" for n in notes)) if notes else ""

        self._sae_status.object = _success_box(
            f"SAE estimation complete. Mean RSE: <b>{result.mean_rse:.2f}%</b>, "
            f"Mean MSE: <b>{result.mean_mse:.4f}</b>."
        )

    @staticmethod
    def _sae_blocking(
        model: Any, ci_prob: float, new_data: pd.DataFrame | None,
    ) -> AreaEstimatesResult:
        return estimate_areas(model, new_data=new_data, ci_prob=ci_prob)

    def _sae_csv_callback(self) -> io.StringIO:
        buf = io.StringIO()
        self._sae_table.value.to_csv(buf, index=False)
        buf.seek(0)
        return buf

    # Layout

    def panel(self) -> pn.Tabs:
        """Return the Panel layout for this tab.

        Returns:
            A ``panel.Tabs`` with four sub-tabs: Convergence Evaluation
            (R-hat/ESS table and diagnostic plots, shown as-is — the GUI
            does not gate on them), Model Comparison (compare one or more
            :class:`~hbsaemp.app._app.SavedModel` snapshots via
            ``compare_models()``), Update Model (an embedded
            :class:`~hbsaemp.app.tabs.update_tab.UpdateModelTab`), and
            SAE Estimation (small-area estimates, including out-of-sample
            areas, with CSV download).
        """
        convergenceevaluation_card = pn.Card(
            pn.Column(
                self._conv_run_btn,
                self._conv_status,
                pn.layout.Divider(),
                pn.Accordion(
                    (
                        "R-hat and ESS",
                        pn.Column(self._rhat_ess_desc, self._rhat_ess_table, sizing_mode="stretch_width"),
                    ),
                    active=[], sizing_mode="stretch_width", margin=(8, 0),
                ),
                self._plots_pane,
                pn.Row(self._plots_download_btn),
            ),
            title="MCMC Convergence Evaluation",
            margin=10,
        )

        modelcomparison_card = pn.Card(
            pn.Column(
                pn.pane.Markdown(
                    "Check one or more models saved from the <b>Modeling</b> tab's "
                    "<i>Save Model</i> button. A single model shows its own LOO/ELPD "
                    "and, if enabled below, Bayes Factor per coefficient; two or more "
                    "add a side-by-side ranking. Check the Convergence Evaluation "
                    "numbers yourself before relying on any of it.",
                    margin=(4, 0, 8, 0),
                ),
                self._compare_table,
                self._compare_bf_cb,
                self._compare_psense_cb,
                pn.Row(self._compare_run_btn),
                self._compare_status,
                self._compare_result_pane,
                pn.layout.Divider(),
                pn.pane.Markdown("**Select a model to use:**", margin=(4, 0, 4, 0)),
                pn.Column(self._use_model_sel, self._use_model_btn),
                self._use_model_status,
            ),
            title="Model Comparison",
            margin=10,
        )

        updatemodel_card = self._update_tab.panel()

        saeestimation_card = pn.Card(
            pn.Column(
                pn.FlexBox(self._ci_prob_in),
                self._sae_new_data_cb,
                self._sae_new_data_file,
                pn.Row(self._sae_run_btn),
                self._sae_status,
                self._sae_notes,
                pn.layout.Divider(),
                self._sae_table,
                pn.Row(self._sae_download_btn),
            ),
            title="SAE Estimation",
            margin=10,
        )

        return pn.Tabs(
            ("Convergence Evaluation", convergenceevaluation_card),
            ("Model Comparison",       modelcomparison_card),
            ("Update Model",           updatemodel_card),
            ("SAE Estimation",         saeestimation_card),
            sizing_mode="stretch_width",
        )

    def __repr__(self) -> str:
        model = getattr(self.state, "model", None)
        has_results = bool(model is not None and getattr(model, "is_fitted", False))
        n_saved = len(self.state.saved_models or {})
        return f"ResultsTab(has_results={has_results}, saved_models={n_saved})"