"""Tab 3 — Model specification and fitting.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import io
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import arviz_plots as azp
import matplotlib.pyplot as plt
import pandas as pd
import panel as pn
import param

from hbsaemp import (
    BaseModel,
    ConvergenceWarning,
    DataValidationError,
    EstimationError,
    FormulaError,
    HBSAEError,
    ModelConfig,
    ModelNotFittedError,
    ModelRegistryError,
    PriorSpecError,
    check_convergence,
    get_family_spec,
    hbm_beta,
    hbm_binomial,
    hbm_gaussian,
    list_families,
)
from hbsaemp._logging import get_logger
from hbsaemp.diagnostics.prior_check import PriorCheckResult, check_prior

if TYPE_CHECKING:
    from hbsaemp.app._app import AppState, SavedModel

logger = get_logger(__name__)

__all__: list[str] = ["ModelTab"]

# Tier-3 dispatch, built from the registry itself (never a hardcoded list) so
# a new family registered in FAMILY_SPECS is picked up with zero code changes
# here, and `set(_HBM_DISPATCH) == set(list_families())` holds by construction.
_HBM_DISPATCH: dict[str, Callable[..., BaseModel]] = {
    "gaussian": hbm_gaussian,
    "beta": hbm_beta,
    "binomial": hbm_binomial,
}
assert set(_HBM_DISPATCH) == set(list_families()), (
    "_HBM_DISPATCH is out of sync with the family registry — "
    "add a hbm_<family> shortcut and wire it in here."
)

# Frontend-owned prose (FamilySpec carries no description field). Only
# families in the registry are described; "Default link: …" sentences are
# omitted since the link widget already shows the resolved default.
_FAMILY_DESC: dict[str, str] = {
    "gaussian": (
        "**Gaussian**: response variable is continuous and unbounded, "
        "suitable for y in the reals. Pass a sampling-variance column to "
        "enable the Fay-Herriot offset."
    ),
    "beta": (
        "**Beta**: response variable is a proportion or rate. The "
        "precision parameter phi_i can be computed from the `n` and `deff` "
        "columns, or left to Bambi's auto-priors if omitted."
    ),
    "binomial": (
        "**Binomial**: response variable is a count of successes out of a "
        "number of trials. The `trials` column is **required**."
    ),
}

# User-facing labels for family-specific widgets, keyed by the *user kwarg*
# name (FamilySpec.user_params values) — never by family, so a family reusing
# an existing parameter name needs no new label entry.
_PARAM_LABEL: dict[str, str] = {
    "trials":       "Trials column (n_i)",
    "n":            "Sample size column (n)",
    "deff":         "Design effect column (deff)",
    "sampling_var": "Sampling variance column (D_i)",
    "squeeze":      "Smithson-Verkuilen squeeze",
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


def _info_box(body: str) -> str:
    return (
        f'<div style="background:#0072B2;color:white;padding:10px 16px;'
        f'border-radius:8px;margin-top:8px">{body}</div>'
    )


def _warn_box(body: str) -> str:
    return (
        f'<div style="background:#f0ad4e;color:#3a2e00;padding:10px 16px;'
        f'border-radius:8px;margin-top:8px">{body}</div>'
    )


def _formula_box(formula: str) -> str:
    return (
        f'<div style="background:#f4f4f4;border-left:4px solid #0072B2;'
        f'padding:10px 16px;border-radius:6px;font-family:monospace;'
        f'font-size:1.05em">'
        f'<b>Formula:</b>&nbsp; {formula}</div>'
    )


def _pending_box(message: str) -> str:
    return (
        f'<div style="background:#fff3cd;border-left:4px solid #d62728;'
        f'color:#7a1f1f;padding:10px 16px;border-radius:6px;'
        f'font-family:monospace;font-size:0.95em">'
        f'<b>Cannot build model yet:</b> {message}</div>'
    )


def _describe_error(exc: Exception) -> tuple[str, str]:
    """Map a backend exception to a (title, body) pair for the UI.

    Every ``hbsaemp`` exception derives from ``HBSAEError`` (backend never
    swallows or reclassifies), so this is the single dispatch point for
    error -> message across the whole tab.
    """
    if isinstance(exc, FormulaError):
        return "Invalid formula", "Please check the selected variables. " + str(exc)
    if isinstance(exc, ModelRegistryError):
        return "Family not available", str(exc)
    if isinstance(exc, DataValidationError):
        return "Data does not meet the family's requirements", str(exc)
    if isinstance(exc, PriorSpecError):
        return "Invalid prior specification", str(exc)
    if isinstance(exc, ModelNotFittedError):
        return "Model has not been fitted", "Please fit the model first."
    if isinstance(exc, EstimationError):
        return "Estimation failed", str(exc)
    if isinstance(exc, ImportError):
        return "Missing dependency", str(exc)
    if isinstance(exc, HBSAEError):
        return "Model configuration error", str(exc)
    if isinstance(exc, (ValueError, TypeError)):
        return "Invalid configuration", str(exc)
    return "Unexpected error", str(exc)


class PredictorCheckboxes(param.Parameterized):
    value   = param.List(default=[])
    options = param.List(default=[])

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._checkboxes: dict[str, pn.widgets.Checkbox] = {}
        self._syncing = False
        self._box = pn.FlexBox(sizing_mode="stretch_width", gap="4px 18px")
        self._rebuild_checkboxes()

    def _rebuild_checkboxes(self) -> None:
        checkboxes = {}
        for col in self.options:
            cb = pn.widgets.Checkbox(name=str(col), value=col in self.value)
            cb.param.watch(self._on_toggle, "value")
            checkboxes[col] = cb
        self._checkboxes = checkboxes
        self._box.objects = list(self._checkboxes.values())

    def _on_toggle(self, event: param.parameterized.Event) -> None:
        if self._syncing:
            return
        self.value = [col for col, cb in self._checkboxes.items() if cb.value]

    @param.depends("options", watch=True)
    def _on_options_changed(self) -> None:
        self._rebuild_checkboxes()

    @param.depends("value", watch=True)
    def _on_value_changed(self) -> None:
        self._syncing = True
        try:
            for col, cb in self._checkboxes.items():
                desired = col in self.value
                if cb.value != desired:
                    cb.value = desired
        finally:
            self._syncing = False

    def panel(self) -> pn.FlexBox:
        return self._box


class ModelTab(param.Parameterized):
    """Model specification and fitting tab.

    Args:
        state: Shared :class:`~hbsaemp.app._app.AppState` instance. Reads
            ``state.data`` (set by
            :class:`~hbsaemp.app.tabs.data_tab.DataTab`); writes
            ``state.model`` — a fitted
            :class:`~hbsaemp.models._base.BaseModel` — after
            :meth:`~hbsaemp.models._base.BaseModel.fit` succeeds, for
            :class:`~hbsaemp.app.tabs.results_tab.ResultsTab` to consume.
    """

    state: AppState = param.Parameter()

    def __init__(self, state: AppState, **params: Any) -> None:
        super().__init__(state=state, **params)

        # The single model draft threaded through Preview -> Build ->
        # Prior Check -> Fit. `None` whenever the current widget
        # selection cannot build a model; the exception message *is* the
        # validation message (no separate `_validate_build()`).
        self._model_draft: BaseModel | None = None
        self._extra_widgets: dict[str, pn.widgets.Widget] = {}

        # --- Variable selection ------------------------------------------------
        self._response_sel = pn.widgets.Select(
            name="Response Variable  (y)", options=[], max_width=280,
        )
        self._predictors_sel = PredictorCheckboxes(
            name="Auxiliary / Predictor Variables  (x)",
        )
        self._group_sel = pn.widgets.Select(
            name="Area / Group Variable  (optional)",
            options=[], value=None, max_width=280,
        )
        self._intercept_cb = pn.widgets.Checkbox(name="Include intercept", value=True)

        # --- Family / link (registry-driven) --------------------------
        families = list_families()
        default_family = "gaussian" if "gaussian" in families else families[0]
        self._family_sel = pn.widgets.Select(
            name="HB Family", options=families, value=default_family, max_width=250,
        )
        self._link_sel = pn.widgets.Select(name="Link Function", max_width=250)
        self._family_desc = pn.pane.Markdown("", margin=(4, 0, 8, 0))
        self._extra_params_pane = pn.Column()
        self._apply_family_spec(default_family)
        self._refresh_extra_params(default_family)
        self._family_sel.param.watch(self._on_family_change, "value")

        # --- Sampler configuration -------------------------------------
        self._draws_in = pn.widgets.IntInput(name="draws", value=1000, start=1, max_width=140)
        self._tune_in  = pn.widgets.IntInput(name="tune",  value=1000, start=0, max_width=140)
        self._chains_in = pn.widgets.IntInput(name="chains", value=4, start=1, max_width=140)
        self._cores_in  = pn.widgets.IntInput(name="cores", value=1, start=1, max_width=140)
        self._target_accept_in = pn.widgets.FloatInput(
            name="target_accept", value=0.8, start=0.01, end=0.99, step=0.01, max_width=160,
        )
        self._seed_cb = pn.widgets.Checkbox(name="Fix random seed", value=False)
        self._seed_in = pn.widgets.IntInput(name="random_seed", value=42, max_width=140, disabled=True)
        self._seed_cb.param.watch(
            lambda e: setattr(self._seed_in, "disabled", not e.new), "value"
        )

        # --- Formula preview ------------------------------------------------
        self._formula_preview = pn.pane.HTML(_pending_box("select a response and at least one predictor."))
        for w in [self._response_sel, self._predictors_sel, self._group_sel,
                  self._family_sel, self._link_sel, self._intercept_cb,
                  self._draws_in, self._tune_in, self._chains_in, self._cores_in,
                  self._target_accept_in, self._seed_cb, self._seed_in]:
            w.param.watch(self._update_preview, "value")

        # --- Code export (save/preview equivalent hbsaemp CLI code) --------
        self._code_view = pn.widgets.CodeEditor(
            value="", language="python", theme="monokai", readonly=True,
            height=320, sizing_mode="stretch_width",
        )
        self._code_download_btn = pn.widgets.FileDownload(
            callback=self._get_code_bytes,
            filename="hbsaemp_model.py",
            label="Save Code (.py)",
            button_type="success",
            disabled=True,
            max_width=220,
        )

        # --- Build Model -----------------------------------------------------
        self._build_btn = pn.widgets.Button(name="Build Model", button_type="primary", max_width=220)
        self._build_status = pn.pane.HTML("")
        self._build_btn.on_click(self._on_build)

        # --- Prior Predictive Check -------------------------------------------
        self._prior_run_btn = pn.widgets.Button(
            name="Run Prior Predictive Check", button_type="primary", max_width=260,
        )
        self._prior_n_draws = pn.widgets.IntInput(name="n_draws", value=50, start=1, max_width=140)
        self._prior_status = pn.pane.HTML("")
        self._prior_summary_table = pn.widgets.Tabulator(pd.DataFrame(), show_index=False)
        self._prior_plot_pane = pn.pane.Matplotlib(sizing_mode="stretch_width", tight=True, max_width=900)
        self._prior_run_btn.on_click(self._on_prior_check)

        # --- Fit Model ---------------------------------------------------------
        self._fit_btn = pn.widgets.Button(name="Fit Model", button_type="primary", max_width=220)
        self._fit_status = pn.pane.HTML("")
        self._fit_btn.on_click(self._on_fit_model)

        # --- Save Model (for the Results tab's Model Comparison) -----------------
        self._save_name_in = pn.widgets.TextInput(
            name="Model name", placeholder="e.g. Model 1", disabled=True, max_width=220,
        )
        self._save_btn = pn.widgets.Button(
            name="Save Model", button_type="success", max_width=160, disabled=True,
        )
        self._save_status = pn.pane.HTML("")
        self._save_btn.on_click(self._on_save_model)

        # --- Posterior Predictive Check ----------------------------------------
        self._postpc_run_btn = pn.widgets.Button(
            name="Run Posterior Predictive Check", button_type="primary", max_width=280,
        )
        self._postpc_status = pn.pane.HTML("")
        self._postpc_dist_pane = pn.pane.Matplotlib(sizing_mode="stretch_width", tight=True, max_width=900)
        self._postpc_interval_pane = pn.pane.Matplotlib(sizing_mode="stretch_width", tight=True, max_width=900)
        self._postpc_run_btn.on_click(self._on_posterior_check)

        self.state.param.watch(self._on_data_change, "data")
        if self.state.data is not None:
            self._populate_selectors(self.state.data)

    # Variable population

    def _on_data_change(self, event: param.parameterized.Event) -> None:
        if event.new is not None:
            self._populate_selectors(event.new)

    def _populate_selectors(self, df: pd.DataFrame) -> None:
        all_cols = df.columns.tolist()
        num_cols = df.select_dtypes(include="number").columns.tolist()

        self._response_sel.options = num_cols
        self._response_sel.value   = num_cols[0] if num_cols else None

        self._predictors_sel.options = num_cols
        self._predictors_sel.value   = num_cols[1:] if len(num_cols) > 1 else []

        self._group_sel.options = [None, *all_cols]
        self._group_sel.value   = None

        self._refresh_extra_params(self._family_sel.value)
        self._update_preview()

    def _apply_family_spec(self, family: str) -> None:
        spec = get_family_spec(family)
        self._link_sel.options = sorted(spec.supported_links)
        self._link_sel.value   = spec.default_link
        self._family_desc.object = _FAMILY_DESC.get(family, "")

    def _on_family_change(self, event: param.parameterized.Event) -> None:
        self._apply_family_spec(event.new)
        self._refresh_extra_params(event.new)
        self._update_preview()

    def _family_param_names(self, family: str) -> list[str]:
        return [uk for uk in get_family_spec(family).user_params.values() if uk != "link"]

    def _param_traits(self, family: str, user_kw: str) -> tuple[type, bool]:
        fn = _HBM_DISPATCH[family]
        p = inspect.signature(fn).parameters[user_kw]
        try:
            ann = inspect.get_annotations(fn, eval_str=True).get(user_kw, p.annotation)
        except Exception:  
            ann = p.annotation
        return ann, p.default is inspect.Parameter.empty

    def _refresh_extra_params(self, family: str) -> None:
        """(Re)generate widgets for `family`'s parameters from the registry.

        Keyed by *parameter name* (not family), so a family reusing an
        existing parameter needs zero new code here.
        """
        data = self.state.data
        numeric_cols = (
            [None, *data.select_dtypes(include="number").columns.tolist()]
            if data is not None else [None]
        )

        widgets: dict[str, pn.widgets.Widget] = {}
        objects: list[Any] = []
        for user_kw in self._family_param_names(family):
            ann, required = self._param_traits(family, user_kw)
            label = _PARAM_LABEL.get(user_kw, user_kw)
            if required:
                label += "  (required)"
            if ann is bool:
                w: pn.widgets.Widget = pn.widgets.Checkbox(name=label, value=False)
            else:
                w = pn.widgets.Select(name=label, options=numeric_cols, value=None)
            w.param.watch(self._update_preview, "value")
            widgets[user_kw] = w
            objects.append(w)

        self._extra_widgets = widgets
        self._extra_params_pane.objects = objects

    def _collect_family_kwargs(self) -> dict[str, Any]:
        return {name: w.value for name, w in self._extra_widgets.items()}

    def _collect_config(self) -> ModelConfig:
        return ModelConfig(
            draws=int(self._draws_in.value),
            tune=int(self._tune_in.value),
            chains=int(self._chains_in.value),
            cores=int(self._cores_in.value),
            target_accept=float(self._target_accept_in.value),
            random_seed=int(self._seed_in.value) if self._seed_cb.value else None,
            progressbar=False,
        )

    def _collect_build_kwargs(self) -> dict[str, Any]:
        """Every keyword `_build_model()` passes to `_HBM_DISPATCH[family]`,
        except `data` (kept separate since it's a DataFrame, not something
        `to_code()` can render as a literal).

        This is the **single source of truth** shared by `_build_model()`
        and `to_code()`: change what gets built here and both the live GUI model and the exported code
        snippet update together — there is no second place to keep in sync.
        """
        return dict(
            response=self._response_sel.value,
            auxiliary=list(self._predictors_sel.value or []),
            area_var=self._group_sel.value or None,
            intercept=self._intercept_cb.value,
            link=self._link_sel.value,
            handle_missing="deleted",
            config=self._collect_config(),
            **self._collect_family_kwargs(),
        )

    def _build_model(self) -> BaseModel:
        """Assemble an unfitted `BaseModel` from the current widget state.
        """
        family = self._family_sel.value
        return _HBM_DISPATCH[family](data=self.state.data, **self._collect_build_kwargs())

    def to_code(self, *, include_fit: bool = True) -> str:
        """Render the current widget selection as a standalone Python script.

        Built from :meth:`_collect_build_kwargs` — the exact kwargs
        `_build_model()` uses — so the snippet always reproduces what
        **Build Model** / **Fit Model** actually do. This is intentionally
        *not* a hand-maintained second copy of the assembly logic: if it
        were, the two could silently drift apart (GUI does one thing, the
        saved code does another) without any test catching it.

        Args:
            include_fit: If ``True`` (default), also emit `model.fit()` and
                an `estimate_areas()` call mirroring the Results tab. Set
                ``False`` to get just the model-construction lines (used by
                fast unit tests that don't want to run real MCMC).
        """
        family = self._family_sel.value
        fn_name = _HBM_DISPATCH[family].__name__  
        kwargs = self._collect_build_kwargs()
        config = kwargs.pop("config")

        imports = ["import pandas as pd", f"from hbsaemp import {fn_name}, ModelConfig"]
        if include_fit:
            imports[-1] += ", estimate_areas"

        lines = [
            *imports,
            "",
            "# Replace this with however you load your own data.",
            'data = pd.read_csv("your_data.csv")',
            "",
            f"config = ModelConfig(draws={config.draws}, tune={config.tune}, "
            f"chains={config.chains}, cores={config.cores}, "
            f"target_accept={config.target_accept}, "
            f"random_seed={config.random_seed!r})",
            "",
            f"model = {fn_name}(",
            "    data=data,",
            *(f"    {key}={value!r}," for key, value in kwargs.items()),
            "    config=config,",
            ")",
        ]
        if include_fit:
            lines += [
                "model.fit()",
                "",
                "result = estimate_areas(model, ci_prob=0.95)",
                "print(result.result_table)",
            ]
        return "\n".join(lines)

    def _update_preview(self, *_: Any) -> None:
        """Preview = `model.formula`"""
        if self.state.data is None:
            self._model_draft = None
            self._formula_preview.object = _pending_box(
                "upload or load a dataset in the Data Upload tab first."
            )
            self._update_code_view()
            return
        if not self._response_sel.value or not self._predictors_sel.value:
            self._model_draft = None
            self._formula_preview.object = _pending_box(
                "select a response and at least one predictor."
            )
            self._update_code_view()
            return
        try:
            model = self._build_model()
        except Exception as exc:  
            self._model_draft = None
            _, body = _describe_error(exc)
            self._formula_preview.object = _pending_box(body)
            self._update_code_view()
            return
        self._model_draft = model
        self._formula_preview.object = _formula_box(model.formula)
        self._update_code_view()

    def _update_code_view(self) -> None:
        """Keep the code editor / download button in lockstep with the
        formula preview: no valid `_model_draft` -> nothing safe to export.
        """
        if self._model_draft is None:
            self._code_view.value = ""
            self._code_download_btn.disabled = True
            return
        self._code_view.value = self.to_code()
        self._code_download_btn.disabled = False

    def _get_code_bytes(self) -> io.BytesIO:
        return io.BytesIO(self.to_code().encode("utf-8"))

    def _on_build(self, event: Any) -> None:
        self._update_preview()
        if self._model_draft is None:
            self._build_status.object = _error_box(
                "Cannot build model", "Fix the formula preview above first."
            )
            return
        try:
            clean = self._model_draft.check_data()
        except HBSAEError as exc:
            logger.exception("ModelTab build/check_data failed")
            title, body = _describe_error(exc)
            self._build_status.object = _error_box(title, body)
            return

        n_dropped = len(self.state.data) - len(clean)
        msg = "Model built and data validated successfully. Moving on to Prior Predictive Check."
        if n_dropped:
            msg += f"<br><br>{n_dropped} row(s) with missing values were dropped before modeling."
        self._build_status.object = _success_box(msg)

    # Prior Predictive Check — check_prior() only, no raw Bambi/hand-rolled plots

    async def _on_prior_check(self, event: Any) -> None:
        if self._prior_run_btn.loading:
            return
        if self._model_draft is None:
            self._prior_status.object = _error_box(
                "Cannot run prior predictive check", "Fix the formula preview above first."
            )
            return

        model = self._model_draft
        n_draws = int(self._prior_n_draws.value)

        self._prior_run_btn.loading = self._prior_run_btn.disabled = True
        self._prior_status.object = _info_box("Running prior predictive check…")
        try:
            loop = asyncio.get_running_loop()
            result: PriorCheckResult = await loop.run_in_executor(
                None, self._prior_blocking, model, n_draws
            )
        except Exception as exc:
            logger.exception("ModelTab prior check failed")
            title, body = _describe_error(exc)
            self._prior_status.object = _error_box(title, body)
            return
        finally:
            self._prior_run_btn.loading = self._prior_run_btn.disabled = False

        if result.prior_summary is not None and len(result.prior_summary):
            self._prior_summary_table.value = result.prior_summary.reset_index().rename(
                columns={"index": "Parameter"}
            )
        if result.prior_predictive_plot is not None:
            self._prior_plot_pane.object = result.prior_predictive_plot
            plt.close(result.prior_predictive_plot)

        self._prior_status.object = _success_box(
            "Prior predictive check complete. Moving on to <b>Fit Model</b>."
        )

    @staticmethod
    def _prior_blocking(model: BaseModel, n_draws: int) -> PriorCheckResult:
        """Synchronous, Panel-free — runs in the executor thread."""
        return check_prior(model, n_draws=n_draws)

    async def _on_fit_model(self, event: Any) -> None:
        if self._fit_btn.loading:  
            return
        if self._model_draft is None:
            self._fit_status.object = _error_box(
                "Cannot fit model", "Fix the formula preview above first."
            )
            return

        model = self._model_draft
        self._fit_btn.loading = self._fit_btn.disabled = True
        self._fit_status.object = _info_box("Running MCMC sampling…")
        try:
            loop = asyncio.get_running_loop()
            fitted = await loop.run_in_executor(None, self._fit_blocking, model)
        except Exception as exc:
            logger.exception("ModelTab fit failed")
            title, body = _describe_error(exc)
            self._fit_status.object = _error_box(title, body)
            return
        finally:
            self._fit_btn.loading = self._fit_btn.disabled = False

        self.state.model = fitted
        self._fit_status.object = _success_box(
            "The MCMC sampling has completed. See the <b>Results</b> tab for "
            "diagnostics and SAE estimates."
        )
        self._save_btn.disabled = self._save_name_in.disabled = False
        if not self._save_name_in.value:
            self._save_name_in.value = f"Model {len(self.state.saved_models) + 1}"

    @staticmethod
    def _fit_blocking(model: BaseModel) -> BaseModel:
        model.fit()
        return model

    async def _on_save_model(self, event: Any) -> None:
        """Freeze the active model as a named snapshot for Model Comparison.

        Runs `check_convergence()` at save time (the same function the
        Results tab uses) so the saved snapshot carries its own
        convergence status — Model Comparison can then refuse to compare
        anything that hasn't converged, without re-running diagnostics
        itself. Stores a `copy.copy()` of the model, not the live
        `state.model` reference: `update_model()` replaces (rather than
        mutates) the attributes it touches, so a shallow copy is enough to
        stop a later refit of the *active* model from silently changing
        an already-saved entry.
        """
        if self._save_btn.loading:  
            return
        model = getattr(self.state, "model", None)
        if model is None or not model.is_fitted:
            self._save_status.object = _error_box(
                "No fitted model", "Fit a model first."
            )
            return
        name = (self._save_name_in.value or "").strip()
        if not name:
            self._save_status.object = _error_box(
                "Name required", "Give this model a name before saving."
            )
            return

        self._save_btn.loading = self._save_btn.disabled = True
        self._save_status.object = _info_box("Checking convergence before saving…")
        try:
            loop = asyncio.get_running_loop()
            converged, warning_messages = await loop.run_in_executor(
                None, self._convergence_check_blocking, model
            )
        except Exception as exc:
            logger.exception("ModelTab save-model convergence check failed")
            title, body = _describe_error(exc)
            self._save_status.object = _error_box(title, body)
            return
        finally:
            self._save_btn.loading = self._save_btn.disabled = False

        from hbsaemp.app._app import SavedModel  # local: avoids a circular import

        snapshot = SavedModel(
            model=copy.copy(model), converged=converged, warnings=warning_messages,
        )
        # Reassign (not in-place mutation) so param.watch on "saved_models" fires.
        self.state.saved_models = {**self.state.saved_models, name: snapshot}

        status = _success_box if converged else _warn_box
        note = "" if converged else (
            "<br>" + "<br>".join(f"• {m}" for m in warning_messages)
            + "<br><br><b>This model can still be saved, but Model Comparison "
            "will not let you select it for comparison until it converges.</b>"
        )
        self._save_status.object = status(
            f"Saved as <b>{name}</b>. Open <b>Results → Model Comparison</b> to "
            f"compare it against other saved models.{note}"
        )
        self._save_name_in.value = f"Model {len(self.state.saved_models) + 1}"

    @staticmethod
    def _convergence_check_blocking(model: BaseModel) -> tuple[bool, list[str]]:
        """Synchronous, Panel-free — runs in the executor thread."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            check_convergence(model)
        messages = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
        return not messages, messages

    async def _on_posterior_check(self, event: Any) -> None:
        if self._postpc_run_btn.loading:
            return
        model = self.state.model
        if model is None or not model.is_fitted:
            self._postpc_status.object = _error_box(
                "Model has not been fitted", "Please click <b>Fit Model</b> first."
            )
            return

        self._postpc_run_btn.loading = self._postpc_run_btn.disabled = True
        self._postpc_status.object = _info_box("Running posterior predictive check…")
        try:
            loop = asyncio.get_running_loop()
            idata = await loop.run_in_executor(None, model.predictive_idata)
        except Exception as exc:
            logger.exception("ModelTab posterior predictive check failed")
            title, body = _describe_error(exc)
            self._postpc_status.object = _error_box(title, body)
            return
        finally:
            self._postpc_run_btn.loading = self._postpc_run_btn.disabled = False

        try:
            pc1 = azp.plot_ppc_dist(idata)
            fig1 = pc1.viz["figure"].item()
            fig1.suptitle("Posterior Predictive Plot", fontsize=11, fontweight="bold")
            fig1.subplots_adjust(top=0.82)
            self._postpc_dist_pane.object = fig1
            plt.close(fig1)

            pc2 = azp.plot_ppc_interval(idata)
            fig2 = pc2.viz["figure"].item()
            fig2.suptitle("Rootgram",
                          fontsize=11, fontweight="bold")
            fig2.subplots_adjust(top=0.82)
            self._postpc_interval_pane.object = fig2
            plt.close(fig2)
        except Exception as exc:  
            logger.exception("ModelTab posterior predictive plotting failed")
            self._postpc_status.object = _error_box("Could not render plots", str(exc))
            return

        self._postpc_status.object = _success_box("Posterior predictive check complete.")

    def panel(self) -> pn.Tabs:
        overview = pn.pane.Markdown("""
                                    This section allows you to specify the variables and model settings used for hierarchical Bayesian modeling.
                                    - **Response Variable:** The outcome variable being modeled.
                                    - **Auxiliary Variables:** Explanatory (independent) variables.
                                    - **Area / Group Variable:** Grouping variable for the random intercept `(1|area)`.
                                    - **HB Family and Link Function:** Loaded from the `hbsaemp` family registry.""")

        createmodel_card = pn.Card(
            pn.Column(
                pn.pane.Markdown("**Select Variables**", margin=(4, 0, 4, 0)),
                pn.FlexBox(self._response_sel, self._group_sel, self._intercept_cb),
                pn.pane.Markdown("*Auxiliary / Predictor Variables (x):*", margin=(4, 0, 2, 0)),
                self._predictors_sel.panel(),
                pn.layout.Divider(),
                pn.pane.Markdown("**Distribution Family and Link Function**", margin=(4, 0, 4, 0)),
                pn.FlexBox(self._family_sel, self._link_sel),
                self._family_desc,
                self._extra_params_pane,
                pn.layout.Divider(),
                pn.pane.Markdown("**Sampler Configuration**", margin=(4, 0, 4, 0)),
                pn.FlexBox(
                    self._draws_in, self._tune_in, self._chains_in, self._cores_in,
                    self._target_accept_in, self._seed_cb, self._seed_in,
                ),
                pn.layout.Divider(),
                pn.pane.Markdown("**Formula Preview**", margin=(4, 0, 4, 0)),
                self._formula_preview,
                pn.layout.Divider(),
                pn.Row(self._build_btn),
                self._build_status,
                pn.pane.Markdown(
                    "**Note:** Save this code to reproduce the model using the "
                    "`hbsaemp` Python API without the GUI.",
                    margin=(8, 0, 4, 0),
                ),
                self._code_view,
                pn.Row(self._code_download_btn),
            ),
            title="Model Building",
            margin=10,
        )

        prior_card = pn.Card(
            pn.Column(
                pn.pane.Markdown(
                    "A prior predictive check assesses the plausibility of the prior before fitting the model",
                    margin=(4, 0, 8, 0),
                ),
                pn.Row(self._prior_run_btn, self._prior_n_draws),
                self._prior_status,
                pn.layout.Divider(),
                pn.pane.Markdown("#### Prior Summary"),
                self._prior_summary_table,
                pn.layout.Divider(),
                pn.pane.Markdown("#### Prior Predictive ECDF Plot"),
                self._prior_plot_pane,
            ),
            title="Prior Predictive Check",
            margin=10,
        )

        fitmodel_card = pn.Card(
            pn.Column(
                pn.pane.Markdown(
                    "Click <b>Fit Model</b> to run MCMC sampling on the model shown in "
                    "the formula preview.",
                    margin=(4, 0, 8, 0),
                ),
                pn.Row(self._fit_btn),
                self._fit_status,
                pn.layout.Divider(),
                pn.pane.Markdown(
                    "**Save this fit** to compare it against other configurations later, "
                    "in <b>Results → Model Comparison</b>.",
                    margin=(4, 0, 4, 0),
                ),
                pn.Row(self._save_name_in, self._save_btn),
                self._save_status,
            ),
            title="Fit Model",
            margin=10,
        )

        postpc_card = pn.Card(
            pn.Column(
                pn.pane.Markdown(
                    "The Posterior Predictive Check compares data replicated by the "
                    "fitted model with the observed data. "
                    "**Available after the model has been fitted.**",
                    margin=(4, 0, 8, 0),
                ),
                pn.Row(self._postpc_run_btn),
                self._postpc_status,
                pn.layout.Divider(),
                pn.pane.Markdown("#### Posterior Predictive Plot"),
                self._postpc_dist_pane,
                pn.layout.Divider(),
                pn.pane.Markdown("#### Posterior Predictive Credible Intervals by Observation"),
                self._postpc_interval_pane,
            ),
            title="Posterior Predictive Check",
            margin=10,
        )

        return pn.Tabs(
            ("Overview", pn.Card(overview, title="Overview", margin=10)),
            ("Model Building", createmodel_card),
            ("Prior Predictive Check", prior_card),
            ("Fit Model", fitmodel_card),
            ("Posterior Predictive Check", postpc_card),
            sizing_mode="stretch_width",
        )

    def get_fitted_model(self) -> BaseModel | None:
        model = getattr(self.state, "model", None)
        return model if (model is not None and model.is_fitted) else None

    def __repr__(self) -> str:
        fitted = getattr(self.state, "model", None) is not None
        return f"ModelTab(fitted={fitted})"
