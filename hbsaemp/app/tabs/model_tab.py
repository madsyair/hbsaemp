"""Tab 3 — Model specification and fitting.

Equivalent to the "Modeling" tab in R hbsaems Shiny app.

v0: stub class.
v1: implemented with Panel widgets + hbm() backend.

Features (v1)
--------------
* **Model configuration**:
  - Response variable selector.
  - Predictor variable multi-select.
  - Group variable selector (random effects).
  - Family / distribution selector (gaussian / beta / binomial).
  - Link function selector (auto-populated based on family).
* **Sampler configuration**: draws, tune, chains, cores, target_accept,
  random_seed — all backed by :class:`~hbsaemp.models._config.ModelConfig`.
* **Prior specification**: per-term prior distribution with name + parameters.
  "Summarize Priors" button shows ``model.prior_predictive()``.
* **Prior predictive check**: plot before fitting.
* **Fit Model** button: calls ``hbm(...).fit()`` in a background thread;
  shows progress indicator.

Mapping from R hbsaems
-----------------------
.. code-block:: text

    R Shiny                             Panel v1 equivalent
    ─────────────────────────────────── ──────────────────────────────────
    selectInput("response_var", …)      pn.widgets.Select
    pickerInput("auxiliary_vars", …)    PredictorCheckboxes (pn.FlexBox of Checkbox)
    selectInput("group_var", …)         pn.widgets.Select
    selectInput("distribution_type", …) pn.widgets.Select (family)
    selectInput("hb_link", …)           pn.widgets.Select (link)
    actionButton("fit_model", …)        pn.widgets.Button
    withProgress(…)                     threading.Thread + status HTML
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

import arviz_plots as azp
import bambi as bmb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import panel as pn
import param

if TYPE_CHECKING:
    from hbsaemp.app._app import AppState

__all__: list[str] = ["ModelTab"]

_FAMILY_OPTIONS: dict[str, str] = {
    "Gaussian":  "gaussian",
    "Beta":      "beta",
    "Lognormal": "lognormal",
    "Binomial":  "binomial",
}
 
_FAMILY_LINK_DEFAULT: dict[str, str] = {
    "gaussian":  "identity",
    "beta":      "logit",
    "lognormal": "log",
    "binomial":  "logit",
}
 
_FAMILY_LINK_OPTIONS: dict[str, list[str]] = {
    "gaussian":  ["identity", "log"],
    "beta":      ["logit", "probit", "cloglog"],
    "lognormal": ["log"],
    "binomial":  ["logit", "probit", "cloglog"],
}
 
_FAMILY_DESC: dict[str, str] = {
    "gaussian": (
        "**Gaussian**: response variable is continuous and unbounded. "
        "Default link: *identity*. Suitable for y ∈ ℝ."
    ),
    "beta": (
        "**Beta**: response variable is a proportion or rate in (0, 1). "
        "Default link: *logit*. The precision parameter φᵢ is calculated "
        "from the `n` and `deff` columns."
    ),
    "lognormal": (
        "**Lognormal**: response variable is continuous and strictly positive (y > 0). "
        "Default link: *log*."
    ),
    "binomial": (
        "**Binomial**: response variable is a count of successes out of a specified number of trials"
        "Default link: *logit*. The `trials` column is **required**."
    ),
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


def _has_group(idata: Any, name: str) -> bool:
    """Check whether a group (e.g. ``'posterior_predictive'``) exists on an InferenceData."""
    return any(g.strip("/") == name for g in idata.groups)


class PredictorCheckboxes(param.Parameterized):
    """A dynamic set of checkboxes used to multi-select predictor columns."""

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
            ``state.idata``, ``state.model``, ``state.y_vals``,
            ``state.pred_names`` and ``state.response_col`` after fitting,
            for :class:`~hbsaemp.app.tabs.results_tab.ResultsTab` to consume.
    """

    state: "AppState" = param.Parameter()

    def __init__(self, state: "AppState", **params: Any) -> None:
        super().__init__(state=state, **params)

        self._response_sel = pn.widgets.Select(
            name="Response Variable  (y)",
            options=[], max_width=280,
        )
        self._predictors_sel = PredictorCheckboxes(
            name="Auxiliary / Predictor Variables  (x)",
        )
        self._group_sel = pn.widgets.Select(
            name="Group / Area Variable  (optional)",
            options=[], value=None, max_width=280,
        )

        self._family_sel = pn.widgets.Select(
            name="HB Family",
            options=_FAMILY_OPTIONS, value="gaussian", max_width=250,
        )
        self._link_sel = pn.widgets.Select(
            name="Link Function",
            options=_FAMILY_LINK_OPTIONS["gaussian"],
            value=_FAMILY_LINK_DEFAULT["gaussian"],
            max_width=250,
        )
        self._family_desc = pn.pane.Markdown(
            _FAMILY_DESC["gaussian"], margin=(4, 0, 8, 0),
        )
        self._family_sel.param.watch(self._on_family_change, "value")

        self._binomial_trials_sel = pn.widgets.Select(
            name="trials column  (number of trials)",
            options=[], value=None, max_width=280,
        )
        self._extra_params_pane = pn.Column()

        self._formula_preview = pn.pane.HTML(self._render_formula_html())
        for w in [self._response_sel, self._predictors_sel, self._group_sel,
                  self._family_sel, self._binomial_trials_sel]:
            w.param.watch(self._update_formula_preview, "value")

        self._build_btn = pn.widgets.Button(
            name="Build Model",
            button_type="primary", max_width=220,
        )
        self._build_status  = pn.pane.HTML("")
        self._build_btn.on_click(self._on_build)

        self._prior_run_btn   = pn.widgets.Button(
            name="Run Prior Predictive Check", button_type="primary", max_width=260,
        )
        self._prior_status    = pn.pane.HTML("")
        self._prior_plot_pane = pn.pane.Matplotlib(sizing_mode="stretch_width", tight=True, max_width=900)
        self._prior_ppc_pane  = pn.pane.Matplotlib(sizing_mode="stretch_width", tight=True, max_width=900)
        self._prior_run_btn.on_click(self._on_prior_check)

        self._fit_btn         = pn.widgets.Button(
            name="Fit Model", button_type="primary", max_width=220,
        )
        self._fit_status      = pn.pane.HTML("")
        self._fit_btn.on_click(self._on_fit_model)

        self._postpc_run_btn = pn.widgets.Button(
            name="Run Posterior Predictive Check", button_type="primary", max_width=280,
        )
        self._postpc_status         = pn.pane.HTML("")
        self._postpc_dist_pane      = pn.pane.Matplotlib(sizing_mode="stretch_width", tight=True, max_width=900)
        self._postpc_interval_pane  = pn.pane.Matplotlib(sizing_mode="stretch_width", tight=True, max_width=900)
        self._postpc_run_btn.on_click(self._on_posterior_check)

        self.state.param.watch(self._on_data_change, "data")
        if self.state.data is not None:
            self._populate_selectors(self.state.data)

    def _on_data_change(self, event: param.parameterized.Event) -> None:
        if event.new is not None:
            self._populate_selectors(event.new)

    def _populate_selectors(self, df: pd.DataFrame) -> None:
        all_cols = df.columns.tolist()
        num_cols = df.select_dtypes(include="number").columns.tolist()

        self._response_sel.options   = num_cols
        self._response_sel.value     = num_cols[0] if num_cols else None

        self._predictors_sel.options = num_cols
        self._predictors_sel.value   = num_cols[1:] if len(num_cols) > 1 else []

        self._group_sel.options = [None] + all_cols
        self._group_sel.value   = None

        self._binomial_trials_sel.options = [None] + num_cols
        self._binomial_trials_sel.value   = None

        self._update_formula_preview()

    def _on_family_change(self, event: param.parameterized.Event) -> None:
        family = event.new
        self._link_sel.options = _FAMILY_LINK_OPTIONS[family]
        self._link_sel.value   = _FAMILY_LINK_DEFAULT[family]
        self._family_desc.object = _FAMILY_DESC[family]
        self._refresh_extra_params(family)

    def _refresh_extra_params(self, family: str) -> None:
        if family == "binomial":
            self._extra_params_pane.objects = [
                pn.pane.Markdown(
                    "**Binomial: specific parameter**, column containing the number of trials (nᵢ) for each observation. "
                    "**Required.**",
                    margin=(4, 0, 6, 0),
                ),
                self._binomial_trials_sel,
            ]
        else:
            self._extra_params_pane.objects = []

    def _build_formula_str(self) -> str:
        response   = self._response_sel.value   or "y"
        predictors = list(self._predictors_sel.value or [])
        group      = self._group_sel.value
        family     = self._family_sel.value
        trials     = self._binomial_trials_sel.value

        lhs = f"p({response}, {trials})" if family == "binomial" and trials else response

        rhs = predictors if predictors else ["1"]
        if group:
            rhs.append(f"(1|{group})")

        return f"{lhs} ~ {' + '.join(rhs)}"

    def _render_formula_html(self) -> str:
        formula = self._build_formula_str()
        return (
            f'<div style="background:#f4f4f4;border-left:4px solid #0072B2;'
            f'padding:10px 16px;border-radius:6px;font-family:monospace;font-size:1.05em">'
            f'<b>Formula:</b>&nbsp; {formula}</div>'
        )

    def _update_formula_preview(self, *_: Any) -> None:
        self._formula_preview.object = self._render_formula_html()

    def _validate_build(self) -> list[str]:
        errors: list[str] = []

        if not self._response_sel.value:
            errors.append("Response variable not yet selected.")
        if not self._predictors_sel.value:
            errors.append("At least one predictor variable and one response variable must be selected.")

        family = self._family_sel.value
        if family == "binomial" and not self._binomial_trials_sel.value:
            errors.append(
                "Family <b>binomial</b> requires a <code>trials</code> column. "
                "Please select the appropriate column."
            )
        return errors

    def _on_build(self, event: Any) -> None:
        errors = self._validate_build()
        if errors:
            self._build_status.object = _error_box(
                "Can not build model", "<br>".join(f"• {e}" for e in errors)
            )
            return

        try:
            _model, _pred_names, had_missing = self._build_bambi_model()
        except Exception as exc:
            self._build_status.object = _error_box("Build model failed", str(exc))
            return

        msg = (
            "Model built successfully. Moving on to the tab <b>Prior Predictive Checking</b>."
        )
        if had_missing:
            msg += (
                "<br><br><i>Missing values were detected and the corresponding "
                "rows were automatically removed before modeling.</i>"
            )
        self._build_status.object = _success_box(msg)

    def _build_bambi_model(self) -> tuple[bmb.Model, list[str], bool]:
        response   = self._response_sel.value
        predictors = list(self._predictors_sel.value or [])
        group      = self._group_sel.value
        family     = self._family_sel.value
        trials_col = self._binomial_trials_sel.value if family == "binomial" else None

        cols = [response, *predictors]
        if group:
            cols.append(group)
        if trials_col:
            cols.append(trials_col)
        cols = list(dict.fromkeys(cols))
        df_model = self.state.data[cols].copy()

        had_missing = bool(df_model.isna().any().any())
        if had_missing:
            df_model = df_model.dropna(subset=cols).reset_index(drop=True)

        if family == "binomial":
            df_model[response]   = df_model[response].astype(int)
            df_model[trials_col] = df_model[trials_col].astype(int)

        formula = self._build_formula_str()

        model = bmb.Model(
            formula, df_model,
            family=family,
            link=self._link_sel.value,
        )
        pred_names = predictors if predictors else ["Intercept"]
        return model, pred_names, had_missing

    @staticmethod
    def _response_array(model: bmb.Model) -> np.ndarray:
        arr = np.asarray(model.response_component.term.data)
        if arr.ndim > 1:       # binomial: kolom [successes, trials]
            arr = arr[:, 0]
        return arr.astype(float).flatten()

    def _on_prior_check(self, event: Any) -> None:
        if self.state.data is None:
            self._prior_status.object = _error_box(
                "Data not available. Please upload the data first in the <b>Data Upload</b> tab."
            )
            return
        if not self._response_sel.value:
            self._prior_status.object = _error_box(
                "Please select a response variable in <b>Model Building</b> and click Build Model."
            )
            return

        self._prior_status.object = (
            '<div style="background:#0072B2;color:white;padding:10px 16px;'
            'border-radius:8px;margin-top:8px">'
            'Running prior predictive check</div>'
        )

        def _run():
            try:
                model, pred_names, _had_missing = self._build_bambi_model()
                model.build()
                prior_idata = model.prior_predictive()

                # Histogram prior for each parameter
                param_names = list(prior_idata.prior.data_vars)
                n_params = len(param_names)
                fig1, axes1 = plt.subplots(1, n_params, figsize=(5 * n_params, 4))
                if n_params == 1:
                    axes1 = [axes1]
                for ax, pname in zip(axes1, param_names):
                    vals = prior_idata.prior[pname].values.flatten()
                    ax.hist(vals, bins=40, color="#0072B2", edgecolor="white", alpha=0.8)
                    ax.set_title(f"Prior: {pname}", fontweight="bold")
                    ax.set_xlabel("Value"); ax.set_ylabel("Frequency")
                fig1.suptitle("Prior Distribution of Each Parameter",
                              fontsize=13, fontweight="bold", y=1.02)
                plt.tight_layout()
                self._prior_plot_pane.object = fig1
                plt.close(fig1)

                # Prior predictive vs actual data
                pc = azp.plot_ppc_dist(
                    prior_idata, group="prior_predictive",
                    visuals={"observed_dist": True},
                )
                fig2 = pc.viz["figure"].item()
                fig2.suptitle("Prior Predictive vs Actual Data",
                               fontsize=11, fontweight="bold")
                fig2.subplots_adjust(top=0.82)
                self._prior_ppc_pane.object = fig2
                plt.close(fig2)

                self._prior_status.object = _success_box(
                    "Prior predictive check complete. Moving on the tab <b>Fit Model</b>."
                )

            except Exception as exc:
                self._prior_status.object = _error_box(
                    "Prior check failed.", str(exc)
                )

        threading.Thread(target=_run, daemon=True).start()

    def _on_fit_model(self, event: Any) -> None:
        errors = self._validate_build()
        if errors:
            self._fit_status.object = _error_box(
                "Unable to fit the model",
                "<br>".join(f"• {e}" for e in errors)
            )
            return

        self._fit_status.object = (
            '<div style="background:#0072B2;color:white;padding:10px 16px;'
            'border-radius:8px;margin-top:8px">'
            'Running MCMC sampling</div>'
        )

        def _run():
            try:
                model, pred_names, had_missing = self._build_bambi_model()

                # Tidak ada draws/tune/chains yang di-hardcode — sepenuhnya
                # memakai default sampling dari Bambi/PyMC (NUTS).
                idata = model.fit()
                model.predict(idata, kind="response", inplace=True)

                self.state.idata         = idata
                self.state.model         = model
                self.state.y_vals        = self._response_array(model)
                self.state.pred_names    = pred_names
                self.state.response_col  = self._response_sel.value

                msg = (
                    "The MCMC process has been completed."
                )
                if had_missing:
                    msg += (
                        "<br><br>⚠ <i>Missing values were detected and the "
                        "corresponding rows were automatically removed before "
                        "modeling.</i>"
                    )
                self._fit_status.object = _success_box(msg)

            except Exception as exc:
                self._fit_status.object = _error_box(
                    "Fit model failed", str(exc)
                )

        threading.Thread(target=_run, daemon=True).start()

    def _on_posterior_check(self, event: Any) -> None:
        idata = getattr(self.state, "idata", None)
        model = getattr(self.state, "model", None)
        y     = getattr(self.state, "y_vals", None)

        if idata is None or model is None or y is None:
            self._postpc_status.object = _error_box(
                "Model has not been fitted",
                "Please click <b>Fit Model</b> before running the Posterior Predictive Check."
            )
            return

        self._postpc_status.object = (
            '<div style="background:#0072B2;color:white;padding:10px 16px;'
            'border-radius:8px;margin-top:8px">'
            'Running posterior predictive check</div>'
        )

        def _run():
            try:
                if not _has_group(idata, "posterior_predictive"):
                    model.predict(idata, kind="response", inplace=True)

                pc1 = azp.plot_ppc_dist(idata)
                fig1 = pc1.viz["figure"].item()
                fig1.suptitle("Posterior Predictive vs Actual Data",
                               fontsize=11, fontweight="bold")
                fig1.subplots_adjust(top=0.82)
                self._postpc_dist_pane.object = fig1
                plt.close(fig1)

                pc2 = azp.plot_ppc_interval(idata)
                fig2 = pc2.viz["figure"].item()
                fig2.suptitle("Credible Interval of Posterior Predictive per Observation",
                               fontsize=11, fontweight="bold")
                fig2.subplots_adjust(top=0.82)
                self._postpc_interval_pane.object = fig2
                plt.close(fig2)

                self._postpc_status.object = _success_box(
                    "Posterior predictive check complete."
                )

            except Exception as exc:
                self._postpc_status.object = _error_box(
                    "Posterior predictive check failed", str(exc)
                )

        threading.Thread(target=_run, daemon=True).start()

    def panel(self) -> pn.Column:
        """Return the Panel layout for this tab."""
        overview = pn.pane.Markdown("""
                                    **This section allows you to specify the variables and model settings used for hierarchical Bayesian modeling.**
                                    - **Response Variable:** The outcome variable being modeled.
                                    - **Auxiliary Variables:** Explanatory (independent) variables — fixed effects.
                                    - **Group Variables:** Grouping variable (e.g., area, cluster) for random effects.
                                    - **HB Family and Link Function:** Bayesian hierarchical family and corresponding link function (e.g., log, logit).""")

        createmodel_card = pn.Card(
            pn.Column(
                pn.pane.Markdown("**Select Variables**", margin=(4, 0, 4, 0)),
                pn.FlexBox(self._response_sel, self._group_sel),
                pn.pane.Markdown("*Auxiliary / Predictor Variables (x):*", margin=(4, 0, 2, 0)),
                self._predictors_sel.panel(),
                pn.layout.Divider(),
                pn.pane.Markdown(
                    "**Distribution Family and Link Function**",
                    margin=(4, 0, 4, 0),
                ),
                pn.FlexBox(self._family_sel, self._link_sel),
                self._family_desc,
                self._extra_params_pane,
                pn.layout.Divider(),
                pn.pane.Markdown("**Formula Preview**", margin=(4, 0, 4, 0)),
                self._formula_preview,
                pn.pane.Markdown(
                    "*Rows with missing values ​​in the columns used will be automatically removed before modeling.*",
                    margin=(4, 0, 4, 0),
                ),
                pn.layout.Divider(),
                pn.Row(self._build_btn),
                self._build_status,
            ),
            title="Model Building",
            margin=10,
        )

        prior_card = pn.Card(
            pn.Column(
                pn.pane.Markdown(
                    "A prior predictive check assesses the plausibility of the prior *before* fitting the model. ",
                    margin=(4, 0, 8, 0),
                ),
                pn.Row(self._prior_run_btn),
                self._prior_status,
                pn.layout.Divider(),
                pn.pane.Markdown("#### Prior Distribution of Each Parameter"),
                self._prior_plot_pane,
                pn.layout.Divider(),
                pn.pane.Markdown("#### Prior Predictive vs Actual Data"),
                self._prior_ppc_pane,
            ),
            title="Prior Predictive Check",
            margin=10,
        )

        fitmodel_card = pn.Card(
            pn.Column(
                pn.pane.Markdown(
                    "Make sure **Build Model** has been run first. Click <b>Fit Model</b> to run MCMC sampling.",
                    margin=(4, 0, 8, 0),
                ),
                pn.Row(self._fit_btn),
                self._fit_status,
            ),
            title="Fit Model",
            margin=10,
        )

        postpc_card = pn.Card(
            pn.Column(
                pn.pane.Markdown(
                    "The Posterior Predictive Check compares data replicated by the model (after fitting) with the observed data. "
                    "**Available after the model has been successfully fitted.**.",
                    margin=(4, 0, 8, 0),
                ),
                pn.Row(self._postpc_run_btn),
                self._postpc_status,
                pn.layout.Divider(),
                pn.pane.Markdown("#### Posterior Predictive vs Actual Data"),
                self._postpc_dist_pane,
                pn.layout.Divider(),
                pn.pane.Markdown("#### Credible Interval for each Observation"),
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

    def get_fitted_model(self) -> bmb.Model | None:
        """Return the fitted Bambi model, or ``None`` if not yet fitted."""
        return getattr(self.state, "model", None)

    def __repr__(self) -> str:
        fitted = getattr(self.state, "model", None) is not None
        return f"ModelTab(fitted={fitted})"

