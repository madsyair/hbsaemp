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
    pickerInput("auxiliary_vars", …)    pn.widgets.MultiChoice
    selectInput("group_var", …)         pn.widgets.Select
    selectInput("distribution_type", …) pn.widgets.Select (family)
    selectInput("hb_link", …)           pn.widgets.Select (link)
    numericInput("num_chains", …)       pn.widgets.IntInput
    sliderInput("adapt_delta", …)       pn.widgets.FloatSlider
    actionButton("fit_model", …)        pn.widgets.Button
    withProgress(…)                     pn.indicators.LoadingSpinner
"""

from __future__ import annotations

import threading
 
import bambi as bmb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import panel as pn
import param
 
import arviz_plots as azp
 
from hbsaemp.app._helpers import (
    POSTERIOR_STATUS_LABEL,
    PRIOR_STATUS_LABEL,
    error_box,
    evaluate_predictive_spread,
    success_box,
)
from hbsaemp.app._state import AppState

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
 
 
class ModelTab(param.Parameterized):
    """Model specification and fitting tab.
 
    Args:
        state: Shared :class:`~hbsaemp.app._state.AppState`.  Reads
            ``state.data``; writes ``state.model``/``state.idata`` and
            related fields after a successful fit.
    """
 
    state: AppState = param.Parameter()
 
    def __init__(self, state: AppState, **params) -> None:
        super().__init__(state=state, **params)
 
        self._response_sel = pn.widgets.Select(
            name="Response Variable  (y)",
            options=[], width=240,
        )
        self._predictors_sel = pn.widgets.CheckBoxGroup(
            name="Auxiliary / Predictor Variables  (x)",
            options=[], value=[],
        )
        self._group_sel = pn.widgets.Select(
            name="Group / Area Variable  (optional)",
            options=[], value=None, width=240,
        )
 
        self._family_sel = pn.widgets.Select(
            name="HB Family",
            options=_FAMILY_OPTIONS, value="gaussian", width=210,
        )
        self._link_sel = pn.widgets.Select(
            name="Link Function",
            options=_FAMILY_LINK_OPTIONS["gaussian"],
            value=_FAMILY_LINK_DEFAULT["gaussian"],
            width=210,
        )
        self._family_desc = pn.pane.Markdown(
            _FAMILY_DESC["gaussian"], margin=(4, 0, 8, 0),
        )
        self._family_sel.param.watch(self._on_family_change, "value")
 
        self._binomial_trials_sel = pn.widgets.Select(
            name="trials column  (number of trials)",
            options=[], value=None, width=240,
        )
        self._extra_params_pane = pn.Column()
 
        self._formula_preview = pn.pane.HTML(self._render_formula_html())
        for w in [self._response_sel, self._predictors_sel, self._group_sel,
                  self._family_sel, self._binomial_trials_sel]:
            w.param.watch(self._update_formula_preview, "value")
 
        self._build_btn = pn.widgets.Button(
            name="Build Model",
            button_type="primary", width=200,
        )
        self._build_status = pn.pane.HTML("")
        self._build_btn.on_click(self._on_build)
 
        self._prior_run_btn        = pn.widgets.Button(
            name="Run Prior Predictive Check", button_type="primary", width=240,
        )
        self._prior_status         = pn.pane.HTML("")
        self._prior_plot_pane      = pn.pane.Matplotlib(sizing_mode="stretch_width", tight=True)
        self._prior_ppc_pane       = pn.pane.Matplotlib(sizing_mode="stretch_width", tight=True)
        self._prior_interpretation = pn.pane.HTML("")
        self._prior_run_btn.on_click(self._on_prior_check)
 
        self._fit_btn     = pn.widgets.Button(
            name="Fit Model", button_type="primary", width=200,
        )
        self._fit_status  = pn.pane.HTML("")
        self._fit_run_btn = self._fit_btn
        self._fit_btn.on_click(self._on_fit_model)
 
        self._postpc_run_btn        = pn.widgets.Button(
            name="Run Posterior Predictive Check", button_type="primary", width=260,
        )
        self._postpc_status         = pn.pane.HTML("")
        self._postpc_dist_pane      = pn.pane.Matplotlib(sizing_mode="stretch_width", tight=True)
        self._postpc_interval_pane  = pn.pane.Matplotlib(sizing_mode="stretch_width", tight=True)
        self._postpc_interpretation = pn.pane.HTML("")
        self._postpc_run_btn.on_click(self._on_posterior_check)
 
        self.state.param.watch(self._on_data_change, "data")
        if self.state.data is not None:
            self._populate_selectors(self.state.data)
 
    def _on_data_change(self, event) -> None:
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
 
        # Binomial trials column options include all numeric cols
        self._binomial_trials_sel.options = [None] + num_cols
        self._binomial_trials_sel.value   = None
 
        self._update_formula_preview()
 
    def _on_family_change(self, event) -> None:
        family = event.new
        self._link_sel.options   = _FAMILY_LINK_OPTIONS[family]
        self._link_sel.value     = _FAMILY_LINK_DEFAULT[family]
        self._family_desc.object = _FAMILY_DESC[family]
        self._refresh_extra_params(family)
 
    def _refresh_extra_params(self, family: str) -> None:
        if family == "binomial":
            self._extra_params_pane.objects = [
                pn.pane.Markdown(
                    "**Binomial-specific parameter** — kolom jumlah trial (nᵢ) per observasi. "
                    "**Wajib diisi.**",
                    margin=(4, 0, 6, 0),
                ),
                self._binomial_trials_sel,
            ]
        else:
            self._extra_params_pane.objects = []
 
    def _build_formula_str(self) -> str:
        response   = self._response_sel.value or "y"
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
 
    def _update_formula_preview(self, *_) -> None:
        self._formula_preview.object = self._render_formula_html()
 
    def _validate_build(self) -> list[str]:
        errors: list[str] = []
 
        if self.state.data is None:
            errors.append(
                "Data belum diunggah. Buka tab <b>Data Upload</b> terlebih dahulu."
            )
            return errors
 
        if not self._response_sel.value:
            errors.append("Response variable belum dipilih.")
        if not self._predictors_sel.value:
            errors.append("Minimal satu predictor variable harus dipilih.")
 
        family = self._family_sel.value
        if family == "binomial" and not self._binomial_trials_sel.value:
            errors.append(
                "Family <b>binomial</b> memerlukan kolom <code>trials</code>. "
                "Pilih kolom yang sesuai."
            )
        return errors
 
    def _on_build(self, event) -> None:
        errors = self._validate_build()
        if errors:
            self._build_status.object = error_box(
                "⚠ Cannot build model", "<br>".join(f"• {e}" for e in errors)
            )
            return
 
        try:
            _model, _pred_names, had_missing = self._build_bambi_model()
        except Exception as exc:
            self._build_status.object = error_box("⚠ Build model gagal", str(exc))
            return
 
        msg = (
            "Model built successfully. Moving on to the tab <b>Prior Predictive Checking</b>."
        )
        if had_missing:
            msg += (
                "<br><br><i>Missing values were detected and the corresponding "
                "rows were automatically removed before modeling.</i>"
            )
        self._build_status.object = success_box(msg)
 
    def _build_bambi_model(self) -> tuple[bmb.Model, list[str], bool]:
        """Build a ``bambi.Model`` from the current UI formula/config.
 
        Every family (gaussian/lognormal/beta/binomial) is a Bambi built-in
        family, so no manual priors/likelihood are written per family here.
 
        Rows with a missing value in any column used by the model are always
        dropped automatically (no user-facing option). The extra
        ``had_missing`` return flags whether any rows were removed, so the
        caller can surface a notice to the user.
        """
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
        """Return the response array actually used by the model (post-dropna)."""
        arr = np.asarray(model.response_component.term.data)
        if arr.ndim > 1:  # binomial: columns [successes, trials]
            arr = arr[:, 0]
        return arr.astype(float).flatten()
 
    def _on_prior_check(self, event) -> None:
        if self.state.data is None:
            self._prior_status.object = error_box(
                "Data not available. Please upload the data first in the <b>Data Upload</b> tab."
            )
            return
        if not self._response_sel.value:
            self._prior_status.object = error_box(
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
                y = self._response_array(model)

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
 
                response_var = list(prior_idata.prior_predictive.data_vars)[0]
                pred_samples = prior_idata.prior_predictive[response_var].values
                evaluation = evaluate_predictive_spread(y, pred_samples)
                label = PRIOR_STATUS_LABEL[evaluation["status"]]
                if evaluation["status"] == "ok":
                    self._prior_interpretation.object = success_box(
                        f"{label}<br>{evaluation['detail']}"
                    )
                else:
                    self._prior_interpretation.object = error_box(label, evaluation["detail"])
 
                n_draws_used = prior_idata.prior_predictive.sizes.get("draw", "N/A")
                self._prior_status.object = success_box(
                    f"Prior predictive check complete. Moving on the tab <b>Fit Model</b>."
                )
 
            except Exception as exc:
                self._prior_status.object = error_box(
                    "Prior check failed.", str(exc)
                )
 
        threading.Thread(target=_run, daemon=True).start()
 
    def _on_fit_model(self, event) -> None:
        errors = self._validate_build()
        if errors:
            self._fit_status.object = error_box(
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
 
                idata = model.fit()
                model.predict(idata, kind="response", inplace=True)
 
                self.state.idata        = idata
                self.state.model        = model
                self.state.y_vals       = self._response_array(model)
                self.state.pred_names   = pred_names
                self.state.response_col = self._response_sel.value
 
                msg = (
                    "The MCMC process has been completed."
                )
                if had_missing:
                    msg += (
                        "<br><br>⚠ <i>Missing values were detected and the "
                        "corresponding rows were automatically removed before "
                        "modeling.</i>"
                    )
                self._fit_status.object = success_box(msg)
 
            except Exception as exc:
                self._fit_status.object = error_box(
                    "Fit model failed", str(exc)
                )
 
        threading.Thread(target=_run, daemon=True).start()
 
    def _on_posterior_check(self, event) -> None:
        idata = self.state.idata
        model = self.state.model
        y     = self.state.y_vals
 
        if idata is None or model is None or y is None:
            self._postpc_status.object = error_box(
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
                from hbsaemp.app._helpers import has_group
 
                if not has_group(idata, "posterior_predictive"):
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
                fig2.suptitle("Credible Interval Posterior Predictive for each Observasi",
                              fontsize=11, fontweight="bold")
                fig2.subplots_adjust(top=0.82)
                self._postpc_interval_pane.object = fig2
                plt.close(fig2)
 
                response_var = list(idata.posterior_predictive.data_vars)[0]
                pred_samples = idata.posterior_predictive[response_var].values
                evaluation = evaluate_predictive_spread(
                    np.asarray(y, dtype=float), pred_samples
                )
                label = POSTERIOR_STATUS_LABEL[evaluation["status"]]
                if evaluation["status"] == "ok":
                    self._postpc_interpretation.object = success_box(
                        f"{label}<br>{evaluation['detail']}"
                    )
                else:
                    self._postpc_interpretation.object = error_box(label, evaluation["detail"])
 
                self._postpc_status.object = success_box(
                    "Posterior predictive check complete."
                )
 
            except Exception as exc:
                self._postpc_status.object = error_box(
                    "Posterior predictive check failed", str(exc)
                )
 
        threading.Thread(target=_run, daemon=True).start()
 
    def panel(self) -> pn.Tabs:
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
                pn.Row(self._response_sel, self._group_sel),
                pn.pane.Markdown("*Auxiliary / Predictor Variables (x):*", margin=(4, 0, 2, 0)),
                self._predictors_sel,
                pn.layout.Divider(),
                pn.pane.Markdown(
                    "**Distribution Family and Link Function**",
                    margin=(4, 0, 4, 0),
                ),
                pn.Row(self._family_sel, self._link_sel),
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
                    "A prior predictive check assesses the plausibility of the prior *before* fitting the model.",
                    margin=(4, 0, 8, 0),
                ),
                pn.Row(self._prior_run_btn),
                self._prior_status,
                pn.layout.Divider(),
                pn.pane.Markdown("#### Distribusi Prior Setiap Parameter"),
                self._prior_plot_pane,
                pn.layout.Divider(),
                pn.pane.Markdown("#### Prior Predictive vs Data Aktual"),
                self._prior_ppc_pane,
                self._prior_interpretation,
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
                pn.pane.Markdown("#### Credible Interval for each Observasi"),
                self._postpc_interval_pane,
                self._postpc_interpretation,
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
        """Return the fitted ``bambi.Model``, or ``None`` if not fitted yet."""
        return self.state.model

    def __repr__(self) -> str:
        return "ModelTab(model_fitted={self.state.model is not None})"
