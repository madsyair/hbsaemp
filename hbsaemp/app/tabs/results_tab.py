"""Tab 4 — Results: diagnostics, SAE estimates, and export.

Equivalent to the "Results" tab in R hbsaems Shiny app.

v1: implemented with Panel + ArviZ plots + pandas tables.

Sub-tabs
--------
1. **Convergence Evaluation** — R-hat / ESS table, trace, autocorrelation
   and density plots (all via ``arviz`` / ``arviz_plots``), plus a
   divergent-transitions badge and an automatic issue summary.
2. **SAE Estimation** — per-observation table (Actual, Prediction, SE,
   RSE%) derived from the posterior predictive distribution, downloadable
   as CSV.

Mapping from R hbsaems
-----------------------
.. code-block:: text

    R Shiny output                          Panel v1 equivalent
    ─────────────────────────────────────── ──────────────────────────────────
    verbatimTextOutput("diag_numerical")    pn.widgets.Tabulator (R-hat/ESS)
    plotOutput("diag_plots")                pn.pane.Matplotlib (az.plot_trace, …)
    DT::dataTableOutput("sae_table")        pn.widgets.Tabulator
    downloadButton("download_estimates")    pn.widgets.FileDownload
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

import arviz as az
import arviz_plots as azp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import panel as pn
import param

if TYPE_CHECKING:
    from hbsaemp.app._app import AppState

__all__: list[str] = ["ResultsTab"]


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


class ResultsTab(param.Parameterized):
    """Results display tab with convergence diagnostics and SAE estimates.

    Args:
        state: Shared :class:`~hbsaemp.app._app.AppState` instance. Reads
            ``state.idata`` and ``state.y_vals`` (set by
            :class:`~hbsaemp.app.tabs.model_tab.ModelTab` after fitting).
    """

    state: "AppState" = param.Parameter()

    def __init__(self, state: "AppState", **params: Any) -> None:
        super().__init__(state=state, **params)

        self._conv_run_btn    = pn.widgets.Button(
            name="Load Convergence Diagnostics", button_type="primary", max_width=280,
        )
        self._conv_status     = pn.pane.HTML("")
        self._rhat_ess_table  = pn.widgets.Tabulator(
            pd.DataFrame(), show_index=False,
            pagination="remote", page_size=15,
        )
        self._trace_pane      = pn.pane.Matplotlib(sizing_mode="stretch_width", tight=True, max_width=900)
        self._divergence_badge = pn.pane.HTML("")
        self._autocorr_pane   = pn.pane.Matplotlib(sizing_mode="stretch_width", tight=True, max_width=900)
        self._density_pane    = pn.pane.Matplotlib(sizing_mode="stretch_width", tight=True, max_width=900)
        self._conv_run_btn.on_click(self._on_load_convergence)

        self._sae_run_btn     = pn.widgets.Button(
            name="Run SAE Estimation", button_type="primary", max_width=280,
        )
        self._sae_status      = pn.pane.HTML("")
        self._sae_table       = pn.widgets.Tabulator(
            pd.DataFrame(), show_index=False,
            pagination="remote", page_size=20,
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

    def _get_idata(self):
        """Return idata from state, or None if not yet available."""
        return getattr(self.state, "idata", None)

    def _get_y(self):
        return getattr(self.state, "y_vals", None)

    def _on_load_convergence(self, event: Any) -> None:
        idata = self._get_idata()
        if idata is None:
            self._conv_status.object = _error_box(
                "Model has not been fitted. Please click <b>Fit Model</b> in the <b>Modeling</b> tab first."
            )
            return

        self._conv_status.object = (
            '<div style="background:#0072B2;color:white;padding:10px 16px;'
            'border-radius:8px;margin-top:8px">Computing diagnostics</div>'
        )

        try:
            var_names = [
                v for v in idata.posterior.data_vars
                if "__obs__" not in idata.posterior[v].dims
            ]

            summary_df = az.summary(idata, var_names=var_names, ci_prob=0.94, round_to=4)

            rhat_col     = "r_hat"     if "r_hat"     in summary_df.columns else None
            ess_bulk_col = "ess_bulk"  if "ess_bulk"  in summary_df.columns else None
            ess_tail_col = "ess_tail"  if "ess_tail"  in summary_df.columns else None

            df_conv = pd.DataFrame({"Parameter": summary_df.index})
            if rhat_col:
                df_conv["R-hat"]     = summary_df[rhat_col].values.round(4)
                df_conv["Converged?"] = df_conv["R-hat"].apply(
                    lambda v: "✔" if v < 1.01 else "✘"
                )
            if ess_bulk_col:
                df_conv["ESS bulk"] = summary_df[ess_bulk_col].values.astype(int)
            if ess_tail_col:
                df_conv["ESS tail"] = summary_df[ess_tail_col].values.astype(int)

            self._rhat_ess_table.value = df_conv.reset_index(drop=True)

            az.plot_trace(idata, var_names=var_names)
            fig_trace = plt.gcf()
            fig_trace.suptitle("Trace Plots", fontsize=12, fontweight="bold")
            plt.tight_layout()
            self._trace_pane.object = fig_trace
            plt.close(fig_trace)

            pc_acf = azp.plot_autocorr(idata, var_names=var_names)
            fig_acf = pc_acf.viz["figure"].item()
            fig_acf.suptitle("Autocorrelation for Each Parameter", fontsize=12, fontweight="bold")
            fig_acf.subplots_adjust(top=0.8)
            self._autocorr_pane.object = fig_acf
            plt.close(fig_acf)

            pc_dens = azp.plot_dist(idata, group="posterior", var_names=var_names, kind="kde")
            fig_dens = pc_dens.viz["figure"].item()
            fig_dens.suptitle("Density Plot for Each Parameter", fontsize=12, fontweight="bold")
            fig_dens.subplots_adjust(top=0.75)
            self._density_pane.object = fig_dens
            plt.close(fig_dens)

            n_divergences = (
                int(idata.sample_stats["diverging"].sum())
                if "diverging" in idata.sample_stats else None
            )
            self._divergence_badge.object = (
                f'<span style="background:{"#d62728" if n_divergences else "#2ca02c"};'
                f'color:white;padding:5px 14px;border-radius:20px;font-weight:bold">'
                f'Divergent Transitions: {n_divergences if n_divergences is not None else "N/A"}'
                f'</span>'
            )

            max_rhat = float(df_conv["R-hat"].max()) if "R-hat" in df_conv.columns else float("nan")
            min_ess  = int(df_conv["ESS bulk"].min()) if "ESS bulk" in df_conv.columns else 0

            issues = []
            if not np.isnan(max_rhat) and max_rhat >= 1.01:
                issues.append(f"Max R-hat {max_rhat:.4f} ≥ 1.01 (indication of a lack of convergence).")
            if min_ess and min_ess < 400:
                issues.append(f"Min ESS bulk {min_ess} < 400 (lack of effective samples).")
            if n_divergences:
                issues.append(
                    f"{n_divergences} divergent transition detected, the posterior results may be biased."
                )

            if issues:
                self._conv_status.object = _error_box(
                    "Potential convergence issues detected",
                    "<br>".join(f"• {i}" for i in issues) +
                    "<br><br>Consider reviewing the model or data."
                )
            else:
                self._conv_status.object = _success_box(
                    f"Diagnostics completed, no convergence issues detected. "
                    f"Max R-hat: <b>{max_rhat:.4f}</b> (good if &lt; 1.01), "
                    f"Min ESS bulk: <b>{min_ess}</b> (good if &gt; 400), "
                    f"Divergent transitions: <b>{n_divergences}</b>."
                )

        except Exception as exc:
            self._conv_status.object = _error_box("Diagnostik failed", str(exc))

    def _on_run_sae_estimation(self, event: Any) -> None:
        idata = self._get_idata()
        y     = self._get_y()
        if idata is None or y is None:
            self._sae_status.object = _error_box(
                "Model has not been fitted. Please fit the model first in the <b>Modeling</b> tab."
            )
            return

        self._sae_status.object = (
            '<div style="background:#0072B2;color:white;padding:10px 16px;'
            'border-radius:8px;margin-top:8px">Computing SAE estimates</div>'
        )

        try:
            y_arr  = np.asarray(y, dtype=float)
            n      = len(y_arr)

            response_var = list(idata.posterior_predictive.data_vars)[0]
            post_y = idata.posterior_predictive[response_var].values.reshape(-1, n)

            y_pred = post_y.mean(axis=0)
            y_sd   = post_y.std(axis=0)

            with np.errstate(divide="ignore", invalid="ignore"):
                rse_obs = np.where(y_pred != 0, np.abs(y_sd / y_pred) * 100, np.nan)

            df_sae = pd.DataFrame({
                "Obs":      np.arange(1, n + 1),
                "Actual":   np.round(y_arr,   4),
                "Prediction": np.round(y_pred,  4),
                "SE":       np.round(y_sd,    4),
                "RSE (%)":  np.round(rse_obs, 4),
            })
            self._sae_table.value = df_sae
            self._sae_download_btn.disabled = False

            self._sae_status.object = _success_box(
                f"SAE Estimation completed."
            )

        except Exception as exc:
            self._sae_status.object = _error_box("SAE Estimation failed", str(exc))

    def _sae_csv_callback(self) -> io.StringIO:
        buf = io.StringIO()
        self._sae_table.value.to_csv(buf, index=False)
        buf.seek(0)
        return buf

    def panel(self) -> pn.Column:
        """Return the Panel layout for this tab."""
        convergenceevaluation_card = pn.Card(
            pn.Column(
                pn.pane.Markdown(
                    "MCMC convergence evaluation is performed to ensure that the Markov Chain Monte Carlo (MCMC) "
                    "sampling process has produced stable and representative posterior samples. The evaluation is "
                    "conducted using several diagnostic measures, namely R-hat, Effective Sample Size (ESS), "
                    "trace plots, autocorrelation plots, and density plots.",
                    margin=(4, 0, 8, 0),
                ),
                self._conv_run_btn,
                self._conv_status,
                pn.layout.Divider(),
                self._divergence_badge,
                pn.Accordion(
                    (
                        "R-hat and ESS",
                        pn.Column(
                            pn.pane.Markdown(
                                "The R-hat (Gelman-Rubin) is used to assess convergence by comparing the "
                                "between-chain variance with the within-chain variance. An R-hat value close "
                                "to 1 indicates that the chains have converged and are sampling from similar "
                                "posterior distributions. In contrast, an R-hat value greater than 1 may "
                                "indicate differences between chains and suggest that the sampling process "
                                "has not fully converged.",
                                margin=(4, 0, 8, 0),
                            ),
                            self._rhat_ess_table,
                            sizing_mode="stretch_width",
                        ),
                    ),
                    active=[], sizing_mode="stretch_width", margin=(8, 0),
                ),
                pn.Accordion(
                    (
                        "Trace Plot",
                        pn.Column(
                            pn.pane.Markdown(
                                "Trace plots display the sampled values of each parameter across iterations "
                                "for each MCMC chain. They are used to visually assess the mixing and "
                                "convergence of the chains. Well-converged chains generally exhibit a "
                                "\u201chairy caterpillar\u201d pattern, where the sampled values fluctuate "
                                "randomly around a stable region without noticeable long-term trends. In "
                                "contrast, systematic trends, separated chains, or slow movement across the "
                                "parameter space may indicate poor mixing or convergence problems.",
                                margin=(4, 0, 8, 0),
                            ),
                            self._trace_pane,
                            sizing_mode="stretch_width",
                        ),
                    ),
                    active=[], sizing_mode="stretch_width", margin=(8, 0),
                ),
                pn.Accordion(
                    (
                        "Autocorrelation Plot",
                        pn.Column(
                            pn.pane.Markdown(
                                "Autocorrelation plots display the correlation between sampled values at a "
                                "given iteration and their values at previous iterations (lags). High "
                                "autocorrelation indicates that consecutive samples are strongly dependent, "
                                "which reduces the effective sample size. Ideally, autocorrelation should "
                                "decrease rapidly as the lag increases, indicating that successive samples "
                                "become increasingly independent.",
                                margin=(4, 0, 8, 0),
                            ),
                            self._autocorr_pane,
                            sizing_mode="stretch_width",
                        ),
                    ),
                    active=[], sizing_mode="stretch_width", margin=(8, 0),
                ),
                pn.Accordion(
                    (
                        "Density Plot",
                        pn.Column(
                            pn.pane.Markdown(
                                "Density plots visualize the estimated posterior distributions of the model "
                                "parameters based on the MCMC samples. They provide information about the "
                                "shape, central tendency, and spread of each posterior distribution. A "
                                "smooth and consistent posterior distribution across chains provides "
                                "additional evidence that the sampling process has adequately represented "
                                "the posterior distribution.",
                                margin=(4, 0, 8, 0),
                            ),
                            self._density_pane,
                            sizing_mode="stretch_width",
                        ),
                    ),
                    active=[], sizing_mode="stretch_width", margin=(8, 0),
                ),
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
        has_results = self._get_idata() is not None
        return f"ResultsTab(has_results={has_results})"
