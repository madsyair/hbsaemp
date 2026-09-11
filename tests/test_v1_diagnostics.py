"""check_convergence() and compare_models() v1 tests — require Bambi MCMC.

Run with:
    pytest -m slow tests/test_v1_diagnostics.py -v

Skip with:
    pytest -m "not slow"
"""
from __future__ import annotations

import warnings

import pandas as pd
import pytest

import hbsaemp as hb

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Shared fitted model fixtures (module-scoped)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def beta_model(data_beta: pd.DataFrame) -> hb.BaseModel:
    cfg = hb.ModelConfig(draws=200, tune=200, chains=2, cores=1,
                         target_accept=0.9, random_seed=42)
    m = hb.create_model(
        "y ~ x1 + x2 + (1|group)",
        family="beta", data=data_beta, n="n", deff="deff", config=cfg,
    )
    m.fit()
    return m


@pytest.fixture(scope="module")
def gaussian_model(data_gaussian: pd.DataFrame) -> hb.BaseModel:
    cfg = hb.ModelConfig(draws=200, tune=200, chains=2, cores=1,
                         target_accept=0.9, random_seed=7)
    m = hb.create_model(
        "y ~ x1 + x2 + (1|group)",
        family="gaussian", data=data_gaussian, config=cfg,
    )
    m.fit()
    return m


@pytest.fixture(scope="module")
def beta_model_small(data_beta: pd.DataFrame) -> hb.BaseModel:
    """Nested competitor of ``beta_model`` on the same data (drops ``x2``)."""
    cfg = hb.ModelConfig(draws=200, tune=200, chains=2, cores=1,
                         target_accept=0.9, random_seed=42)
    m = hb.create_model(
        "y ~ x1 + (1|group)",
        family="beta", data=data_beta, n="n", deff="deff", config=cfg,
    )
    m.fit()
    return m


# ===========================================================================
# check_convergence()
# ===========================================================================

class TestCheckConvergence:

    def test_returns_convergence_result(self, beta_model):
        result = hb.check_convergence(beta_model)
        assert isinstance(result, hb.ConvergenceResult)

    def test_rhat_ess_is_dataframe(self, beta_model):
        result = hb.check_convergence(beta_model, plot_types=[])
        assert isinstance(result.rhat_ess, pd.DataFrame)

    def test_rhat_ess_not_empty(self, beta_model):
        result = hb.check_convergence(beta_model, plot_types=[])
        assert not result.rhat_ess.empty

    def test_rhat_ess_has_r_hat_column(self, beta_model):
        result = hb.check_convergence(beta_model, plot_types=[])
        assert "r_hat" in result.rhat_ess.columns

    def test_rhat_ess_has_ess_columns(self, beta_model):
        result = hb.check_convergence(beta_model, plot_types=[])
        assert "ess_bulk" in result.rhat_ess.columns

    def test_rhat_values_near_one(self, beta_model):
        """Well-behaved model: Rhat should be close to 1.

        Threshold 1.1: Vehtari et al. (2021) recommend < 1.01 for production
        analyses; we allow 1.1 here because the fixture uses short MCMC
        (200 draws × 2 chains) which inflates Rhat for stable models.
        """
        result = hb.check_convergence(beta_model, plot_types=[])
        max_rhat = result.rhat_ess["r_hat"].max()
        assert max_rhat < 1.1, f"Rhat too high: {max_rhat:.4f} (model may have divergences)"

    def test_ess_positive(self, beta_model):
        result = hb.check_convergence(beta_model, plot_types=[])
        assert (result.rhat_ess["ess_bulk"] > 0).all()

    def test_plots_dict_returned(self, beta_model):
        result = hb.check_convergence(beta_model, plot_types=["trace", "dens"])
        assert isinstance(result.plots, dict)

    def test_plots_keys_match_requested(self, beta_model):
        result = hb.check_convergence(beta_model, plot_types=["trace", "dens"])
        # Some plots might fail on CI — just check keys are a subset
        assert set(result.plots.keys()).issubset({"trace", "dens"})

    def test_empty_plot_types(self, beta_model):
        """Passing plot_types=[] should skip all plots."""
        result = hb.check_convergence(beta_model, plot_types=[])
        assert result.plots == {}

    def test_not_fitted_raises(self, data_beta):
        cfg = hb.ModelConfig(draws=100, tune=100, chains=2)
        m = hb.create_model("y ~ x1", family="beta",
                             data=data_beta, n="n", deff="deff", config=cfg)
        with pytest.raises(hb.ModelNotFittedError):
            hb.check_convergence(m)

    def test_r_alias_hbcc(self, beta_model):
        result = hb.hbcc(beta_model, plot_types=[])
        assert isinstance(result, hb.ConvergenceResult)

    def test_summary_contains_rhat(self, beta_model):
        result = hb.check_convergence(beta_model, plot_types=[])
        s = result.summary()
        assert "Rhat" in s or "rhat" in s.lower()

    def test_repr(self, beta_model):
        result = hb.check_convergence(beta_model, plot_types=[])
        r = repr(result)
        assert "ConvergenceResult" in r

    def test_convergence_warning_on_bad_rhat(self, data_gaussian):
        """Mock az.summary to return a bad Rhat — must trigger ConvergenceWarning.

        Patches ``arviz.summary`` directly: ``check_convergence`` imports arviz
        lazily inside the function, so ``mock.patch("hbsaemp.diagnostics.
        convergence.az")`` cannot work — ``az`` is not a module-level attribute.
        """
        import unittest.mock as mock

        cfg = hb.ModelConfig(draws=200, tune=200, chains=2, cores=1, random_seed=0)
        m = hb.create_model("y ~ x1", family="gaussian",
                             data=data_gaussian, config=cfg)
        m.fit()

        # Rhat 1.5 is unambiguously above any reasonable warning threshold
        # (Vehtari et al. 2021: < 1.01 ideal; > 1.1 unacceptable). Using 1.5
        # avoids fragility if the implementation tightens its threshold.
        bad_summary = pd.DataFrame(
            {"mean": [0.0], "sd": [0.1], "r_hat": [1.5],
             "ess_bulk": [500.0], "ess_tail": [500.0]},
            index=["b_x1"],
        )
        with mock.patch("arviz.summary", return_value=bad_summary):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                hb.check_convergence(m, plot_types=[])
                cw = [w for w in caught if issubclass(w.category, hb.ConvergenceWarning)]
                assert len(cw) >= 1, "Expected ConvergenceWarning for Rhat=1.5"

    # Per-observation params (mu/kappa) must not pollute the summary,
    # group-level effects (1|group) must be retained, and trace plots must not
    # crash on the subplot cap.

    def test_convergence_summary_excludes_per_obs_params(self, beta_model):
        conv = hb.check_convergence(beta_model, plot_types=[])
        idx = conv.rhat_ess.index.astype(str)
        assert not idx.str.startswith("mu[").any()
        assert not idx.str.startswith("kappa[").any()
        assert not idx.str.startswith("p[").any()

    def test_convergence_summary_retains_group_effects(self, beta_model):
        conv = hb.check_convergence(beta_model, plot_types=[])
        idx = conv.rhat_ess.index.astype(str)
        assert idx.str.startswith("1|group[").any()

    def test_convergence_trace_plot_no_crash(self, beta_model):
        import matplotlib.figure

        conv = hb.check_convergence(beta_model, plot_types=["trace"])
        assert "trace" in conv.plots
        assert "Requested" not in conv.plot_errors.get("trace", "")
        # The Figure must actually be extracted from the ArviZ 1.1 PlotCollection
        # return value — a None here (stored silently) was the follow-on bug.
        assert conv.plots["trace"] is not None
        assert isinstance(conv.plots["trace"], matplotlib.figure.Figure)

    def test_convergence_energy_plot_no_crash(self, beta_model):
        """plot_energy reads sample_stats and rejects var_names — the loop must
        exempt it (``_NO_VAR_NAMES_PLOTS``) and still render the energy/BFMI plot."""
        import matplotlib.figure

        conv = hb.check_convergence(beta_model, plot_types=["energy"])
        # The var_names exemption must hold — a TypeError about var_names here
        # would mean the blanket injection wasn't skipped for energy.
        assert "var_names" not in conv.plot_errors.get("energy", "")
        assert "energy" in conv.plots
        assert isinstance(conv.plots["energy"], matplotlib.figure.Figure)

    def test_ess_threshold_scales_with_chains(self, beta_model):
        """ESS floor follows arviz.diagnose: 100 per chain (fixture uses 2)."""
        conv = hb.check_convergence(beta_model, plot_types=[])
        assert conv.ess_threshold == 200

    def test_sampler_checks_populated(self, beta_model):
        """NUTS records diverging / reached_max_treedepth / energy, so all three
        sampler checks must be reported."""
        conv = hb.check_convergence(beta_model, plot_types=[])
        assert conv.diagnose is not None
        assert {"divergent", "treedepth", "bfmi"} <= conv.diagnose.keys()

    def test_rhat_ess_is_numeric(self, beta_model):
        """Unrounded floats — auto-rounded strings hid R-hat breaches."""
        conv = hb.check_convergence(beta_model, plot_types=[])
        for col in ("mean", "sd", "r_hat", "ess_bulk", "ess_tail"):
            assert pd.api.types.is_float_dtype(conv.rhat_ess[col]), col


# ===========================================================================
# compare_models() — single model
# ===========================================================================

class TestCompareModelsSingle:

    def test_returns_comparison_result(self, beta_model):
        result = hb.compare_models(beta_model)
        assert isinstance(result, hb.ComparisonResult)

    def test_loo_not_none(self, beta_model):
        result = hb.compare_models(beta_model)
        assert result.loo is not None

    def test_no_comparison_table_for_single(self, beta_model):
        result = hb.compare_models(beta_model)
        assert result.comparison_table is None

    def test_compare_plot_none_for_single(self, beta_model):
        """compare_plot is multi-model only (needs the az.compare table)."""
        result = hb.compare_models(beta_model)
        assert result.compare_plot is None

    def test_pp_check_plot_is_figure(self, beta_model):
        """pp_check_plot must be a matplotlib Figure (or None if plotting fails)."""
        import matplotlib.figure
        result = hb.compare_models(beta_model)
        if result.pp_check_plot is not None:
            assert isinstance(result.pp_check_plot, matplotlib.figure.Figure)

    def test_params_plot_is_figure(self, beta_model):
        import matplotlib.figure
        result = hb.compare_models(beta_model)
        if result.params_plot is not None:
            assert isinstance(result.params_plot, matplotlib.figure.Figure)

    def test_comparison_uses_public_api(self, beta_model):
        """compare_models must not crash and must leave result.idata pristine.

        The PPC block sources its idata from the public predictive_idata()
        (never the private _predict_idata) and checks groups via _idata_groups
        (never .groups(), which crashes on ArviZ 1.1 DataTree)."""
        hb.compare_models([beta_model])
        # stored posterior untouched — parameter vars still present after comparison
        assert "Intercept" in beta_model.result.idata.posterior

    def test_compare_models_generates_ppc_plot(self, beta_model):
        """pp_check_plot must be a Figure now that arviz_plots.plot_ppc_dist is
        wired (az.plot_ppc was removed in ArviZ 1.1, leaving it None before)."""
        import matplotlib.figure as mfig
        result = hb.compare_models([beta_model])
        assert result.pp_check_plot is not None
        assert isinstance(result.pp_check_plot, mfig.Figure)

    def test_compare_models_ppc_does_not_mutate_idata(self, beta_model):
        """PPC plotting sources Y_rep via predictive_idata() (inplace=False),
        so result.idata's posterior must be untouched (no vars appended)."""
        vars0 = set(beta_model.result.idata.posterior.data_vars)
        hb.compare_models([beta_model])
        assert set(beta_model.result.idata.posterior.data_vars) == vars0

    def test_not_fitted_raises(self, data_beta):
        cfg = hb.ModelConfig(draws=100, tune=100, chains=2)
        m = hb.create_model("y ~ x1", family="beta",
                             data=data_beta, n="n", deff="deff", config=cfg)
        with pytest.raises(hb.ModelNotFittedError):
            hb.compare_models(m)

    def test_r_alias_hbmc(self, beta_model):
        result = hb.hbmc(beta_model)
        assert isinstance(result, hb.ComparisonResult)

    def test_summary_contains_loo(self, beta_model):
        result = hb.compare_models(beta_model)
        s = result.summary()
        assert "LOO" in s or "loo" in s.lower()

    def test_summary_loo_shows_numeric_elpd(self, beta_model):
        """summary() must print the real elpd, not the '?' placeholder (A-1)."""
        import arviz as az
        result = hb.compare_models(beta_model)
        s = result.summary()
        loo_line = next(ln for ln in s.splitlines() if "LOO" in ln)
        assert "?" not in loo_line
        expected = f"{float(az.loo(beta_model.result.idata).elpd):.2f}"
        assert expected in loo_line

    def test_repr(self, beta_model):
        result = hb.compare_models(beta_model)
        r = repr(result)
        assert "ComparisonResult" in r

    def test_summary_reports_pareto_k(self, beta_model):
        s = hb.compare_models(beta_model).summary()
        assert "Pareto k" in s

    def test_plot_errors_empty_on_success(self, beta_model):
        result = hb.compare_models(beta_model)
        assert result.plot_errors == {}


# ===========================================================================
# compare_models() — Bayes factors and prior sensitivity
# ===========================================================================

class TestCompareModelsBayesFactor:

    def test_bf_table_per_coefficient(self, beta_model):
        result = hb.compare_models(beta_model, metrics=["loo", "bf"])
        bf = result.bayes_factor
        assert isinstance(bf, pd.DataFrame)
        assert set(bf.index) == {"x1", "x2"}  # intercept excluded
        assert list(bf.columns) == ["BF10", "BF01"]
        assert (bf["BF10"] > 0).all()
        assert ((bf["BF10"] * bf["BF01"]) - 1).abs().max() < 1e-9

    def test_bf_only_skips_loo(self, beta_model):
        result = hb.compare_models(beta_model, metrics="bf")
        assert result.loo is None
        assert result.bayes_factor is not None
        assert "BF10" in result.summary()

    def test_bf_list_input_is_dict(self, beta_model):
        result = hb.compare_models([beta_model], metrics=["bf"])
        assert isinstance(result.bayes_factor, dict)
        assert "model_0" in result.bayes_factor

    def test_bf_does_not_mutate_idata(self, beta_model):
        groups0 = tuple(beta_model.result.idata.groups)
        hb.compare_models(beta_model, metrics=["bf"])
        assert tuple(beta_model.result.idata.groups) == groups0


class TestCompareModelsPriorSensitivity:

    def test_psense_table(self, beta_model):
        result = hb.compare_models(beta_model, run_prior_sensitivity=True)
        ps = result.prior_sensitivity
        assert isinstance(ps, pd.DataFrame)
        assert {"prior", "likelihood", "diagnosis"} <= set(ps.columns)
        assert pd.api.types.is_float_dtype(ps["prior"])

    def test_sensitivity_vars_restrict_rows(self, beta_model):
        result = hb.compare_models(
            beta_model, run_prior_sensitivity=True, sensitivity_vars=["x1"]
        )
        assert len(result.prior_sensitivity) == 1

    def test_does_not_mutate_idata(self, beta_model):
        """log_prior and the rebuilt offsets live on a copy only."""
        groups0 = tuple(beta_model.result.idata.groups)
        hb.compare_models(beta_model, run_prior_sensitivity=True)
        assert tuple(beta_model.result.idata.groups) == groups0
        posterior_vars = beta_model.result.idata.posterior.data_vars
        assert not any(str(v).endswith("_offset") for v in posterior_vars)

    def test_summary_reports_prior_sensitivity(self, beta_model):
        s = hb.compare_models(beta_model, run_prior_sensitivity=True).summary()
        assert "Prior sensitivity" in s


# ===========================================================================
# compare_models() — multiple models
# ===========================================================================

class TestCompareModelsMulti:

    def test_comparison_table_exists(self, beta_model, beta_model_small):
        """Multi-model comparison should produce a comparison table."""
        result = hb.compare_models([beta_model, beta_model_small])
        assert result.comparison_table is not None

    def test_comparison_table_is_dataframe(self, beta_model, beta_model_small):
        result = hb.compare_models([beta_model, beta_model_small])
        assert isinstance(result.comparison_table, pd.DataFrame)

    def test_loo_is_dict_for_multi(self, beta_model, beta_model_small):
        models = [beta_model, beta_model_small]
        result = hb.compare_models(models)
        assert isinstance(result.loo, dict)
        # Keys follow the documented "model_<i>" convention (see comparison.py).
        for i in range(len(models)):
            assert f"model_{i}" in result.loo

    def test_compare_plot_for_multi(self, beta_model, beta_model_small):
        """≥2 models → az.plot_compare summary; failures land in plot_errors."""
        import matplotlib.figure
        result = hb.compare_models([beta_model, beta_model_small])
        assert "compare" not in result.plot_errors
        assert isinstance(result.compare_plot, matplotlib.figure.Figure)

    def test_rejects_models_fitted_to_different_data(self, beta_model, gaussian_model):
        """Equal row counts (100 each) but different responses — az.compare
        alone would accept this; the guard must not."""
        with pytest.raises(ValueError, match="different observations"):
            hb.compare_models([beta_model, gaussian_model])


# ===========================================================================
# compare_models() — argument validation wiring (P1-4)
# ===========================================================================
# The pure-logic contract lives in tests/test_v0_comparison_args.py (fast, no
# Bambi). These confirm the validator is actually wired into compare_models on a
# real fitted model — unsupported args must raise, not be ignored.

class TestCompareModelsArgValidation:

    def test_rejects_waic(self, beta_model):
        with pytest.raises(NotImplementedError):
            hb.compare_models(beta_model, metrics=["waic"])

    def test_rejects_empty_metrics(self, beta_model):
        with pytest.raises(ValueError):
            hb.compare_models(beta_model, metrics=[])

    def test_rejects_sensitivity_vars_without_run(self, beta_model):
        with pytest.raises(ValueError, match="run_prior_sensitivity"):
            hb.compare_models(beta_model, sensitivity_vars=["x1"])

    def test_rejects_invalid_ppc_draws(self, beta_model):
        with pytest.raises(ValueError):
            hb.compare_models(beta_model, n_draws_ppc=0)
        with pytest.raises(ValueError):
            hb.compare_models(beta_model, n_draws_ppc=-10)

    def test_accepts_loo_metric(self, beta_model):
        result = hb.compare_models(beta_model, metrics=["loo"])
        assert result.loo is not None

    def test_accepts_single_string_metric(self, beta_model):
        result = hb.compare_models(beta_model, metrics="loo")
        assert result.loo is not None


class TestCompareModelsUsesPrecomputedLoo:
    """Multi-model az.compare must receive the precomputed ELPDData dict (no
    second LOO pass), keyed ``model_<i>``."""

    def test_compare_receives_elpd_dict(self, beta_model, beta_model_small):
        import unittest.mock as mock

        import arviz as az

        original_compare = az.compare
        with mock.patch("arviz.compare", wraps=original_compare) as spy:
            hb.compare_models([beta_model, beta_model_small])

        assert spy.call_count == 1
        compare_arg = spy.call_args.args[0]
        assert isinstance(compare_arg, dict)
        assert all(key.startswith("model_") for key in compare_arg)
        # Values are precomputed LOO results (ELPDData), not raw InferenceData.
        assert all(type(v).__name__ == "ELPDData" for v in compare_arg.values())
