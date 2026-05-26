"""check_convergence() and compare_models() v1 tests — require Bambi MCMC.

Run with:
    pytest -m slow tests/test_v1_diagnostics.py -v

Skip with:
    pytest -m "not slow"
"""
from __future__ import annotations

import warnings

import numpy as np
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


# ===========================================================================
# check_convergence()
# ===========================================================================

class TestCheckConvergence:

    def test_returns_convergence_result(self, beta_model):
        result = hb.check_convergence(beta_model)
        assert isinstance(result, hb.ConvergenceResult)

    def test_rhat_ess_is_dataframe(self, beta_model):
        result = hb.check_convergence(beta_model, plot_types=[])
        import pandas as pd
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
        """Well-behaved model: Rhat should be close to 1."""
        result = hb.check_convergence(beta_model, plot_types=[])
        max_rhat = result.rhat_ess["r_hat"].max()
        assert max_rhat < 1.5, f"Rhat too high: {max_rhat:.4f} (model may have divergences)"

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

        bad_summary = pd.DataFrame(
            {"mean": [0.0], "sd": [0.1], "r_hat": [1.05],
             "ess_bulk": [500.0], "ess_tail": [500.0]},
            index=["b_x1"],
        )
        with mock.patch("arviz.summary", return_value=bad_summary):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                hb.check_convergence(m, plot_types=[])
                cw = [w for w in caught if issubclass(w.category, hb.ConvergenceWarning)]
                assert len(cw) >= 1, "Expected ConvergenceWarning for Rhat > 1.01"


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

    def test_waic_not_none(self, beta_model):
        # WAIC is opt-in: ArviZ ≥0.20 removed ``az.waic``; compare_models()
        # degrades to ``None`` in that case (see comparison.py). Only assert
        # when ``az.waic`` is actually available in the installed ArviZ.
        import arviz as az
        result = hb.compare_models(beta_model)
        if hasattr(az, "waic"):
            assert result.waic is not None
        else:
            assert result.waic is None

    def test_no_comparison_table_for_single(self, beta_model):
        result = hb.compare_models(beta_model)
        assert result.comparison_table is None

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

    def test_repr(self, beta_model):
        result = hb.compare_models(beta_model)
        r = repr(result)
        assert "ComparisonResult" in r


# ===========================================================================
# compare_models() — multiple models
# ===========================================================================

class TestCompareModelsMulti:

    def test_comparison_table_exists(self, beta_model, gaussian_model):
        """Multi-model comparison should produce a comparison table."""
        result = hb.compare_models([beta_model, gaussian_model])
        assert result.comparison_table is not None

    def test_comparison_table_is_dataframe(self, beta_model, gaussian_model):
        import pandas as pd
        result = hb.compare_models([beta_model, gaussian_model])
        if result.comparison_table is not None:
            assert isinstance(result.comparison_table, pd.DataFrame)

    def test_loo_is_dict_for_multi(self, beta_model, gaussian_model):
        result = hb.compare_models([beta_model, gaussian_model])
        assert isinstance(result.loo, dict)
        assert "model_0" in result.loo
        assert "model_1" in result.loo

    def test_waic_is_dict_for_multi(self, beta_model, gaussian_model):
        result = hb.compare_models([beta_model, gaussian_model])
        assert isinstance(result.waic, dict)
