"""M5 integration tests — end-to-end workflow using load_dataset().

Tests the full pipeline:
    create_model → fit → estimate_areas → check_convergence → compare_models

Uses real built-in datasets (30 areas each), not synthetic fixtures.

Run with:
    pytest -m slow tests/test_v1_integration.py -v

Skip with:
    pytest -m "not slow"
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import hbsaemp as hb

pytestmark = pytest.mark.slow

# Fast config shared across all integration tests
_CFG = hb.ModelConfig(
    draws=200, tune=200, chains=2, cores=1,
    target_accept=0.9, random_seed=42,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_area_result(result: hb.AreaEstimatesResult, n_areas: int) -> None:
    """Common assertions for any AreaEstimatesResult."""
    assert isinstance(result, hb.AreaEstimatesResult)
    assert not result.result_table.empty
    assert len(result.result_table) == n_areas
    required = {"mean", "sd", "ci_lower", "ci_upper", "rse_pct", "mse", "rmse"}
    assert required.issubset(set(result.result_table.columns))
    assert (result.result_table["sd"] > 0).all()
    assert (result.result_table["ci_lower"] <= result.result_table["ci_upper"]).all()
    assert isinstance(result.mean_rse, float)
    assert isinstance(result.mean_mse, float)


def _check_convergence_result(result: hb.ConvergenceResult) -> None:
    """Common assertions for ConvergenceResult."""
    assert isinstance(result, hb.ConvergenceResult)
    assert result.rhat_ess is not None
    assert not result.rhat_ess.empty
    assert "r_hat" in result.rhat_ess.columns
    assert "ess_bulk" in result.rhat_ess.columns


def _check_comparison_result(result: hb.ComparisonResult) -> None:
    """Common assertions for single-model ComparisonResult."""
    import arviz as az
    assert isinstance(result, hb.ComparisonResult)
    assert result.loo is not None
    # WAIC is opt-in: ArviZ ≥0.20 removed ``az.waic`` and compare_models()
    # degrades to None in that case.
    if hasattr(az, "waic"):
        assert result.waic is not None
    else:
        assert result.waic is None
    assert result.comparison_table is None  # single model → no table


# ===========================================================================
# Gaussian Fay-Herriot integration
# ===========================================================================

class TestIntegrationGaussian:
    """Full workflow with data_fhnorm (Gaussian Fay-Herriot)."""

    @pytest.fixture(scope="class")
    def model(self) -> hb.BaseModel:
        df = hb.load_dataset("data_fhnorm")
        m = hb.create_model(
            "y ~ x1 + x2 + (1|group)",
            family="gaussian",
            data=df,
            sampling_var="D",
            config=_CFG,
        )
        m.fit()
        return m

    @pytest.fixture(scope="class")
    def df(self) -> pd.DataFrame:
        return hb.load_dataset("data_fhnorm")

    def test_fit_success(self, model):
        assert model.is_fitted
        assert model.result.family == "gaussian"

    def test_idata_groups(self, model):
        idata = model.result.idata
        assert hasattr(idata, "posterior")
        assert hasattr(idata, "log_likelihood")

    def test_fh_log_sqrt_D_added(self, model):
        assert "log_sqrt_D" in model.result.data.columns

    def test_estimate_areas(self, model, df):
        result = hb.estimate_areas(model)
        _check_area_result(result, len(model.result.data))
        # Gaussian: means are unbounded reals
        assert np.isfinite(result.result_table["mean"]).all()

    def test_estimate_areas_has_group(self, model):
        result = hb.estimate_areas(model)
        assert "group" in result.result_table.columns

    def test_check_convergence(self, model):
        result = hb.check_convergence(model, plot_types=[])
        _check_convergence_result(result)

    def test_compare_models_single(self, model):
        result = hb.compare_models(model)
        _check_comparison_result(result)

    def test_estimate_areas_out_of_sample(self, model, df):
        new_data = df.tail(5).reset_index(drop=True)
        result = hb.estimate_areas(model, new_data=new_data)
        assert len(result.result_table) == 5


# ===========================================================================
# Beta logit-normal integration
# ===========================================================================

class TestIntegrationBeta:
    """Full workflow with data_betalogitnorm (Beta logit-normal)."""

    @pytest.fixture(scope="class")
    def model(self) -> hb.BaseModel:
        df = hb.load_dataset("data_betalogitnorm")
        m = hb.create_model(
            "y ~ x1 + x2 + (1|group)",
            family="beta",
            data=df,
            n="n", deff="deff",
            config=_CFG,
        )
        m.fit()
        return m

    @pytest.fixture(scope="class")
    def df(self) -> pd.DataFrame:
        return hb.load_dataset("data_betalogitnorm")

    def test_fit_success(self, model):
        assert model.is_fitted
        assert model.result.family == "beta"

    def test_idata_groups(self, model):
        assert hasattr(model.result.idata, "posterior")
        assert hasattr(model.result.idata, "log_likelihood")

    def test_y_in_unit_interval_after_preprocessing(self, model):
        """y remains in (0, 1) after preprocessing (no squeeze=True; data_betalogitnorm
        clips y to (1e-6, 1-1e-6) at generation time)."""
        y = model.result.data["y"]
        assert (y > 0).all() and (y < 1).all()

    def test_log_phi_added(self, model):
        assert "log_phi" in model.result.data.columns

    def test_estimate_areas(self, model):
        result = hb.estimate_areas(model)
        _check_area_result(result, len(model.result.data))
        # Beta: means must be in (0, 1)
        means = result.result_table["mean"]
        assert (means > 0).all() and (means < 1).all()

    def test_check_convergence(self, model):
        result = hb.check_convergence(model, plot_types=[])
        _check_convergence_result(result)

    def test_compare_models_single(self, model):
        result = hb.compare_models(model)
        _check_comparison_result(result)

    def test_update_model(self, model):
        new_result = hb.update_model(model, draws=150, tune=150)
        assert isinstance(new_result, hb.ModelResult)
        assert new_result.config.draws == 150
        # model._result updated in-place
        assert model.result is new_result


# ===========================================================================
# Binomial logit-normal integration
# ===========================================================================

class TestIntegrationBinomial:
    """Full workflow with data_binlogitnorm (Binomial logit-normal)."""

    @pytest.fixture(scope="class")
    def model(self) -> hb.BaseModel:
        df = hb.load_dataset("data_binlogitnorm")
        m = hb.create_model(
            "y ~ x1 + x2 + (1|group)",
            family="binomial",
            data=df,
            trials="n",
            config=_CFG,
        )
        m.fit()
        return m

    @pytest.fixture(scope="class")
    def df(self) -> pd.DataFrame:
        return hb.load_dataset("data_binlogitnorm")

    def test_fit_success(self, model):
        assert model.is_fitted
        assert model.result.family == "binomial"

    def test_idata_groups(self, model):
        assert hasattr(model.result.idata, "posterior")
        assert hasattr(model.result.idata, "log_likelihood")

    def test_extra_trials_col(self, model):
        assert model.result.extra["trials_col"] == "n"

    def test_estimate_areas(self, model):
        result = hb.estimate_areas(model)
        _check_area_result(result, len(model.result.data))
        # estimate_areas() uses kind="response_params" → idata.posterior["p"]
        # Binomial: SAE target is the success probability p_i ∈ (0, 1),
        # NOT counts y_i* (which posterior predictive would return).
        means = result.result_table["mean"]
        assert (means > 0).all() and (means < 1).all(), \
            f"Binomial p estimates must be in (0, 1): {means.describe()}"

    def test_check_convergence(self, model):
        result = hb.check_convergence(model, plot_types=[])
        _check_convergence_result(result)

    def test_compare_models_single(self, model):
        result = hb.compare_models(model)
        _check_comparison_result(result)


# ===========================================================================
# Lognormal integration
# ===========================================================================

class TestIntegrationLognormal:
    """Full workflow with data_lnln (Lognormal-lognormal)."""

    @pytest.fixture(scope="class")
    def model(self) -> hb.BaseModel:
        df = hb.load_dataset("data_lnln")
        m = hb.create_model(
            "y_log_obs ~ x1 + x2 + (1|group)",
            family="lognormal",
            data=df,
            sampling_var="psi_i",
            config=_CFG,
        )
        m.fit()
        return m

    @pytest.fixture(scope="class")
    def df(self) -> pd.DataFrame:
        return hb.load_dataset("data_lnln")

    def test_fit_success(self, model):
        assert model.is_fitted
        assert model.result.family == "lognormal"

    def test_idata_groups(self, model):
        assert hasattr(model.result.idata, "posterior")
        assert hasattr(model.result.idata, "log_likelihood")

    def test_log_sqrt_D_added(self, model):
        assert "log_sqrt_D" in model.result.data.columns

    def test_log_sqrt_D_values(self, model):
        df = model.result.data
        np.testing.assert_allclose(
            df["log_sqrt_D"].values,
            0.5 * np.log(df["psi_i"].values),
            rtol=1e-6,
        )

    def test_estimate_areas_log_scale(self, model):
        result = hb.estimate_areas(model)
        _check_area_result(result, len(model.result.data))
        # Lognormal: predictions on log scale → any real value is valid
        assert np.isfinite(result.result_table["mean"]).all()

    def test_check_convergence(self, model):
        result = hb.check_convergence(model, plot_types=[])
        _check_convergence_result(result)

    def test_compare_models_single(self, model):
        result = hb.compare_models(model)
        _check_comparison_result(result)


# ===========================================================================
# Multi-model comparison
# ===========================================================================

class TestMultiModelComparison:
    """compare_models() with two models — comparison table must be produced."""

    @pytest.fixture(scope="class")
    def two_beta_models(self) -> tuple[hb.BaseModel, hb.BaseModel]:
        df = hb.load_dataset("data_betalogitnorm")
        cfg = hb.ModelConfig(draws=200, tune=200, chains=2, cores=1,
                             random_seed=0)
        # Model 1: with phi pinned
        m1 = hb.create_model(
            "y ~ x1 + x2 + (1|group)",
            family="beta", data=df, n="n", deff="deff", config=cfg,
        )
        m1.fit()
        # Model 2: without phi pinning (Bambi estimates kappa)
        m2 = hb.create_model(
            "y ~ x1 + x2",
            family="beta", data=df, config=cfg,
        )
        m2.fit()
        return m1, m2

    def test_comparison_table_exists(self, two_beta_models):
        m1, m2 = two_beta_models
        result = hb.compare_models([m1, m2])
        assert result.comparison_table is not None

    def test_comparison_table_is_dataframe(self, two_beta_models):
        m1, m2 = two_beta_models
        result = hb.compare_models([m1, m2])
        assert isinstance(result.comparison_table, pd.DataFrame)

    def test_loo_dict_keys(self, two_beta_models):
        m1, m2 = two_beta_models
        result = hb.compare_models([m1, m2])
        assert isinstance(result.loo, dict)
        assert "model_0" in result.loo
        assert "model_1" in result.loo

    def test_waic_dict_keys(self, two_beta_models):
        m1, m2 = two_beta_models
        result = hb.compare_models([m1, m2])
        assert isinstance(result.waic, dict)


# ===========================================================================
# R-style alias smoke tests
# ===========================================================================

class TestRAliases:
    """All R-style aliases must work end-to-end."""

    @pytest.fixture(scope="class")
    def model(self) -> hb.BaseModel:
        df = hb.load_dataset("data_fhnorm")
        cfg = hb.ModelConfig(draws=200, tune=200, chains=2, cores=1, random_seed=1)
        m = hb.hbm(
            "y ~ x1 + x2 + (1|group)",
            family="gaussian",
            data=df,
            sampling_var="D",
            config=cfg,
        )
        m.fit()
        return m

    def test_hbm_creates_model(self, model):
        """hbm() is create_model()."""
        assert isinstance(model, hb.BaseModel)
        assert model.is_fitted

    def test_hbsae_alias(self, model):
        """hbsae() is estimate_areas()."""
        result = hb.hbsae(model)
        assert isinstance(result, hb.AreaEstimatesResult)
        assert not result.result_table.empty

    def test_hbcc_alias(self, model):
        """hbcc() is check_convergence()."""
        result = hb.hbcc(model, plot_types=[])
        assert isinstance(result, hb.ConvergenceResult)
        assert result.rhat_ess is not None

    def test_hbmc_alias(self, model):
        """hbmc() is compare_models()."""
        result = hb.hbmc(model)
        assert isinstance(result, hb.ComparisonResult)
        assert result.loo is not None

    def test_update_hbm_alias(self, model):
        """update_hbm() is update_model()."""
        result = hb.update_hbm(model, draws=150, tune=150)
        assert isinstance(result, hb.ModelResult)
        assert result.config.draws == 150


# ===========================================================================
# AreaEstimatesResult.summary() integration
# ===========================================================================

class TestAreaEstimatesSummary:

    def test_summary_output(self):
        df = hb.load_dataset("data_betalogitnorm")
        cfg = hb.ModelConfig(draws=200, tune=200, chains=2, cores=1, random_seed=5)
        m = hb.create_model(
            "y ~ x1 + x2",
            family="beta", data=df, config=cfg,
        )
        m.fit()
        result = hb.estimate_areas(m)
        s = result.summary()
        assert "Areas" in s
        assert "RSE" in s or "rse" in s.lower()

    def test_convergence_summary_output(self):
        df = hb.load_dataset("data_fhnorm")
        cfg = hb.ModelConfig(draws=200, tune=200, chains=2, cores=1, random_seed=6)
        m = hb.create_model(
            "y ~ x1 + x2 + (1|group)",
            family="gaussian", data=df, sampling_var="D", config=cfg,
        )
        m.fit()
        result = hb.check_convergence(m, plot_types=[])
        s = result.summary()
        assert "ConvergenceResult" in s
        assert "Rhat" in s or "rhat" in s.lower()
