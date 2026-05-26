"""estimate_areas() and update_model() v1 tests — require Bambi MCMC.

Run with:
    pytest -m slow tests/test_v1_estimation.py -v

Skip with:
    pytest -m "not slow"
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import hbsaemp as hb

pytestmark = pytest.mark.slow

# Expected columns in every AreaEstimatesResult.result_table
_EXPECTED_COLS = {"mean", "sd", "ci_lower", "ci_upper", "rse_pct", "mse", "rmse"}


# ---------------------------------------------------------------------------
# Shared fixtures (module-scoped — fit once)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def beta_fitted(data_beta: pd.DataFrame) -> hb.BaseModel:
    """Fitted BetaModel used by all estimation tests."""
    cfg = hb.ModelConfig(draws=200, tune=200, chains=2, cores=1,
                         target_accept=0.9, random_seed=42)
    model = hb.create_model(
        "y ~ x1 + x2 + (1|group)",
        family="beta",
        data=data_beta,
        n="n", deff="deff",
        config=cfg,
    )
    model.fit()
    return model


@pytest.fixture(scope="module")
def gaussian_fitted(data_gaussian: pd.DataFrame) -> hb.BaseModel:
    """Fitted GaussianModel used by estimation tests."""
    cfg = hb.ModelConfig(draws=200, tune=200, chains=2, cores=1,
                         target_accept=0.9, random_seed=7)
    model = hb.create_model(
        "y ~ x1 + x2 + (1|group)",
        family="gaussian",
        data=data_gaussian,
        config=cfg,
    )
    model.fit()
    return model


# ---------------------------------------------------------------------------
# estimate_areas() — in-sample
# ---------------------------------------------------------------------------

class TestEstimateAreasInSample:

    def test_returns_area_estimates_result(self, beta_fitted):
        result = hb.estimate_areas(beta_fitted)
        assert isinstance(result, hb.AreaEstimatesResult)

    def test_result_table_not_empty(self, beta_fitted):
        result = hb.estimate_areas(beta_fitted)
        assert not result.result_table.empty

    def test_result_table_columns(self, beta_fitted):
        result = hb.estimate_areas(beta_fitted)
        cols = set(result.result_table.columns)
        assert _EXPECTED_COLS.issubset(cols), \
            f"Missing columns: {_EXPECTED_COLS - cols}"

    def test_result_table_n_rows_matches_data(self, beta_fitted):
        result = hb.estimate_areas(beta_fitted)
        n_fitted = len(beta_fitted.result.data)
        assert len(result.result_table) == n_fitted

    def test_mean_in_unit_interval_for_beta(self, beta_fitted):
        """Beta model: posterior of mu is in (0, 1).

        estimate_areas() uses kind='response_params' → draws from
        idata.posterior['mu'], which is the shrinkage estimate of the Beta
        mean parameter, strictly in (0, 1).
        """
        result = hb.estimate_areas(beta_fitted)
        means = result.result_table["mean"]
        assert (means > 0).all() and (means < 1).all(), \
            f"Beta means out of (0, 1): {means.describe()}"

    def test_sd_positive(self, beta_fitted):
        result = hb.estimate_areas(beta_fitted)
        assert (result.result_table["sd"] > 0).all()

    def test_ci_lower_le_mean_le_upper(self, beta_fitted):
        result = hb.estimate_areas(beta_fitted)
        df = result.result_table
        assert (df["ci_lower"] <= df["mean"]).all()
        assert (df["mean"] <= df["ci_upper"]).all()

    def test_rse_positive(self, beta_fitted):
        result = hb.estimate_areas(beta_fitted)
        assert (result.result_table["rse_pct"] > 0).all()

    def test_mse_equals_var(self, beta_fitted):
        """MSE = posterior variance; RMSE = sqrt(MSE)."""
        result = hb.estimate_areas(beta_fitted)
        df = result.result_table
        np.testing.assert_allclose(df["rmse"].values, np.sqrt(df["mse"].values), rtol=1e-6)

    def test_mean_rse_scalar(self, beta_fitted):
        result = hb.estimate_areas(beta_fitted)
        assert isinstance(result.mean_rse, float)
        assert result.mean_rse > 0

    def test_mean_mse_scalar(self, beta_fitted):
        result = hb.estimate_areas(beta_fitted)
        assert isinstance(result.mean_mse, float)
        assert result.mean_mse > 0

    def test_group_column_present(self, beta_fitted):
        """Group column appended when model.result.extra['group'] is set."""
        result = hb.estimate_areas(beta_fitted)
        group_col = beta_fitted.result.extra.get("group")
        if group_col is not None:
            assert group_col in result.result_table.columns

    def test_r_alias_hbsae(self, beta_fitted):
        result = hb.hbsae(beta_fitted)
        assert isinstance(result, hb.AreaEstimatesResult)


# ---------------------------------------------------------------------------
# estimate_areas() — out-of-sample (new_data)
# ---------------------------------------------------------------------------

class TestEstimateAreasOutOfSample:

    def test_new_data_shape(self, beta_fitted, data_beta):
        new_data = data_beta.tail(10).reset_index(drop=True)
        result = hb.estimate_areas(beta_fitted, new_data=new_data)
        assert len(result.result_table) == 10

    def test_new_data_has_columns(self, beta_fitted, data_beta):
        new_data = data_beta.tail(10).reset_index(drop=True)
        result = hb.estimate_areas(beta_fitted, new_data=new_data)
        assert _EXPECTED_COLS.issubset(set(result.result_table.columns))

    def test_ci_prob_narrower_at_50(self, beta_fitted):
        """50% HDI should be narrower than 95% HDI."""
        r95 = hb.estimate_areas(beta_fitted, ci_prob=0.95)
        r50 = hb.estimate_areas(beta_fitted, ci_prob=0.50)
        width95 = (r95.result_table["ci_upper"] - r95.result_table["ci_lower"]).mean()
        width50 = (r50.result_table["ci_upper"] - r50.result_table["ci_lower"]).mean()
        assert width50 < width95, \
            f"50% CI ({width50:.4f}) not narrower than 95% CI ({width95:.4f})"


# ---------------------------------------------------------------------------
# estimate_areas() — ModelNotFittedError guard
# ---------------------------------------------------------------------------

class TestEstimateAreasGuard:

    def test_not_fitted_raises(self, data_beta):
        cfg = hb.ModelConfig(draws=100, tune=100, chains=2)
        m = hb.create_model("y ~ x1", family="beta",
                             data=data_beta, n="n", deff="deff", config=cfg)
        with pytest.raises(hb.ModelNotFittedError):
            hb.estimate_areas(m)


# ---------------------------------------------------------------------------
# AreaEstimatesResult helpers
# ---------------------------------------------------------------------------

class TestAreaEstimatesResultHelpers:

    def test_summary_contains_areas(self, beta_fitted):
        result = hb.estimate_areas(beta_fitted)
        s = result.summary()
        assert "Areas" in s or "areas" in s.lower()

    def test_repr_contains_n_areas(self, beta_fitted):
        result = hb.estimate_areas(beta_fitted)
        assert "n_areas" in repr(result)


# ---------------------------------------------------------------------------
# estimate_areas() — Gaussian model (continuous response)
# ---------------------------------------------------------------------------

class TestEstimateAreasGaussian:

    def test_gaussian_means_finite(self, gaussian_fitted):
        result = hb.estimate_areas(gaussian_fitted)
        assert np.isfinite(result.result_table["mean"]).all()

    def test_gaussian_n_rows(self, gaussian_fitted):
        result = hb.estimate_areas(gaussian_fitted)
        assert len(result.result_table) == len(gaussian_fitted.result.data)

    def test_gaussian_uses_posterior_mu(self, gaussian_fitted):
        """estimate_areas() must use posterior of μ_i (kind='response_params'),
        not the posterior predictive.  The posterior of μ has strictly smaller
        variance than the posterior predictive (no added sampling variance D_i),
        so SD must be positive and strictly less than the observed data range.
        """
        result = hb.estimate_areas(gaussian_fitted)
        df = result.result_table
        # All SDs positive (posterior of mu has uncertainty)
        assert (df["sd"] > 0).all()
        # SD < observed y range (shrinkage: posterior of mu is tighter than raw obs)
        y_range = float(
            gaussian_fitted.result.data[
                gaussian_fitted.result.extra["response"]
            ].max()
            - gaussian_fitted.result.data[
                gaussian_fitted.result.extra["response"]
            ].min()
        )
        assert (df["sd"] < y_range).all(), \
            f"Posterior SD exceeds observed range — likely using posterior predictive."


# ---------------------------------------------------------------------------
# update_model()
# ---------------------------------------------------------------------------

class TestUpdateModel:

    def test_returns_model_result(self, beta_fitted):
        result = hb.update_model(beta_fitted, draws=150, tune=150)
        assert isinstance(result, hb.ModelResult)

    def test_updated_draws(self, beta_fitted):
        """New result should reflect updated draws config."""
        result = hb.update_model(beta_fitted, draws=150, tune=150, chains=2)
        assert result.config.draws == 150

    def test_model_result_updated_in_place(self, beta_fitted):
        """After update_model(), model.result should reflect new fit."""
        result = hb.update_model(beta_fitted, draws=150, tune=150)
        assert beta_fitted.result is result

    def test_update_with_config_object(self, beta_fitted):
        new_cfg = hb.ModelConfig(draws=120, tune=120, chains=2, cores=1, random_seed=1)
        result = hb.update_model(beta_fitted, config=new_cfg)
        assert result.config.draws == 120
        assert result.config.chains == 2

    def test_update_with_new_data(self, beta_fitted, data_beta):
        new_df = data_beta.head(50).copy()
        result = hb.update_model(beta_fitted, new_data=new_df,
                                 draws=150, tune=150)
        assert isinstance(result, hb.ModelResult)
        assert result.is_fitted

    def test_update_new_data_changes_n(self, beta_fitted, data_beta):
        new_df = data_beta.head(50).copy()
        result = hb.update_model(beta_fitted, new_data=new_df,
                                 draws=150, tune=150)
        assert len(result.data) == 50

    def test_not_fitted_raises(self, data_beta):
        cfg = hb.ModelConfig(draws=100, tune=100, chains=2)
        m = hb.create_model("y ~ x1", family="beta",
                             data=data_beta, n="n", deff="deff", config=cfg)
        with pytest.raises(hb.ModelNotFittedError):
            hb.update_model(m, draws=200)

    def test_r_alias_update_hbm(self, beta_fitted):
        result = hb.update_hbm(beta_fitted, draws=150, tune=150)
        assert isinstance(result, hb.ModelResult)


# ---------------------------------------------------------------------------
# update_model() — regression tests for P0-1 (duplicate-kwarg TypeError)
# ---------------------------------------------------------------------------
# Before the _FAMILY_PARAMS dispatch fix, BaseModel.__init__ swallowed
# unrelated family kwargs into self._kwargs; update_model() then forwarded
# both explicit family kwargs AND **model._kwargs, producing
# "TypeError: got multiple values for keyword argument" on the FIRST update
# for every family. These tests pin that behaviour going forward.


def _build_and_fit(family: str, df: pd.DataFrame) -> hb.BaseModel:
    """Tiny MCMC fit (draws=120, chains=2) per family for regression tests.

    Lognormal is intentionally excluded — it requires the response to be
    already on log scale, which must be handled per-test (see test_update_lognormal).
    """
    cfg = hb.ModelConfig(draws=120, tune=120, chains=2, cores=1,
                         target_accept=0.9, random_seed=42)
    if family == "beta":
        model = hb.create_model(
            "y ~ x1 + (1|group)", family="beta", data=df,
            n="n", deff="deff", config=cfg,
        )
    elif family == "binomial":
        model = hb.create_model(
            "y ~ x1 + (1|group)", family="binomial", data=df,
            trials="n", config=cfg,
        )
    elif family == "gaussian":
        model = hb.create_model(
            "y ~ x1 + (1|group)", family="gaussian", data=df, config=cfg,
        )
    else:
        raise ValueError(f"Unsupported family in _build_and_fit: {family!r}. "
                         "Use test_update_lognormal for lognormal (log-scale response required).")
    model.fit()
    return model


class TestUpdateModelAllFamilies:
    """update_model() must not crash with duplicate-kwarg TypeError."""

    def test_update_gaussian(self, data_gaussian):
        m = _build_and_fit("gaussian", data_gaussian)
        result = hb.update_model(m, draws=80, tune=80)
        assert result.is_fitted

    def test_update_beta(self, data_beta):
        m = _build_and_fit("beta", data_beta)
        result = hb.update_model(m, draws=80, tune=80)
        assert result.is_fitted

    def test_update_binomial(self, data_binomial):
        m = _build_and_fit("binomial", data_binomial)
        result = hb.update_model(m, draws=80, tune=80)
        assert result.is_fitted

    def test_update_lognormal(self, data_lognormal):
        # Lognormal requires the response on log scale (per CLAUDE.md).
        df = data_lognormal.copy()
        df["y_log"] = np.log(df["y"])
        cfg = hb.ModelConfig(draws=120, tune=120, chains=2, cores=1,
                             target_accept=0.9, random_seed=42)
        m = hb.create_model(
            "y_log ~ x1 + (1|group)", family="lognormal", data=df, config=cfg,
        )
        m.fit()
        result = hb.update_model(m, draws=80, tune=80)
        assert result.is_fitted


class TestUpdateModelChained:
    """Chained updates: the duplicate-kwarg bug surfaced on the SECOND call.

    With _kwargs removed, calling update_model twice in a row must work.
    """

    def test_chained_update_beta(self, beta_fitted):
        first = hb.update_model(beta_fitted, draws=80, tune=80)
        assert first.is_fitted
        second = hb.update_model(beta_fitted, draws=100, tune=100)
        assert second.is_fitted
        # model.result must reflect the LATEST fit (in-place sync contract).
        assert beta_fitted.result is second
        assert beta_fitted._config.draws == 100
