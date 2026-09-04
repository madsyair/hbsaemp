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
        """50% HDI must be strictly narrower than 95% HDI for every area.

        beta_fitted is seeded (random_seed=42), so this is deterministic.
        Asserting per-area rather than mean width catches a partial regression
        where one area collapses to width≈0.
        """
        r95 = hb.estimate_areas(beta_fitted, ci_prob=0.95)
        r50 = hb.estimate_areas(beta_fitted, ci_prob=0.50)
        width95 = r95.result_table["ci_upper"] - r95.result_table["ci_lower"]
        width50 = r50.result_table["ci_upper"] - r50.result_table["ci_lower"]
        assert (width50 < width95).all(), (
            f"50% HDI not strictly narrower than 95% HDI in all areas; "
            f"min(width95 - width50) = {(width95 - width50).min():.4f}"
        )

    def test_estimate_areas_in_sample_correct_after_oos(self, beta_fitted, data_beta):
        """In-sample estimates must be identical before and after an OOS call.

        Regression guard: predict() must not mutate result.idata['mu'] in place
        (inplace=False). Otherwise an intervening OOS estimate_areas() would
        corrupt the in-sample mu and the next in-sample call returns wrong
        numbers. The module-scoped fixture also makes this a cross-test
        leakage regression.
        """
        is_before = hb.estimate_areas(beta_fitted)
        _ = hb.estimate_areas(beta_fitted, new_data=data_beta.head(5).reset_index(drop=True))
        is_after = hb.estimate_areas(beta_fitted)
        pd.testing.assert_frame_equal(is_before.result_table, is_after.result_table)


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

    def test_not_fitted_is_not_wrapped(self, data_beta):
        """The fitted-model guard must not be reclassified as EstimationError.

        `estimate_areas()` wraps its estimation body, but the guards before it
        keep their own contract: callers distinguish "wrong call order" from
        "this model/data cannot be estimated".
        """
        cfg = hb.ModelConfig(draws=100, tune=100, chains=2)
        m = hb.create_model("y ~ x1", family="beta",
                             data=data_beta, n="n", deff="deff", config=cfg)
        with pytest.raises(hb.ModelNotFittedError):
            hb.estimate_areas(m)
        assert not issubclass(hb.ModelNotFittedError, hb.EstimationError)


class TestEstimateAreasFailureWrapping:
    """Estimation failures surface as EstimationError, not raw library errors."""

    def test_new_data_missing_predictor_raises_estimation_error(
        self, beta_fitted, data_beta
    ):
        bad = data_beta.tail(10).drop(columns=["x1"]).reset_index(drop=True)
        with pytest.raises(hb.HBSAEError) as exc_info:
            hb.estimate_areas(beta_fitted, new_data=bad)
        # Either the data layer catches it first (DataValidationError) or the
        # estimation body does (EstimationError) — both are package errors with
        # an actionable message; a raw KeyError/ValueError would not be.
        assert isinstance(
            exc_info.value, (hb.EstimationError, hb.DataValidationError)
        )

    def test_estimation_error_is_hbsae_error(self):
        assert issubclass(hb.EstimationError, hb.HBSAEError)


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
        """estimate_areas() must use posterior of mu, not posterior predictive.

        Direct check: posterior mu draws have strictly smaller per-area SD than
        posterior predictive draws, because mu omits the residual sampling
        variance. Comparing against the predictive SD is a tight contract
        check; comparing against the observed y-range is fragile when data
        variance is high.
        """
        result = hb.estimate_areas(gaussian_fitted)
        latent_sd = result.result_table["sd"].values
        pp_draws = gaussian_fitted.predict(kind="response")
        pp_sd = pp_draws.std(axis=0)
        assert (latent_sd > 0).all(), "Posterior of mu has zero uncertainty — unexpected"
        # Latent strictly less than predictive (excludes sampling variance).
        # rtol=1e-3 absorbs MCMC noise without masking a swapped extraction.
        assert (latent_sd < pp_sd * (1 - 1e-3)).all(), (
            f"estimate_areas() SD not strictly less than predictive SD — "
            f"likely returned posterior predictive instead of posterior of mu. "
            f"max(latent_sd - pp_sd) = {(latent_sd - pp_sd).max():.4f}"
        )


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

    def test_update_overrides_target_accept(self, beta_fitted):
        """target_accept is the key non-convergence lever — must be overridable (TODO-11)."""
        result = hb.update_model(beta_fitted, draws=120, tune=120, target_accept=0.95)
        assert result.config.target_accept == 0.95
        assert beta_fitted._config.target_accept == 0.95

    def test_update_overrides_random_seed(self, beta_fitted):
        result = hb.update_model(beta_fitted, draws=120, tune=120, random_seed=123)
        assert result.config.random_seed == 123

    def test_update_invalid_target_accept_raises(self, beta_fitted):
        """Override flows through ModelConfig.__post_init__, which rejects target_accept ∉ (0,1)
        before any refit happens."""
        with pytest.raises(ValueError):
            hb.update_model(beta_fitted, target_accept=1.5)


# ---------------------------------------------------------------------------
# update_model() — regression tests for the duplicate-kwarg TypeError
# ---------------------------------------------------------------------------
# Before the FAMILY_SPECS dispatch fix, BaseModel.__init__ swallowed
# unrelated family kwargs into self._kwargs; update_model() then forwarded
# both explicit family kwargs AND **model._kwargs, producing
# "TypeError: got multiple values for keyword argument" on the FIRST update
# for every family. These tests pin that behaviour going forward.


def _build_and_fit(family: str, df: pd.DataFrame) -> hb.BaseModel:
    """Tiny MCMC fit (draws=120, chains=2) per active family."""
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
        raise ValueError(f"Unsupported family in _build_and_fit: {family!r}.")
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
