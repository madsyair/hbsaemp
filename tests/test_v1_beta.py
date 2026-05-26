"""BetaModel v1 tests — require Bambi MCMC (mark: slow).

Run with:
    pytest -m slow tests/test_v1_beta.py -v

Skip with:
    pytest -m "not slow"
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import hbsaemp as hb

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def beta_model_fitted(data_beta: pd.DataFrame) -> hb.BaseModel:
    """Fit a BetaModel with n_col/deff_col on the conftest beta fixture."""
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
def beta_model_no_phi(data_beta: pd.DataFrame) -> hb.BaseModel:
    """BetaModel without n_col/deff_col — Bambi estimates kappa."""
    cfg = hb.ModelConfig(draws=200, tune=200, chains=2, cores=1,
                         target_accept=0.9, random_seed=42)
    model = hb.create_model(
        "y ~ x1 + x2",
        family="beta",
        data=data_beta,
        config=cfg,
    )
    model.fit()
    return model


# ---------------------------------------------------------------------------
# fit() — ModelResult contract
# ---------------------------------------------------------------------------

class TestBetaFit:

    def test_returns_model_result(self, beta_model_fitted):
        assert isinstance(beta_model_fitted.result, hb.ModelResult)

    def test_is_fitted(self, beta_model_fitted):
        assert beta_model_fitted.is_fitted

    def test_result_family(self, beta_model_fitted):
        assert beta_model_fitted.result.family == "beta"

    def test_result_formula(self, beta_model_fitted):
        assert "y" in beta_model_fitted.result.formula
        assert "x1" in beta_model_fitted.result.formula

    def test_idata_has_posterior(self, beta_model_fitted):
        idata = beta_model_fitted.result.idata
        assert hasattr(idata, "posterior")
        assert idata.posterior is not None

    def test_idata_has_log_likelihood(self, beta_model_fitted):
        """log_likelihood group must exist — required for LOO/WAIC (M4)."""
        idata = beta_model_fitted.result.idata
        assert hasattr(idata, "log_likelihood"), \
            "idata must have log_likelihood group (idata_kwargs={'log_likelihood': True})"

    def test_result_data_is_clean(self, beta_model_fitted):
        """result.data is the preprocessed DataFrame — y in (0, 1).

        No squeeze=True was passed; conftest data is pre-clipped to (1e-6, 1-1e-6).
        """
        df = beta_model_fitted.result.data
        assert not df["y"].isna().any()
        # Data already in (0, 1) — no squeeze transform applied here
        assert (df["y"] > 0).all() and (df["y"] < 1).all()

    def test_result_extra_response(self, beta_model_fitted):
        assert beta_model_fitted.result.extra["response"] == "y"

    def test_result_extra_group(self, beta_model_fitted):
        assert beta_model_fitted.result.extra["group"] == "group"

    def test_fitted_at_is_set(self, beta_model_fitted):
        assert beta_model_fitted.result.fitted_at is not None

    def test_backend_model_is_bambi(self, beta_model_fitted):
        import bambi as bmb
        assert isinstance(beta_model_fitted.result.backend_model, bmb.Model)

    def test_posterior_mu_exists(self, beta_model_fitted):
        """mu (mean parameter) must appear in posterior."""
        posterior = beta_model_fitted.result.idata.posterior
        # Bambi stores intercept-related terms; check at least one variable
        assert len(list(posterior.data_vars)) > 0

    def test_fit_without_phi_pinning(self, beta_model_no_phi):
        """Model without n/deff also fits — Bambi estimates kappa."""
        assert beta_model_no_phi.is_fitted
        assert beta_model_no_phi.result.extra["n_col"] is None


# ---------------------------------------------------------------------------
# predict() — shape and range
# ---------------------------------------------------------------------------

class TestBetaPredict:

    def test_predict_in_sample_shape(self, beta_model_fitted, data_beta):
        draws = beta_model_fitted.predict()
        total_draws = beta_model_fitted.result.config.draws * \
                      beta_model_fitted.result.config.chains
        n_obs = len(beta_model_fitted.result.data)
        assert draws.shape == (total_draws, n_obs), \
            f"Expected ({total_draws}, {n_obs}), got {draws.shape}"

    def test_predict_values_in_unit_interval(self, beta_model_fitted):
        draws = beta_model_fitted.predict()
        assert (draws >= 0).all() and (draws <= 1).all(), \
            "Beta response-scale predictions must be in [0, 1]"

    def test_predict_out_of_sample_shape(self, beta_model_fitted, data_beta):
        """Predict on a held-out subset (last 10 rows)."""
        new_data = data_beta.tail(10).reset_index(drop=True)
        draws = beta_model_fitted.predict(new_data=new_data)
        total_draws = beta_model_fitted.result.config.draws * \
                      beta_model_fitted.result.config.chains
        assert draws.shape == (total_draws, 10), \
            f"Expected ({total_draws}, 10), got {draws.shape}"

    def test_predict_n_samples_subset(self, beta_model_fitted):
        draws = beta_model_fitted.predict(n_samples=50)
        assert draws.shape[0] == 50

    def test_predict_before_fit_raises(self, data_beta):
        cfg = hb.ModelConfig(draws=100, tune=100, chains=2)
        m = hb.create_model("y ~ x1", family="beta", data=data_beta, config=cfg)
        with pytest.raises(hb.ModelNotFittedError):
            m.predict()

    def test_predict_kind_linear_removed(self, beta_model_fitted):
        """kind='linear' was removed in Bambi 0.14+ — must raise ValueError."""
        with pytest.raises(ValueError, match="linear"):
            beta_model_fitted.predict(kind="linear")

    def test_predict_kind_response_params_in_unit_interval(self, beta_model_fitted):
        """kind='response_params' returns posterior of mu (logit link → (0,1))."""
        draws = beta_model_fitted.predict(kind="response_params")
        assert (draws >= 0).all() and (draws <= 1).all()
        assert np.isfinite(draws).all()


# ---------------------------------------------------------------------------
# Custom priors
# ---------------------------------------------------------------------------

class TestBetaCustomPriors:

    def test_fit_with_custom_priors(self, data_beta):
        cfg = hb.ModelConfig(draws=100, tune=100, chains=2, cores=1,
                             random_seed=0)
        priors = {
            "x1": {"dist": "Normal", "mu": 0, "sigma": 0.5},
            "x2": {"dist": "Normal", "mu": 0, "sigma": 0.5},
        }
        model = hb.create_model(
            "y ~ x1 + x2",
            family="beta",
            data=data_beta,
            priors=priors,
            config=cfg,
        )
        result = model.fit()
        assert result.is_fitted
        assert result.priors is priors


# ---------------------------------------------------------------------------
# ModelResult.summary()
# ---------------------------------------------------------------------------

class TestBetaSummary:

    def test_summary_contains_family(self, beta_model_fitted):
        s = beta_model_fitted.summary()
        assert "beta" in s.lower()

    def test_summary_contains_fitted(self, beta_model_fitted):
        s = beta_model_fitted.result.summary()
        assert "beta" in s.lower()
        assert "fitted" in s.lower() or "bambi" in s.lower()
