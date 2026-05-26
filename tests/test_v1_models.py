"""GaussianModel, LognormalModel, BinomialModel v1 tests — require Bambi MCMC.

Run with:
    pytest -m slow tests/test_v1_models.py -v

Skip with:
    pytest -m "not slow"

Coverage:
    - GaussianModel  (plain + Fay-Herriot with sampling_var)
    - LognormalModel (plain + FH with sampling_var, log-scale response)
    - BinomialModel  (formula rewrite, trials_col)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import hbsaemp as hb

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Extra fixtures — not in conftest.py
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def data_gaussian_fh() -> pd.DataFrame:
    """100-row Gaussian FH DataFrame with known sampling variances D."""
    rng = np.random.default_rng(42)
    n, g = 100, 10
    group = np.repeat(np.arange(1, g + 1), n // g)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    re = rng.normal(0, 0.3, g)[group - 1]
    D = rng.uniform(0.05, 0.3, n)           # known sampling variances
    y = 1.0 + 0.4 * x1 - 0.2 * x2 + re + rng.normal(0, np.sqrt(D))
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "D": D,
                         "group": group})


@pytest.fixture(scope="module")
def data_lognormal_plain() -> pd.DataFrame:
    """100-row DataFrame for LognormalModel — response is log-transformed."""
    rng = np.random.default_rng(42)
    n, g = 100, 10
    group = np.repeat(np.arange(1, g + 1), n // g)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    re = rng.normal(0, 0.3, g)[group - 1]
    # y_log is already on log scale (can be any real number)
    y_log = 1.0 + 0.3 * x1 - 0.2 * x2 + re + rng.normal(0, 0.4, n)
    return pd.DataFrame({"y_log": y_log, "x1": x1, "x2": x2,
                         "group": group})


@pytest.fixture(scope="module")
def data_lognormal_fh() -> pd.DataFrame:
    """100-row DataFrame for LognormalModel FH — with sampling variances psi."""
    rng = np.random.default_rng(99)
    n, g = 100, 10
    group = np.repeat(np.arange(1, g + 1), n // g)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    re = rng.normal(0, 0.3, g)[group - 1]
    psi = rng.uniform(0.05, 0.2, n)         # known log-scale sampling variances
    y_log = 1.0 + 0.3 * x1 - 0.2 * x2 + re + rng.normal(0, np.sqrt(psi))
    return pd.DataFrame({"y_log": y_log, "x1": x1, "x2": x2,
                         "psi": psi, "group": group})


# ---------------------------------------------------------------------------
# Fitted model fixtures (module-scoped — fit once per module)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gaussian_plain_fitted(data_gaussian) -> hb.BaseModel:
    """Plain GaussianModel (no sampling_var) fitted on conftest data_gaussian."""
    cfg = hb.ModelConfig(draws=200, tune=200, chains=2, cores=1,
                         target_accept=0.9, random_seed=42)
    model = hb.create_model(
        "y ~ x1 + x2 + (1|group)",
        family="gaussian",
        data=data_gaussian,
        config=cfg,
    )
    model.fit()
    return model


@pytest.fixture(scope="module")
def gaussian_fh_fitted(data_gaussian_fh) -> hb.BaseModel:
    """GaussianModel FH (with sampling_var) fitted on data_gaussian_fh."""
    cfg = hb.ModelConfig(draws=200, tune=200, chains=2, cores=1,
                         target_accept=0.9, random_seed=42)
    model = hb.create_model(
        "y ~ x1 + x2 + (1|group)",
        family="gaussian",
        data=data_gaussian_fh,
        sampling_var="D",
        config=cfg,
    )
    model.fit()
    return model


@pytest.fixture(scope="module")
def lognormal_plain_fitted(data_lognormal_plain) -> hb.BaseModel:
    """LognormalModel (no sampling_var) fitted on log-scale data."""
    cfg = hb.ModelConfig(draws=200, tune=200, chains=2, cores=1,
                         target_accept=0.9, random_seed=42)
    model = hb.create_model(
        "y_log ~ x1 + x2 + (1|group)",
        family="lognormal",
        data=data_lognormal_plain,
        config=cfg,
    )
    model.fit()
    return model


@pytest.fixture(scope="module")
def lognormal_fh_fitted(data_lognormal_fh) -> hb.BaseModel:
    """LognormalModel FH (with sampling_var) fitted."""
    cfg = hb.ModelConfig(draws=200, tune=200, chains=2, cores=1,
                         target_accept=0.9, random_seed=42)
    model = hb.create_model(
        "y_log ~ x1 + x2 + (1|group)",
        family="lognormal",
        data=data_lognormal_fh,
        sampling_var="psi",
        config=cfg,
    )
    model.fit()
    return model


@pytest.fixture(scope="module")
def binomial_fitted(data_binomial) -> hb.BaseModel:
    """BinomialModel fitted on conftest data_binomial."""
    cfg = hb.ModelConfig(draws=200, tune=200, chains=2, cores=1,
                         target_accept=0.9, random_seed=42)
    model = hb.create_model(
        "y ~ x1 + x2 + (1|group)",
        family="binomial",
        data=data_binomial,
        trials="n",
        config=cfg,
    )
    model.fit()
    return model


# ===========================================================================
# GaussianModel tests
# ===========================================================================

class TestGaussianFit:

    def test_returns_model_result(self, gaussian_plain_fitted):
        assert isinstance(gaussian_plain_fitted.result, hb.ModelResult)

    def test_is_fitted(self, gaussian_plain_fitted):
        assert gaussian_plain_fitted.is_fitted

    def test_result_family(self, gaussian_plain_fitted):
        assert gaussian_plain_fitted.result.family == "gaussian"

    def test_result_formula(self, gaussian_plain_fitted):
        assert "y" in gaussian_plain_fitted.result.formula
        assert "x1" in gaussian_plain_fitted.result.formula

    def test_idata_has_posterior(self, gaussian_plain_fitted):
        idata = gaussian_plain_fitted.result.idata
        assert hasattr(idata, "posterior") and idata.posterior is not None

    def test_idata_has_log_likelihood(self, gaussian_plain_fitted):
        idata = gaussian_plain_fitted.result.idata
        assert hasattr(idata, "log_likelihood"), \
            "idata must have log_likelihood (required for LOO/WAIC)"

    def test_extra_response(self, gaussian_plain_fitted):
        assert gaussian_plain_fitted.result.extra["response"] == "y"

    def test_extra_group(self, gaussian_plain_fitted):
        assert gaussian_plain_fitted.result.extra["group"] == "group"

    def test_extra_no_sampling_var(self, gaussian_plain_fitted):
        assert gaussian_plain_fitted.result.extra["sampling_var_col"] is None

    def test_fitted_at_set(self, gaussian_plain_fitted):
        assert gaussian_plain_fitted.result.fitted_at is not None

    def test_backend_model_is_bambi(self, gaussian_plain_fitted):
        import bambi as bmb
        assert isinstance(gaussian_plain_fitted.result.backend_model, bmb.Model)


class TestGaussianFayHerriott:

    def test_fh_is_fitted(self, gaussian_fh_fitted):
        assert gaussian_fh_fitted.is_fitted

    def test_fh_family(self, gaussian_fh_fitted):
        assert gaussian_fh_fitted.result.family == "gaussian"

    def test_fh_sampling_var_col_in_extra(self, gaussian_fh_fitted):
        assert gaussian_fh_fitted.result.extra["sampling_var_col"] == "D"

    def test_fh_data_has_log_sqrt_D(self, gaussian_fh_fitted):
        """DataPreprocessor must have added log_sqrt_D column."""
        df = gaussian_fh_fitted.result.data
        assert "log_sqrt_D" in df.columns

    def test_fh_log_sqrt_D_values(self, gaussian_fh_fitted):
        """log_sqrt_D = 0.5 * log(D) — spot check."""
        df = gaussian_fh_fitted.result.data
        D = df["D"].values
        expected = 0.5 * np.log(D)
        np.testing.assert_allclose(df["log_sqrt_D"].values, expected, rtol=1e-6)

    def test_fh_idata_log_likelihood(self, gaussian_fh_fitted):
        assert hasattr(gaussian_fh_fitted.result.idata, "log_likelihood")


class TestGaussianPredict:

    def test_predict_in_sample_shape(self, gaussian_plain_fitted):
        draws = gaussian_plain_fitted.predict()
        total = gaussian_plain_fitted.result.config.draws * \
                gaussian_plain_fitted.result.config.chains
        n_obs = len(gaussian_plain_fitted.result.data)
        assert draws.shape == (total, n_obs)

    def test_predict_out_of_sample_shape(self, gaussian_plain_fitted, data_gaussian):
        new_data = data_gaussian.tail(10).reset_index(drop=True)
        draws = gaussian_plain_fitted.predict(new_data=new_data)
        total = gaussian_plain_fitted.result.config.draws * \
                gaussian_plain_fitted.result.config.chains
        assert draws.shape == (total, 10)

    def test_predict_n_samples(self, gaussian_plain_fitted):
        draws = gaussian_plain_fitted.predict(n_samples=30)
        assert draws.shape[0] == 30

    def test_predict_before_fit_raises(self, data_gaussian):
        cfg = hb.ModelConfig(draws=100, tune=100, chains=2)
        m = hb.create_model("y ~ x1", family="gaussian", data=data_gaussian, config=cfg)
        with pytest.raises(hb.ModelNotFittedError):
            m.predict()

    def test_predict_finite(self, gaussian_plain_fitted):
        draws = gaussian_plain_fitted.predict()
        assert np.isfinite(draws).all()


# ===========================================================================
# LognormalModel tests
# ===========================================================================

class TestLognormalFit:

    def test_is_fitted(self, lognormal_plain_fitted):
        assert lognormal_plain_fitted.is_fitted

    def test_result_family(self, lognormal_plain_fitted):
        """ModelResult.family must be 'lognormal' even though Bambi uses Gaussian."""
        assert lognormal_plain_fitted.result.family == "lognormal"

    def test_extra_response(self, lognormal_plain_fitted):
        assert lognormal_plain_fitted.result.extra["response"] == "y_log"

    def test_idata_has_posterior(self, lognormal_plain_fitted):
        assert hasattr(lognormal_plain_fitted.result.idata, "posterior")

    def test_idata_has_log_likelihood(self, lognormal_plain_fitted):
        assert hasattr(lognormal_plain_fitted.result.idata, "log_likelihood")

    def test_backend_model_is_bambi(self, lognormal_plain_fitted):
        import bambi as bmb
        assert isinstance(lognormal_plain_fitted.result.backend_model, bmb.Model)

    def test_extra_no_sampling_var(self, lognormal_plain_fitted):
        assert lognormal_plain_fitted.result.extra["sampling_var_col"] is None


class TestLognormalFayHerriott:

    def test_fh_is_fitted(self, lognormal_fh_fitted):
        assert lognormal_fh_fitted.is_fitted

    def test_fh_sampling_var_col(self, lognormal_fh_fitted):
        assert lognormal_fh_fitted.result.extra["sampling_var_col"] == "psi"

    def test_fh_has_log_sqrt_D(self, lognormal_fh_fitted):
        df = lognormal_fh_fitted.result.data
        assert "log_sqrt_D" in df.columns

    def test_fh_log_sqrt_D_values(self, lognormal_fh_fitted):
        df = lognormal_fh_fitted.result.data
        psi = df["psi"].values
        np.testing.assert_allclose(df["log_sqrt_D"].values, 0.5 * np.log(psi), rtol=1e-6)


class TestLognormalPredict:

    def test_predict_shape(self, lognormal_plain_fitted):
        draws = lognormal_plain_fitted.predict()
        total = lognormal_plain_fitted.result.config.draws * \
                lognormal_plain_fitted.result.config.chains
        n_obs = len(lognormal_plain_fitted.result.data)
        assert draws.shape == (total, n_obs)

    def test_predict_finite(self, lognormal_plain_fitted):
        """Log-scale predictions can be any real number — just check finite."""
        draws = lognormal_plain_fitted.predict()
        assert np.isfinite(draws).all()

    def test_predict_before_fit_raises(self, data_lognormal_plain):
        cfg = hb.ModelConfig(draws=100, tune=100, chains=2)
        m = hb.create_model("y_log ~ x1", family="lognormal",
                             data=data_lognormal_plain, config=cfg)
        with pytest.raises(hb.ModelNotFittedError):
            m.predict()

    def test_predict_n_samples(self, lognormal_plain_fitted):
        draws = lognormal_plain_fitted.predict(n_samples=20)
        assert draws.shape[0] == 20


# ===========================================================================
# BinomialModel tests
# ===========================================================================

class TestBinomialFit:

    def test_is_fitted(self, binomial_fitted):
        assert binomial_fitted.is_fitted

    def test_result_family(self, binomial_fitted):
        assert binomial_fitted.result.family == "binomial"

    def test_result_formula(self, binomial_fitted):
        assert "y" in binomial_fitted.result.formula
        assert "x1" in binomial_fitted.result.formula

    def test_idata_has_posterior(self, binomial_fitted):
        assert hasattr(binomial_fitted.result.idata, "posterior")

    def test_idata_has_log_likelihood(self, binomial_fitted):
        assert hasattr(binomial_fitted.result.idata, "log_likelihood")

    def test_extra_response(self, binomial_fitted):
        assert binomial_fitted.result.extra["response"] == "y"

    def test_extra_trials_col(self, binomial_fitted):
        assert binomial_fitted.result.extra["trials_col"] == "n"

    def test_extra_group(self, binomial_fitted):
        assert binomial_fitted.result.extra["group"] == "group"

    def test_backend_model_is_bambi(self, binomial_fitted):
        import bambi as bmb
        assert isinstance(binomial_fitted.result.backend_model, bmb.Model)

    def test_fitted_at_set(self, binomial_fitted):
        assert binomial_fitted.result.fitted_at is not None


class TestBinomialPredict:

    def test_predict_in_sample_shape(self, binomial_fitted):
        draws = binomial_fitted.predict()
        total = binomial_fitted.result.config.draws * \
                binomial_fitted.result.config.chains
        n_obs = len(binomial_fitted.result.data)
        assert draws.shape == (total, n_obs)

    def test_predict_out_of_sample_shape(self, binomial_fitted, data_binomial):
        new_data = data_binomial.tail(10).reset_index(drop=True)
        draws = binomial_fitted.predict(new_data=new_data)
        total = binomial_fitted.result.config.draws * \
                binomial_fitted.result.config.chains
        assert draws.shape == (total, 10)

    def test_predict_n_samples(self, binomial_fitted):
        draws = binomial_fitted.predict(n_samples=40)
        assert draws.shape[0] == 40

    def test_predict_before_fit_raises(self, data_binomial):
        cfg = hb.ModelConfig(draws=100, tune=100, chains=2)
        m = hb.create_model("y ~ x1", family="binomial",
                             data=data_binomial, trials="n", config=cfg)
        with pytest.raises(hb.ModelNotFittedError):
            m.predict()

    def test_predict_kind_linear_removed(self, binomial_fitted):
        """kind='linear' was removed in Bambi 0.14+ — must raise ValueError."""
        with pytest.raises(ValueError, match="linear"):
            binomial_fitted.predict(kind="linear")

    def test_predict_kind_response_params_in_unit_interval(self, binomial_fitted):
        """kind='response_params' extracts idata.posterior['p'] — success prob in (0,1)."""
        draws = binomial_fitted.predict(kind="response_params")
        assert (draws >= 0).all() and (draws <= 1).all()
        assert np.isfinite(draws).all()


class TestBinomialMissingTrials:

    def test_create_model_without_trials_raises(self, data_binomial):
        """factory already guards this — confirm ValueError is raised."""
        cfg = hb.ModelConfig(draws=100, tune=100, chains=2)
        with pytest.raises(ValueError, match="trials"):
            hb.create_model("y ~ x1", family="binomial",
                            data=data_binomial, config=cfg)


# ===========================================================================
# Summary tests (all three models)
# ===========================================================================

class TestModelSummary:

    def test_gaussian_summary(self, gaussian_plain_fitted):
        s = gaussian_plain_fitted.summary()
        assert "gaussian" in s.lower()

    def test_lognormal_summary(self, lognormal_plain_fitted):
        s = lognormal_plain_fitted.summary()
        assert "lognormal" in s.lower()

    def test_binomial_summary(self, binomial_fitted):
        s = binomial_fitted.summary()
        assert "binomial" in s.lower()
