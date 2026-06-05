"""GaussianModel and BinomialModel v1 tests — require Bambi MCMC.

Run with:
    pytest -m slow tests/test_v1_models.py -v

Skip with:
    pytest -m "not slow"

Coverage:
    - GaussianModel  (plain + Fay-Herriot with sampling_var)
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
            "idata must have log_likelihood (required for LOO)"

    def test_log_likelihood_has_response_var(self, gaussian_plain_fitted):
        ll = gaussian_plain_fitted.result.idata.log_likelihood
        response = gaussian_plain_fitted.result.extra["response"]
        assert response in ll

    def test_log_likelihood_finite(self, gaussian_plain_fitted):
        import numpy as np
        ll = gaussian_plain_fitted.result.idata.log_likelihood
        response = gaussian_plain_fitted.result.extra["response"]
        assert np.isfinite(ll[response].values).all()

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

    def test_log_likelihood_has_response_var(self, binomial_fitted):
        # Bambi stores the Binomial response under the wrapped LHS `p(y, n)`,
        # so match the response name as a substring of some log_likelihood var.
        ll = binomial_fitted.result.idata.log_likelihood
        response = binomial_fitted.result.extra["response"]
        assert any(response in str(v) for v in ll.data_vars)

    def test_log_likelihood_finite(self, binomial_fitted):
        import numpy as np
        ll = binomial_fitted.result.idata.log_likelihood
        assert all(np.isfinite(ll[v].values).all() for v in ll.data_vars)

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
        """kind='linear' was removed in Bambi 0.18+ — must raise ValueError."""
        with pytest.raises(ValueError, match="linear"):
            binomial_fitted.predict(kind="linear")

    def test_predict_kind_response_params_in_unit_interval(self, binomial_fitted):
        """kind='response_params' returns posterior of p (continuous), not counts.

        The (0, 1) range alone is not sufficient — posterior predictive counts
        normalised by trials would also pass. Verify draws are continuous floats
        so a swapped extraction key (counts vs probabilities) would fail.
        """
        draws = binomial_fitted.predict(kind="response_params")
        assert (draws >= 0).all() and (draws <= 1).all()
        assert np.isfinite(draws).all()
        # Counts would be integer-valued after / trials. Real probabilities
        # are continuous — at least one fractional value beyond {0, 1/n, ...}.
        assert not np.allclose(draws, np.round(draws)), (
            "response_params should return continuous probabilities, not rounded counts"
        )


class TestBinomialMissingTrials:

    def test_create_model_without_trials_raises(self, data_binomial):
        """factory already guards this — confirm ValueError is raised."""
        cfg = hb.ModelConfig(draws=100, tune=100, chains=2)
        with pytest.raises(ValueError, match="trials"):
            hb.create_model("y ~ x1", family="binomial",
                            data=data_binomial, config=cfg)


# ===========================================================================
# Summary tests
# ===========================================================================

class TestModelSummary:

    def test_gaussian_summary(self, gaussian_plain_fitted):
        s = gaussian_plain_fitted.summary()
        assert "gaussian" in s.lower()

    def test_binomial_summary(self, binomial_fitted):
        s = binomial_fitted.summary()
        assert "binomial" in s.lower()
