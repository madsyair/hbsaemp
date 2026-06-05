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
from hbsaemp.diagnostics._plot_utils import _idata_groups

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
        """log_likelihood group must exist — required for LOO (M4)."""
        idata = beta_model_fitted.result.idata
        assert hasattr(idata, "log_likelihood"), \
            "idata must have log_likelihood group (bmodel.compute_log_likelihood)"

    def test_log_likelihood_has_response_var(self, beta_model_fitted):
        """log_likelihood group must contain the response variable."""
        ll = beta_model_fitted.result.idata.log_likelihood
        response = beta_model_fitted.result.extra["response"]
        assert response in ll

    def test_log_likelihood_finite(self, beta_model_fitted):
        """All log-likelihood values must be finite."""
        import numpy as np
        ll = beta_model_fitted.result.idata.log_likelihood
        response = beta_model_fitted.result.extra["response"]
        assert np.isfinite(ll[response].values).all()

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
        """kind='linear' was removed in Bambi 0.18+ — must raise ValueError."""
        with pytest.raises(ValueError, match="linear"):
            beta_model_fitted.predict(kind="linear")

    def test_predict_kind_response_params_in_unit_interval(self, beta_model_fitted):
        """kind='response_params' returns posterior of mu (logit link → (0,1))."""
        draws = beta_model_fitted.predict(kind="response_params")
        assert (draws >= 0).all() and (draws <= 1).all()
        assert np.isfinite(draws).all()

    def test_predict_in_sample_value_unchanged(self, beta_model_fitted):
        """predict() in-sample is deterministic — inplace=False must not change
        the returned values, only whether idata is mutated."""
        out1 = beta_model_fitted.predict(kind="response_params")
        out2 = beta_model_fitted.predict(kind="response_params")
        np.testing.assert_allclose(out1, out2)

    def test_predict_new_data_does_not_mutate_result_idata(
        self, beta_model_fitted, data_beta
    ):
        """predict(new_data) must NOT mutate result.idata — inplace=False keeps
        the stored parameter posterior pristine (no extra vars appended)."""
        idata = beta_model_fitted.result.idata
        vars_before = set(idata.posterior.data_vars)
        val_before = float(idata.posterior["Intercept"].values.flatten()[0])

        new_data = data_beta.head(10).reset_index(drop=True)
        beta_model_fitted.predict(new_data=new_data, kind="response_params")

        assert set(idata.posterior.data_vars) == vars_before
        assert float(idata.posterior["Intercept"].values.flatten()[0]) == val_before

    def test_predictive_idata_no_mutation(self, beta_model_fitted):
        """Public predictive_idata() populates posterior_predictive on a fresh
        idata without mutating result.idata. Also exercises _idata_groups on a
        real ArviZ 1.1 DataTree (T2 validation)."""
        m = beta_model_fitted
        before = _idata_groups(m.result.idata)
        ppc = m.predictive_idata()
        assert "posterior_predictive" in _idata_groups(ppc)   # fresh idata filled
        assert _idata_groups(m.result.idata) == before        # stored idata untouched


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

        # Layer 1: hbsaemp stores the user's prior dict in ModelResult.priors.
        # Restored from git HEAD (dropped when migrating off bmb.Model.priors).
        # Use == not `is` — robust against future defensive-copy semantics.
        assert result.priors == priors

        # Layer 2: Bambi 0.18 removed `bmb.Model.priors` — priors now live on
        # the parent distributional component's terms. Walk the public path
        # (stable since Bambi 0.13).
        bmodel = result.backend_model
        parent_name = bmodel.family.likelihood.parent        # "mu" for Beta
        parent_terms = bmodel.distributional_components[parent_name].terms
        for name, spec in priors.items():
            assert name in parent_terms, (
                f"Custom prior for {name!r} not in parent_terms. "
                f"Available terms: {list(parent_terms.keys())}"
            )
            assert parent_terms[name].prior.name == spec["dist"], (
                f"Expected prior dist {spec['dist']!r} for {name!r}, "
                f"got {parent_terms[name].prior.name!r}"
            )


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
