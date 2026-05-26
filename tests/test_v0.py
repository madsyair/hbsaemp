"""v0 smoke tests — run without Bambi, PyMC, or ArviZ.

All model/diagnostic/estimation stubs must raise NotImplementedError.
All result dataclasses must be constructable.
All validation (ModelConfig, AppConfig) must work.
"""
import warnings
import pytest
import numpy as np
import pandas as pd
import hbsaemp as hb
from hbsaemp.utils._formula import parse_formula
from hbsaemp._exceptions import FormulaError


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------

def test_version():
    assert hb.__version__ == "1.0.0"


def test_all_exports_present():
    expected = [
        "configure_logging", "ModelConfig", "DEFAULT_CONFIG",
        "BaseModel", "ModelResult",
        "create_model",       # primary Python name
        "hbm",                # R-style alias
        "MODEL_REGISTRY",
        "DataValidator", "DataPreprocessor", "load_dataset",
        "ConvergenceResult", "check_convergence", "hbcc",
        "PriorCheckResult", "check_prior", "hbpc",
        "ComparisonResult", "compare_models", "hbmc",
        "AreaEstimatesResult", "estimate_areas", "hbsae",
        "update_model", "update_hbm",
        "launch_app", "App", "AppConfig", "DEFAULT_APP_CONFIG",
    ]
    for name in expected:
        assert hasattr(hb, name), f"Missing: {name}"


def test_r_aliases_identity():
    assert hb.hbm is hb.create_model      # hbm = R alias for create_model
    assert hb.hbcc is hb.check_convergence
    assert hb.hbpc is hb.check_prior
    assert hb.hbmc is hb.compare_models
    assert hb.hbsae is hb.estimate_areas
    assert hb.update_hbm is hb.update_model


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------

def test_model_config_defaults():
    cfg = hb.ModelConfig()
    assert cfg.draws == 1000
    assert cfg.chains == 4
    assert cfg.total_draws == 4000


def test_model_config_custom():
    cfg = hb.ModelConfig(draws=2000, tune=500, chains=2, cores=2)
    assert cfg.total_draws == 4000
    kw = cfg.to_sampler_kwargs()
    assert kw["draws"] == 2000 and kw["chains"] == 2


@pytest.mark.parametrize("bad,exc", [
    ({"draws": 0}, ValueError),
    ({"tune": -1}, ValueError),
    ({"chains": 0}, ValueError),
    ({"target_accept": 1.5}, ValueError),
    ({"sample_prior": "yes"}, ValueError),
])
def test_model_config_validation(bad, exc):
    with pytest.raises(exc):
        hb.ModelConfig(**bad)


# ---------------------------------------------------------------------------
# create_model() factory  (hbm is the R-style alias)
# ---------------------------------------------------------------------------

@pytest.fixture
def small_df(rng):
    n = 30
    return pd.DataFrame({
        "y": rng.normal(0, 1, n),
        "y_beta": np.clip(rng.beta(2, 5, n), 0.01, 0.99),
        "y_binom": rng.binomial(20, 0.3, n).astype(int),
        "y_pos": np.exp(rng.normal(0, 0.5, n)),
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(0, 1, n),
        "group": np.repeat(range(3), 10),
        "n": np.full(n, 20),
        "deff": rng.uniform(1, 2, n),
    })


@pytest.fixture
def cfg():
    return hb.ModelConfig(draws=100, chains=2)


def test_create_model_all_families(small_df, cfg):
    m1 = hb.create_model("y ~ x1",       family="gaussian",  data=small_df, group="group", config=cfg)
    m2 = hb.create_model("y_beta ~ x1",  family="beta",       data=small_df, n="n", deff="deff", config=cfg)
    m3 = hb.create_model("y_binom ~ x1", family="binomial",   data=small_df, trials="n", config=cfg)
    m4 = hb.create_model("y_pos ~ x1",   family="lognormal",  data=small_df, group="group", config=cfg)
    assert type(m1).__name__ == "GaussianModel"
    assert type(m2).__name__ == "BetaModel"
    assert type(m3).__name__ == "BinomialModel"
    assert type(m4).__name__ == "LognormalModel"


def test_hbm_alias_identical_to_create_model(small_df, cfg):
    """hbm() must produce the same result as create_model() — it is the same object."""
    m_py = hb.create_model("y ~ x1", family="gaussian", data=small_df, config=cfg)
    m_r  = hb.hbm("y ~ x1", family="gaussian", data=small_df, config=cfg)
    assert type(m_py) is type(m_r)
    assert m_py.formula == m_r.formula
    assert m_py.family == m_r.family


def test_create_model_auto_re_inject(small_df, cfg):
    m = hb.create_model("y ~ x1", family="gaussian", data=small_df, group="group", config=cfg)
    assert "(1|group)" in m.formula


def test_create_model_no_double_re(small_df, cfg):
    m = hb.create_model("y ~ x1 + (1|group)", family="gaussian", data=small_df,
                group="group", config=cfg)
    assert m.formula.count("(1|group)") == 1


def test_create_model_unknown_family(small_df, cfg):
    with pytest.raises(hb.ModelRegistryError) as exc_info:
        hb.create_model("y ~ x1", family="tweedie", data=small_df, config=cfg)
    assert exc_info.value.family == "tweedie"
    assert "gaussian" in exc_info.value.registered


def test_create_model_binomial_requires_trials(small_df, cfg):
    with pytest.raises(ValueError, match="trials"):
        hb.create_model("y_binom ~ x1", family="binomial", data=small_df, config=cfg)


def test_create_model_beta_n_deff_together(small_df, cfg):
    with pytest.raises(ValueError):
        hb.create_model("y_beta ~ x1", family="beta", data=small_df, n="n", config=cfg)
    with pytest.raises(ValueError):
        hb.create_model("y_beta ~ x1", family="beta", data=small_df, deff="deff", config=cfg)


def test_create_model_bad_data_type(cfg):
    with pytest.raises(TypeError):
        hb.create_model("y ~ x1", family="gaussian", data="not_a_df", config=cfg)


# ---------------------------------------------------------------------------
# P0-2 — cross-family argument validation
# ---------------------------------------------------------------------------
# `_FAMILY_PARAMS` in _factory.py defines which user-facing kwargs each
# family accepts. Anything else must be rejected by create_model() with a
# ValueError before the model is constructed — so users do not get a
# silently misconfigured model from a typo'd or mis-routed argument.

class TestCreateModelCrossFamilyValidation:

    def test_gaussian_rejects_trials(self, small_df, cfg):
        with pytest.raises(ValueError, match="not valid for family='gaussian'"):
            hb.create_model("y ~ x1", family="gaussian", data=small_df,
                            trials="n", config=cfg)

    def test_lognormal_rejects_n(self, small_df, cfg):
        with pytest.raises(ValueError, match="not valid for family='lognormal'"):
            hb.create_model("y_pos ~ x1", family="lognormal", data=small_df,
                            n="n", config=cfg)

    def test_beta_rejects_sampling_var(self, small_df, cfg):
        with pytest.raises(ValueError, match="not valid for family='beta'"):
            hb.create_model("y_beta ~ x1", family="beta", data=small_df,
                            n="n", deff="deff", sampling_var="deff", config=cfg)

    def test_binomial_rejects_n_deff(self, small_df, cfg):
        with pytest.raises(ValueError) as exc_info:
            hb.create_model("y_binom ~ x1", family="binomial", data=small_df,
                            trials="n", n="n", deff="deff", config=cfg)
        msg = str(exc_info.value)
        assert "not valid for family='binomial'" in msg
        assert "'deff'" in msg and "'n'" in msg

    @pytest.mark.parametrize("family,extra", [
        ("gaussian",  {"sampling_var": "deff"}),
        ("lognormal", {}),
        ("binomial",  {"trials": "n"}),
    ])
    def test_non_beta_rejects_squeeze_true(self, family, extra, small_df, cfg):
        formula = {"gaussian": "y ~ x1", "lognormal": "y_pos ~ x1",
                   "binomial": "y_binom ~ x1"}[family]
        with pytest.raises(ValueError, match="squeeze"):
            hb.create_model(formula, family=family, data=small_df,
                            squeeze=True, config=cfg, **extra)

    def test_non_beta_allows_default_squeeze_false(self, small_df, cfg):
        # Explicit squeeze=False on a non-beta family must NOT regress —
        # False is the default-safe value and must not trip the validator.
        m = hb.create_model("y ~ x1", family="gaussian", data=small_df,
                            squeeze=False, config=cfg)
        assert type(m).__name__ == "GaussianModel"

    def test_squeeze_non_bool_type_error(self, small_df, cfg):
        with pytest.raises(TypeError, match="squeeze"):
            hb.create_model("y_beta ~ x1", family="beta", data=small_df,
                            n="n", deff="deff", squeeze="yes", config=cfg)


def test_create_model_rejects_unknown_kwargs(small_df, cfg):
    """Unknown / mistyped kwargs must fail fast (no silent _kwargs sink).

    Regression test for P0-1: BaseModel used to swallow unknown kwargs into
    self._kwargs, which both hid typos and caused duplicate-kwarg TypeError
    on subsequent update_model() calls. With _FAMILY_PARAMS dispatch in
    _factory.py and **kwargs removed from BaseModel.__init__, unknown
    kwargs must raise TypeError at construction time.
    """
    with pytest.raises(TypeError):
        hb.create_model(
            "y ~ x1", family="gaussian", data=small_df,
            config=cfg, samplng_var="D",  # typo for sampling_var
        )


# ---------------------------------------------------------------------------
# Model state
# ---------------------------------------------------------------------------

def test_model_not_fitted(small_df, cfg):
    m = hb.create_model("y ~ x1", family="gaussian", data=small_df, config=cfg)
    assert not m.is_fitted
    with pytest.raises(hb.ModelNotFittedError):
        _ = m.result


def test_model_summary_before_fit(small_df, cfg):
    m = hb.create_model("y ~ x1", family="gaussian", data=small_df, config=cfg)
    s = m.summary()
    assert "not fitted" in s


def test_all_stubs_raise(small_df, cfg):
    """All backend stubs are now concrete (M1–M4 done).

    Previous stubs and their status:
    - model.fit() / predict()        → concrete M1/M2
    - estimate_areas() / update_model() → concrete M3, raise ModelNotFittedError
    - check_convergence() / compare_models() → concrete M4, raise ModelNotFittedError
    - check_prior                    → frontend scope, not tested here

    This test is now a no-op placeholder kept for history.
    """
    m = hb.create_model("y ~ x1", family="gaussian", data=small_df, config=cfg)
    assert not m.is_fitted  # model exists but is not yet fitted


# ---------------------------------------------------------------------------
# Data layer — concrete v1 (no Bambi required)
# ---------------------------------------------------------------------------

class TestDataLayer:
    """DataValidator, DataPreprocessor, and load_dataset are concrete in v1."""

    # ── DataValidator ────────────────────────────────────────────────────────

    def test_validator_gaussian_passes(self, data_gaussian):
        val = hb.DataValidator()
        # Read-only: returns None and does not mutate the input.
        before = data_gaussian.copy()
        assert val.validate(data_gaussian, "y", ["x1", "x2"], family="gaussian") is None
        pd.testing.assert_frame_equal(data_gaussian, before)

    def test_validator_gaussian_fh_passes(self, data_gaussian):
        val = hb.DataValidator()
        assert val.validate(
            data_gaussian, "y", ["x1"], family="gaussian",
            sampling_var_col=None,  # no FH column in this fixture
        ) is None

    def test_validator_beta_passes(self, data_beta):
        val = hb.DataValidator()
        assert val.validate(
            data_beta, "y", ["x1", "x2"], family="beta",
            n_col="n", deff_col="deff",
        ) is None

    def test_validator_beta_out_of_range(self, data_beta):
        bad = data_beta.copy()
        bad.loc[0, "y"] = 1.5  # outside (0, 1)
        with pytest.raises(hb.DataValidationError):
            hb.DataValidator().validate(bad, "y", ["x1"], family="beta")

    def test_validator_beta_boundary_rejected_without_squeeze(self, data_beta):
        """y=0 or y=1 is rejected when squeeze=False (default)."""
        bad = data_beta.copy()
        bad.loc[0, "y"] = 0.0
        with pytest.raises(hb.DataValidationError, match=r"\(0, 1\)"):
            hb.DataValidator().validate(bad, "y", ["x1"], family="beta")

    def test_validator_beta_boundary_accepted_with_squeeze(self, data_beta):
        """y=0 or y=1 is accepted when squeeze=True (closed interval [0,1])."""
        boundary = data_beta.copy()
        boundary.loc[0, "y"] = 0.0
        boundary.loc[1, "y"] = 1.0
        # Should not raise — boundary values allowed when squeeze=True
        assert hb.DataValidator().validate(
            boundary, "y", ["x1"], family="beta", squeeze=True
        ) is None

    def test_validator_beta_phi_nonpositive(self, data_beta):
        """n/deff <= 1 means phi <= 0 — must raise."""
        bad = data_beta.copy()
        bad.loc[0, "deff"] = bad.loc[0, "n"] * 2  # n/deff < 1
        with pytest.raises(hb.DataValidationError):
            hb.DataValidator().validate(
                bad, "y", ["x1"], family="beta", n_col="n", deff_col="deff"
            )

    def test_validator_lognormal_nonpositive(self, data_lognormal):
        bad = data_lognormal.copy()
        bad.loc[0, "y"] = -1.0
        with pytest.raises(hb.DataValidationError):
            hb.DataValidator().validate(bad, "y", ["x1"], family="lognormal")

    def test_validator_binomial_y_exceeds_n(self, data_binomial):
        bad = data_binomial.copy()
        bad.loc[0, "y"] = bad.loc[0, "n"] + 10
        with pytest.raises(hb.DataValidationError):
            hb.DataValidator().validate(
                bad, "y", ["x1"], family="binomial", trials_col="n"
            )

    def test_validator_missing_column(self, data_gaussian):
        with pytest.raises(hb.DataValidationError):
            hb.DataValidator().validate(
                data_gaussian, "y", ["nonexistent_col"], family="gaussian"
            )

    def test_validator_non_numeric(self, data_gaussian):
        bad = data_gaussian.copy()
        bad["x1"] = "text"
        with pytest.raises(hb.DataValidationError):
            hb.DataValidator().validate(bad, "y", ["x1"], family="gaussian")

    def test_validator_inf_in_response_rejected(self, data_gaussian):
        """np.inf in response must be rejected — MCMC gradients would become NaN."""
        bad = data_gaussian.copy()
        bad.loc[0, "y"] = np.inf
        with pytest.raises(hb.DataValidationError, match="Infinite"):
            hb.DataValidator().validate(bad, "y", ["x1"], family="gaussian")

    def test_validator_neg_inf_in_predictor_rejected(self, data_gaussian):
        """-np.inf in a predictor must be rejected."""
        bad = data_gaussian.copy()
        bad.loc[0, "x1"] = -np.inf
        with pytest.raises(hb.DataValidationError, match="Infinite"):
            hb.DataValidator().validate(bad, "y", ["x1"], family="gaussian")

    def test_validator_empty_dataframe(self):
        with pytest.raises(hb.DataValidationError):
            hb.DataValidator().validate(pd.DataFrame(), "y", [], family="gaussian")

    def test_validator_unknown_family(self, data_gaussian):
        with pytest.raises(hb.DataValidationError):
            hb.DataValidator().validate(data_gaussian, "y", [], family="tweedie")

    # ── DataPreprocessor ─────────────────────────────────────────────────────

    def test_preprocessor_beta_default_no_squeeze(self, data_beta):
        """Default (squeeze=False): y unchanged, log_phi added when n_col/deff_col present."""
        proc = hb.DataPreprocessor()
        original_y = data_beta["y"].values.copy()
        df_clean = proc.process(
            data_beta, "y", ["x1", "x2"], group="group",
            family="beta", n_col="n", deff_col="deff",
        )
        # y must NOT be modified (no squeeze by default)
        np.testing.assert_array_almost_equal(df_clean["y"].values, original_y)
        # log_phi column must exist and be finite
        assert "log_phi" in df_clean.columns
        assert np.isfinite(df_clean["log_phi"]).all()

    def test_preprocessor_beta_squeeze_explicit(self, data_beta):
        """squeeze=True: y is transformed by Smithson-Verkuilen, log_phi added."""
        proc = hb.DataPreprocessor()
        original_y = data_beta["y"].values.copy()
        df_clean = proc.process(
            data_beta, "y", ["x1"], family="beta",
            n_col="n", deff_col="deff", squeeze=True,
        )
        # y MUST differ from original (squeeze applied)
        assert not np.allclose(df_clean["y"].values, original_y)
        # squeezed values must still be in (0, 1)
        assert (df_clean["y"] > 0).all() and (df_clean["y"] < 1).all()
        assert "log_phi" in df_clean.columns

    def test_preprocessor_beta_no_squeeze_without_n_deff(self, data_beta):
        """Without n_col/deff_col, response is left unchanged, no log_phi."""
        proc = hb.DataPreprocessor()
        df_clean = proc.process(data_beta, "y", ["x1"], family="beta")
        assert "log_phi" not in df_clean.columns
        # y values should be identical to input (no squeeze)
        np.testing.assert_array_equal(
            df_clean["y"].values, data_beta.loc[df_clean.index, "y"].values
        )

    def test_preprocessor_gaussian_log_sqrt_D(self):
        """Gaussian FH: log_sqrt_D column is added when sampling_var_col is given."""
        rng = np.random.default_rng(0)
        m = 20
        df = pd.DataFrame({
            "y": rng.normal(5, 1, m),
            "x1": rng.normal(0, 1, m),
            "D": rng.uniform(0.1, 0.5, m),
        })
        proc = hb.DataPreprocessor()
        df_clean = proc.process(df, "y", ["x1"], family="gaussian", sampling_var_col="D")
        assert "log_sqrt_D" in df_clean.columns
        np.testing.assert_allclose(df_clean["log_sqrt_D"], 0.5 * np.log(df["D"]))

    def test_preprocessor_drops_nan(self, data_with_missing):
        proc = hb.DataPreprocessor()
        df_clean = proc.process(
            data_with_missing, "y", ["x1"], family="gaussian"
        )
        assert df_clean["y"].isna().sum() == 0
        assert df_clean["x1"].isna().sum() == 0
        assert len(df_clean) < len(data_with_missing)

    def test_preprocessor_all_nan_raises(self):
        """DataFrame where ALL rows have NaN in response must raise DataValidationError."""
        all_nan = pd.DataFrame({"y": [np.nan] * 5, "x1": [1.0] * 5})
        with pytest.raises(hb.DataValidationError, match="All"):
            hb.DataPreprocessor().process(all_nan, "y", ["x1"], family="gaussian")

    def test_preprocessor_returns_copy(self, data_gaussian):
        proc = hb.DataPreprocessor()
        df_clean = proc.process(data_gaussian, "y", ["x1"], family="gaussian")
        assert df_clean is not data_gaussian

    def test_preprocessor_unsupported_missing(self):
        with pytest.raises(NotImplementedError):
            hb.DataPreprocessor(handle_missing="model")

    # ── load_dataset ─────────────────────────────────────────────────────────

    def test_load_dataset_all_four(self):
        for name in hb.AVAILABLE_DATASETS:
            df = hb.load_dataset(name)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 30

    def test_load_dataset_fhnorm_columns(self):
        df = hb.load_dataset("data_fhnorm")
        expected = {"y", "D", "x1", "x2", "x3", "theta_true", "u", "group", "sre"}
        assert expected.issubset(set(df.columns))

    def test_load_dataset_betalogitnorm_columns(self):
        df = hb.load_dataset("data_betalogitnorm")
        expected = {"y", "theta", "x1", "x2", "x3", "n", "deff", "group", "sre"}
        assert expected.issubset(set(df.columns))
        assert (df["y"] > 0).all() and (df["y"] < 1).all()

    def test_load_dataset_binlogitnorm_columns(self):
        df = hb.load_dataset("data_binlogitnorm")
        expected = {
            "n", "y", "p", "x1", "x2", "x3",
            "u_true", "eta_true", "p_true", "psi_i",
            "y_obs", "p_obs", "group", "sre",
        }
        assert expected.issubset(set(df.columns))
        assert (df["y"] >= 0).all()
        assert (df["y"] <= df["n"]).all()

    def test_load_dataset_lnln_columns(self):
        df = hb.load_dataset("data_lnln")
        expected = {
            "group", "x1", "x2", "x3", "u_true", "theta_true",
            "mu_orig_true", "n", "y_obs", "lambda_dir", "y_log_obs",
            "psi_i", "sre",
        }
        assert expected.issubset(set(df.columns))
        assert (df["y_obs"] > 0).all()

    def test_load_dataset_reproducible(self):
        df1 = hb.load_dataset("data_fhnorm")
        df2 = hb.load_dataset("data_fhnorm")
        pd.testing.assert_frame_equal(df1, df2)

    def test_load_dataset_unknown(self):
        with pytest.raises(ValueError):
            hb.load_dataset("nonexistent_dataset")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

def test_result_dataclasses_constructable():
    cr = hb.ConvergenceResult()
    assert cr.rhat_ess is None and cr.plots == {}

    pr = hb.PriorCheckResult()
    assert pr.prior_predictive_plot is None

    comp = hb.ComparisonResult()
    assert comp.loo is None and comp.waic is None

    ae = hb.AreaEstimatesResult()
    assert ae.result_table.empty
    assert "not fitted" in ae.summary().lower() or "stub" in ae.summary().lower()

    mr = hb.ModelResult(formula="y~x", family="gaussian")
    assert not mr.is_fitted
    assert "not fitted" in mr.summary()


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

def test_exception_hierarchy():
    e = hb.DataValidationError("y has NaN", column="y", context={"n_nan": 3})
    assert isinstance(e, hb.ValidationError)
    assert isinstance(e, hb.HBSAEError)
    assert e.column == "y"
    assert e.context["n_nan"] == 3


def test_convergence_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warnings.warn("Rhat=1.05", hb.ConvergenceWarning)
    assert len(w) == 1
    assert issubclass(w[0].category, hb.ConvergenceWarning)


def test_model_registry_error():
    err = hb.ModelRegistryError("bad family", family="foo", registered=["gaussian"])
    assert err.family == "foo"
    assert "gaussian" in err.registered


# ---------------------------------------------------------------------------
# AppConfig and GUI
# ---------------------------------------------------------------------------

def test_app_config_defaults():
    cfg = hb.DEFAULT_APP_CONFIG
    assert cfg.port == 8080
    assert cfg.open_browser is True


def test_app_config_custom():
    cfg = hb.AppConfig(port=9000, title="Test", open_browser=False)
    assert cfg.port == 9000


@pytest.mark.parametrize("bad", [
    {"port": 0}, {"port": 70000}, {"max_upload_mb": 0}, {"log_level": "VERBOSE"},
])
def test_app_config_validation(bad):
    with pytest.raises(ValueError):
        hb.AppConfig(**bad)


def test_app_no_panel_needed():
    app = hb.App(app_config=hb.AppConfig(port=9999, open_browser=False))
    assert not app.state["data"]
    assert not app.state["model"]


def test_launch_app_raises_without_panel():
    with pytest.raises(NotImplementedError):
        hb.launch_app(open_browser=False)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def test_available_datasets():
    assert "data_fhnorm" in hb.AVAILABLE_DATASETS
    assert len(hb.AVAILABLE_DATASETS) == 4


def test_load_dataset_unknown():
    with pytest.raises(ValueError):
        hb.load_dataset("nonexistent_dataset")


# ---------------------------------------------------------------------------
# MODEL_REGISTRY
# ---------------------------------------------------------------------------

def test_registry_contents():
    assert set(hb.MODEL_REGISTRY.keys()) == {"gaussian", "beta", "binomial", "lognormal"}
    for cls in hb.MODEL_REGISTRY.values():
        assert issubclass(cls, hb.BaseModel)


# ---------------------------------------------------------------------------
# parse_formula
# ---------------------------------------------------------------------------

def test_parse_formula_basic():
    r = parse_formula("y ~ x1 + x2 + (1|area)")
    assert r["response"] == "y"
    assert r["fixed"] == ["x1", "x2"]
    assert r["random_groups"] == ["area"]


def test_parse_formula_no_re():
    r = parse_formula("y ~ x1 + x2")
    assert r["response"] == "y"
    assert r["fixed"] == ["x1", "x2"]
    assert r["random_groups"] == []


def test_parse_formula_whitespace_re():
    """(1 | group) with space around | must be parsed correctly — Bug 1 fix."""
    r = parse_formula("y ~ x1 + (1 | area)")
    assert r["random_groups"] == ["area"]
    assert r["fixed"] == ["x1"]


def test_parse_formula_whitespace_re_extra_spaces():
    """Multiple spaces around | still captured."""
    r = parse_formula("y ~ x1 + (1  |  area )")
    assert r["random_groups"] == ["area"]


def test_parse_formula_suppress_intercept():
    """0 token (no intercept) must be excluded from fixed, not raise."""
    r = parse_formula("y ~ 0 + x1 + x2")
    assert r["fixed"] == ["x1", "x2"]
    assert "0" not in r["fixed"]


def test_parse_formula_explicit_intercept():
    """1 token (explicit intercept) excluded from fixed, not raise."""
    r = parse_formula("y ~ 1 + x1")
    assert r["fixed"] == ["x1"]


def test_parse_formula_binomial_lhs():
    """Bambi binomial LHS 'y | trials(n) ~ ...' — response is 'y'."""
    r = parse_formula("y | trials(n) ~ x1 + (1|g)")
    assert r["response"] == "y"
    assert r["fixed"] == ["x1"]
    assert r["random_groups"] == ["g"]


def test_parse_formula_function_token_raises():
    """np.log(x) leaves 'np.log' after paren removal → FormulaError, not silent drop."""
    with pytest.raises(FormulaError, match="not a valid column identifier"):
        parse_formula("y ~ np.log(x1) + x2")


def test_parse_formula_interaction_raises():
    """'x1:x2' interaction token → FormulaError; user must pre-compute."""
    with pytest.raises(FormulaError, match="not a valid column identifier"):
        parse_formula("y ~ x1:x2 + x3")


def test_parse_formula_missing_tilde_raises():
    with pytest.raises(FormulaError):
        parse_formula("y x1 + x2")


def test_parse_formula_error_carries_formula():
    """FormulaError.context must include the offending formula string."""
    with pytest.raises(FormulaError) as exc_info:
        parse_formula("y ~ np.log(x1)")
    assert "np.log" in exc_info.value.context.get("formula", "")
