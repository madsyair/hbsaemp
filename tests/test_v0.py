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


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------

def test_version():
    assert hb.__version__ == "0.0.0"


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
    m = hb.create_model("y ~ x1", family="gaussian", data=small_df, config=cfg)
    stubs = [
        lambda: m.fit(),
        lambda: m.predict(),
        lambda: hb.check_convergence(m),
        lambda: hb.check_prior(m),
        lambda: hb.compare_models(m),
        lambda: hb.estimate_areas(m),
        lambda: hb.update_model(m),
        lambda: hb.DataValidator().validate(small_df, "y", ["x1"], family="gaussian", group=None),
        lambda: hb.DataPreprocessor().process(small_df, "y", ["x1"]),
        lambda: hb.load_dataset("data_fhnorm"),
    ]
    for fn in stubs:
        with pytest.raises(NotImplementedError):
            fn()


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
