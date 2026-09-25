"""BaseModel build/fit seam — `prior_predictive_idata()` and `check_data()`.

These lock the contract the GUI's *Prior Predictive Check* and *Build Model*
buttons depend on (see docs/prd-integrasi-gui.md): a prior predictive must be
obtainable from an **unfitted** model, and must leave that model unfitted.

Run with:
    pytest -m slow tests/test_v1_prior_predictive.py -v

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

# Groups every prior predictive run must produce.
_PRIOR_GROUPS = {"prior", "prior_predictive", "observed_data"}

# Small: these tests exercise the build path, not sampling quality.
_DRAWS = 20


# ---------------------------------------------------------------------------
# Unfitted model builders — one per family (tier 1, the GUI's entry point)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gaussian_unfitted(data_gaussian: pd.DataFrame) -> hb.BaseModel:
    return hb.hbm_gaussian("y", ["x1", "x2"], data_gaussian, area_var="group")


@pytest.fixture(scope="module")
def beta_unfitted(data_beta: pd.DataFrame) -> hb.BaseModel:
    return hb.hbm_beta("y", ["x1", "x2"], data_beta,
                       n="n", deff="deff", area_var="group")


@pytest.fixture(scope="module")
def binomial_unfitted(data_binomial: pd.DataFrame) -> hb.BaseModel:
    return hb.hbm_binomial("y", ["x1", "x2"], data_binomial,
                           trials="n", area_var="group")


# ---------------------------------------------------------------------------
# prior_predictive_idata() — the seam
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture_name", [
    "gaussian_unfitted", "beta_unfitted", "binomial_unfitted",
])
def test_returns_prior_groups(fixture_name, request):
    """Every family yields prior + prior_predictive + observed_data."""
    model = request.getfixturevalue(fixture_name)
    idata = model.prior_predictive_idata(draws=_DRAWS, random_seed=42)

    # _idata_groups, never idata.groups() — ArviZ 1.1 DataTree compatibility.
    assert _PRIOR_GROUPS <= set(_idata_groups(idata))


def test_model_stays_unfitted(gaussian_unfitted: hb.BaseModel):
    """The seam must not leave the model looking fitted."""
    gaussian_unfitted.prior_predictive_idata(draws=_DRAWS, random_seed=42)

    assert gaussian_unfitted.is_fitted is False
    with pytest.raises(hb.ModelNotFittedError):
        _ = gaussian_unfitted.result


def test_fit_still_works_after_prior_check(data_gaussian: pd.DataFrame):
    """A prior check must not poison the subsequent fit().

    Guards against `_build_backend` leaving state behind — the refactor's
    core risk.
    """
    cfg = hb.ModelConfig(draws=100, tune=100, chains=2, cores=1, random_seed=42)
    model = hb.hbm_gaussian("y", ["x1"], data_gaussian,
                            area_var="group", config=cfg)

    model.prior_predictive_idata(draws=_DRAWS, random_seed=42)
    result = model.fit()

    assert result.is_fitted
    # log_likelihood is mandatory downstream (LOO) — prove fit() still emits it.
    assert "log_likelihood" in _idata_groups(result.idata)


def test_gaussian_fh_offset_builds_outside_fit(data_gaussian: pd.DataFrame):
    """The distributional FH formula must build without sampling.

    `sampling_var` triggers `sigma ~ 1 + offset(log_sqrt_D)` plus the pinned
    `sigma_Intercept` prior. If `_build_backend` skipped any of the preprocess
    or prior-workaround steps, Bambi would fail on the missing offset column.
    """
    df = data_gaussian.assign(D=np.abs(data_gaussian["y"]) * 0.1 + 0.05)
    model = hb.hbm_gaussian("y", ["x1"], df, sampling_var="D", area_var="group")

    idata = model.prior_predictive_idata(draws=_DRAWS, random_seed=42)

    assert _PRIOR_GROUPS <= set(_idata_groups(idata))


# ---------------------------------------------------------------------------
# Pinned parameters actually reach the backend
#
# These read the *prior*, which is where a pin either bites or does not — no
# sampling quality involved. They exist because the pin silently did nothing
# for a long time: the prior was sent under a flat `"sigma_Intercept"` key,
# which Bambi routes nowhere, so it kept its own Normal(0, 1) and the
# parameter was merely scaled by the offset instead of pinned to it. Every
# test that existed checked structure — the column, the formula, the extra
# dict — and none checked the value, so nothing caught it.
# ---------------------------------------------------------------------------

_PIN_SD = 1e-3


def _pinned_value(model: hb.BaseModel, param: str) -> float:
    """Reconstruct the pinned parameter from its prior and its offset column."""
    idata = model.prior_predictive_idata(draws=1000, random_seed=7)
    intercept = np.asarray(idata.prior[f"{param}_Intercept"].values)
    (fixed,) = [fp for fp in model._active_fixed_params() if fp.param == param]
    offset = float(model.check_data()[fixed.offset_col].iloc[0])
    # The sub-formula is `param ~ 1 + offset(col)` under a log link.
    return float(np.exp(offset + intercept.mean())), float(intercept.std())


def test_fh_pins_sigma_to_sqrt_sampling_variance(data_gaussian: pd.DataFrame):
    """Gaussian FH must fit `sigma = sqrt(D)`, not merely a multiple of it."""
    df = data_gaussian.assign(D=0.25)
    model = hb.hbm_gaussian("y", ["x1"], df, sampling_var="D", area_var="group")

    value, spread = _pinned_value(model, "sigma")

    assert spread == pytest.approx(_PIN_SD, rel=0.2), (
        "sigma_Intercept is not pinned — the prior did not reach Bambi."
    )
    assert value == pytest.approx(np.sqrt(0.25), rel=1e-3)


def test_beta_pins_kappa_to_precision(data_beta: pd.DataFrame):
    """Beta must fit `kappa = n/deff - 1`."""
    df = data_beta.assign(n=100.0, deff=2.0)
    model = hb.hbm_beta("y", ["x1"], df, n="n", deff="deff", area_var="group")

    value, spread = _pinned_value(model, "kappa")

    assert spread == pytest.approx(_PIN_SD, rel=0.2)
    assert value == pytest.approx(100.0 / 2.0 - 1.0, rel=1e-3)


@pytest.mark.parametrize("source,expected", [("sd_col", 2.0), (3.0, 3.0)])
def test_user_fixed_params_pin_the_value_given(
    data_gaussian: pd.DataFrame, source, expected
):
    """`fixed_params=` takes the parameter's own value — hbsaemp applies the link.

    Both accepted forms are covered: a column name and a bare scalar.
    """
    df = data_gaussian.assign(sd_col=2.0)
    model = hb.create_model(
        "y ~ x1 + (1|group)", family="gaussian", data=df,
        fixed_params={"sigma": source},
    )

    value, spread = _pinned_value(model, "sigma")

    assert spread == pytest.approx(_PIN_SD, rel=0.2)
    assert value == pytest.approx(expected, rel=1e-3)


def test_unpinned_family_leaves_the_parameter_free(data_gaussian: pd.DataFrame):
    """Without a pin, sigma keeps Bambi's own prior — the contrast case."""
    model = hb.hbm_gaussian("y", ["x1"], data_gaussian, area_var="group")

    assert model._active_fixed_params() == ()
    idata = model.prior_predictive_idata(draws=500, random_seed=7)
    assert "sigma_Intercept" not in idata.prior.data_vars
    assert float(np.asarray(idata.prior["sigma"].values).std()) > _PIN_SD * 100


def test_reproducible_with_seed(gaussian_unfitted: hb.BaseModel):
    """Same seed -> identical prior draws."""
    a = gaussian_unfitted.prior_predictive_idata(draws=_DRAWS, random_seed=123)
    b = gaussian_unfitted.prior_predictive_idata(draws=_DRAWS, random_seed=123)

    var = next(iter(a.prior.data_vars))
    np.testing.assert_allclose(
        np.asarray(a.prior[var].values), np.asarray(b.prior[var].values)
    )


def test_var_names_restricts_prior(gaussian_unfitted: hb.BaseModel):
    """`var_names` narrows what is sampled."""
    idata = gaussian_unfitted.prior_predictive_idata(
        draws=_DRAWS, var_names=["Intercept"], random_seed=42
    )

    assert "Intercept" in idata.prior.data_vars


def test_seed_falls_back_to_config(data_gaussian: pd.DataFrame):
    """`random_seed=None` uses `config.random_seed`."""
    cfg = hb.ModelConfig(draws=100, tune=100, chains=2, cores=1, random_seed=99)
    m1 = hb.hbm_gaussian("y", ["x1"], data_gaussian, config=cfg)
    m2 = hb.hbm_gaussian("y", ["x1"], data_gaussian, config=cfg)

    a = m1.prior_predictive_idata(draws=_DRAWS)
    b = m2.prior_predictive_idata(draws=_DRAWS)

    np.testing.assert_allclose(
        np.asarray(a.prior["Intercept"].values),
        np.asarray(b.prior["Intercept"].values),
    )
