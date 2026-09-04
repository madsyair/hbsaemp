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
# Unfitted model builders — one per family (tier 3, the GUI's entry point)
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
