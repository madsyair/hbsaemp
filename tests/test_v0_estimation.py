"""Bambi-free checks for estimate_areas() inputs and the mean-prediction frame.

No MCMC: the fitted state is faked with a `ModelResult` built from
`check_data()`, which is exactly the frame `fit()` would store.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import hbsaemp as hb


def _fake_fit(model: hb.BaseModel) -> hb.BaseModel:
    """Attach a fitted-looking result without sampling."""
    model._result = hb.ModelResult(
        family=model.family,
        data=model.check_data(),
        is_fitted=True,
        extra={
            "response": model.response_name,
            "group": "group",
            **model._extra_result_dict(),
        },
    )
    return model


@pytest.fixture()
def beta_faked(data_beta: pd.DataFrame, default_config) -> hb.BaseModel:
    """Beta model with the precision offset (`log_phi`) active."""
    return _fake_fit(hb.create_model(
        "y ~ x1 + x2 + (1|group)", family="beta", data=data_beta,
        n="n", deff="deff", config=default_config,
    ))


# ---------------------------------------------------------------------------
# estimate_areas() argument guard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ci_prob", [0.0, 1.0, 1.5, -0.1, math.nan])
def test_ci_prob_out_of_range_raises(data_beta, default_config, ci_prob):
    """Rejected before the fitted-model guard, so no fit is needed."""
    m = hb.create_model("y ~ x1", family="beta", data=data_beta,
                        n="n", deff="deff", config=default_config)
    with pytest.raises(ValueError, match="ci_prob"):
        hb.estimate_areas(m, ci_prob=ci_prob)


# ---------------------------------------------------------------------------
# AreaEstimatesResult.summary()
# ---------------------------------------------------------------------------

def test_summary_empty_says_not_fitted():
    assert "not fitted" in hb.AreaEstimatesResult().summary()


def test_summary_tolerates_missing_aggregates():
    table = pd.DataFrame({"mean": [0.5], "sd": [0.1]})
    res = hb.AreaEstimatesResult(result_table=table, mean_rse=None, mean_mse=math.nan)
    assert res.summary().count("n/a") == 2


# ---------------------------------------------------------------------------
# BaseModel._prepare_mean_data() — frame used for out-of-sample mu/p
# ---------------------------------------------------------------------------

def test_prepare_mean_data_needs_no_response_or_design_columns(beta_faked, data_beta):
    """A non-sampled area has no y and no survey design; mu needs neither."""
    nd = data_beta.tail(4).drop(columns=["y", "n", "deff"]).reset_index(drop=True)
    nd["group"] = [901, 902, 903, 904]
    out = beta_faked._prepare_mean_data(nd)
    assert len(out) == 4
    assert out["group"].tolist() == [901, 902, 903, 904]
    # The kappa sub-formula still evaluates the offset, so the column exists.
    assert (out["log_phi"] == 0.0).all()


def test_prepare_mean_data_keeps_rows_with_missing_response(beta_faked, data_beta):
    nd = data_beta.tail(3).reset_index(drop=True)
    nd["y"] = np.nan
    assert len(beta_faked._prepare_mean_data(nd)) == 3


def test_prepare_mean_data_drops_rows_with_missing_predictor(beta_faked, data_beta):
    nd = data_beta.tail(3).reset_index(drop=True)
    nd.loc[1, "x1"] = np.nan
    out = beta_faked._prepare_mean_data(nd)
    assert len(out) == 2
    assert out.index.tolist() == [0, 1]


@pytest.mark.parametrize("col", ["x1", "group"])
def test_prepare_mean_data_missing_required_column_raises(beta_faked, data_beta, col):
    nd = data_beta.tail(3).drop(columns=[col])
    with pytest.raises(hb.DataValidationError, match=col) as exc_info:
        beta_faked._prepare_mean_data(nd)
    assert exc_info.value.column == col


def test_prepare_mean_data_all_rows_incomplete_raises(beta_faked, data_beta):
    nd = data_beta.tail(2).reset_index(drop=True)
    nd["x2"] = np.nan
    with pytest.raises(hb.DataValidationError):
        beta_faked._prepare_mean_data(nd)


def test_prepare_mean_data_does_not_mutate_input(beta_faked, data_beta):
    nd = data_beta.tail(3).drop(columns=["y"]).reset_index(drop=True)
    before = nd.copy()
    beta_faked._prepare_mean_data(nd)
    pd.testing.assert_frame_equal(nd, before)


def test_prepare_mean_data_adds_no_offset_when_inactive(data_gaussian, default_config):
    """Gaussian without sampling_var has no sigma offset to satisfy."""
    m = _fake_fit(hb.create_model(
        "y ~ x1 + (1|group)", family="gaussian", data=data_gaussian,
        config=default_config,
    ))
    out = m._prepare_mean_data(data_gaussian.tail(2).drop(columns=["y"]))
    assert "log_sqrt_D" not in out.columns


@pytest.mark.parametrize("source", ["sd_col", 3.0])
def test_prepare_mean_data_adds_the_caller_pin_offset(
    data_gaussian, default_config, source
):
    """A pin from `fixed_params=` needs its offset column just like a family's.

    Regression guard. This frame satisfied only the family's own offset column
    (`log_sqrt_D`), never `hbsaemp_<par>_fixed`, so every out-of-sample mean
    prediction on a model built with `fixed_params=` died on
    `KeyError: 'hbsaemp_sigma_fixed'` — `predict(kind="response_params")` and
    `estimate_areas(new_data=...)` alike, which is the non-sampled-area
    workflow SAE exists for.
    """
    df = data_gaussian.assign(sd_col=2.0)
    m = _fake_fit(hb.create_model(
        "y ~ x1 + (1|group)", family="gaussian", data=df,
        config=default_config, fixed_params={"sigma": source},
    ))

    # A non-sampled area carries neither the response nor the pin column.
    out = m._prepare_mean_data(df.tail(2).drop(columns=["y", "sd_col"]))

    assert (out["hbsaemp_sigma_fixed"] == 0.0).all()
