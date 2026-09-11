"""update_model() argument handling — no MCMC.

`BaseModel.fit` is replaced by a stub that records the model it was called on
and returns a fitted-looking `ModelResult`, so the formula / prior / config /
data plumbing is checked without sampling.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

import hbsaemp as hb
from hbsaemp.models._base import BaseModel
from hbsaemp.utils._formula import update_formula

# ---------------------------------------------------------------------------
# update_formula()
# ---------------------------------------------------------------------------

_OLD = "y ~ x1 + x2 + (1|group)"


@pytest.mark.parametrize(
    ("template", "expected"),
    [
        (". ~ . + x3", "y ~ x1 + x2 + (1|group) + x3"),
        (". ~ . - x1", "y ~ x2 + (1|group)"),
        (". ~ . + x3 - x1", "y ~ x2 + (1|group) + x3"),
        (". ~ . - (1 | group)", "y ~ x1 + x2"),
        (". ~ . - 1", "y ~ 0 + x1 + x2 + (1|group)"),
        (". ~ . + x1", "y ~ x1 + x2 + (1|group)"),  # already present: no duplicate
        ("z ~ .", "z ~ x1 + x2 + (1|group)"),
        ("y ~ x3", "y ~ x3"),
    ],
)
def test_update_formula(template, expected):
    assert update_formula(_OLD, template) == expected


def test_update_formula_removing_every_term_leaves_intercept():
    assert update_formula("y ~ x1", ". ~ . - x1") == "y ~ 1"


@pytest.mark.parametrize("template", ["x3", ". ~ . + log(x3)"])
def test_update_formula_rejects_bad_template(template):
    with pytest.raises(hb.FormulaError):
        update_formula(_OLD, template)


# ---------------------------------------------------------------------------
# update_model() with a sampling-free fit
# ---------------------------------------------------------------------------

@pytest.fixture()
def stub_fit(monkeypatch) -> list[BaseModel]:
    """Replace `BaseModel.fit` with a stub; returns the list of models it fitted."""
    fitted_models: list[BaseModel] = []

    def fake_fit(self: BaseModel) -> hb.ModelResult:
        fitted_models.append(self)
        self._result = hb.ModelResult(
            formula=self._formula, family=self._family, data=self._data,
            config=self._config, priors=self._priors, is_fitted=True,
            extra={"response": self.response_name, "group": self._group},
        )
        return self._result

    monkeypatch.setattr(BaseModel, "fit", fake_fit)
    return fitted_models


@pytest.fixture()
def fh_data(data_gaussian: pd.DataFrame) -> pd.DataFrame:
    """Gaussian fixture plus a known sampling-variance column `D`."""
    df = data_gaussian.copy()
    df["D"] = np.linspace(0.1, 0.5, len(df))
    return df


@pytest.fixture()
def fitted(fh_data, default_config, stub_fit) -> BaseModel:
    """Fay-Herriot model (offset active) with user priors on x1 and x2."""
    m = hb.create_model(
        "y ~ x1 + x2", family="gaussian", data=fh_data, group="group",
        sampling_var="D", config=default_config,
        priors={"x1": hb.Prior("Normal", mu=0, sigma=1),
                "x2": hb.Prior("Normal", mu=0, sigma=2)},
    )
    m.fit()
    stub_fit.clear()
    return m


def test_formula_template_updates_model_formula(fitted):
    hb.update_model(fitted, formula=". ~ . + x3 - x1")
    assert fitted.formula == "y ~ x2 + (1|group) + x3"
    assert fitted.result.formula == fitted.formula


def test_added_term_read_from_current_data(fitted, stub_fit):
    """hbsaems' auto-fallback: without new_data, new terms come from the stored data."""
    hb.update_model(fitted, formula=". ~ . + x3")
    (refit,) = stub_fit
    assert refit.data is fitted.data
    assert "x3" in refit.data.columns


def test_group_kept_while_formula_models_it(fitted):
    hb.update_model(fitted, formula=". ~ . + x3")
    assert fitted._group == "group"


def test_removing_random_effect_removes_group(fitted):
    hb.update_model(fitted, formula=". ~ . - (1|group)")
    assert fitted._group is None
    assert "group" not in fitted.formula


def test_priors_merged_new_entries_win(fitted):
    replacement = hb.Prior("StudentT", nu=3, mu=0, sigma=1)
    hb.update_model(
        fitted, formula=". ~ . + x3",
        priors={"x2": replacement, "x3": hb.Prior("Normal", mu=0, sigma=5)},
    )
    assert fitted._priors["x2"] is replacement
    assert fitted._priors["x1"].params == {"mu": 0, "sigma": 1}  # carried over
    assert "x3" in fitted._priors


def test_prior_of_removed_term_dropped(fitted, caplog):
    with caplog.at_level(logging.WARNING, logger="hbsaemp.estimation.update"):
        hb.update_model(fitted, formula=". ~ . - x1")
    assert set(fitted._priors) == {"x2"}
    assert "dropping prior(s) for ['x1']" in caplog.text


def test_malformed_prior_rejected_before_refit(fitted, stub_fit):
    with pytest.raises(hb.PriorSpecError):
        hb.update_model(fitted, priors={"x1": {"mu": 0}})  # no "dist"
    assert stub_fit == []


def test_config_and_override_together_refused(fitted, default_config, stub_fit):
    with pytest.raises(ValueError, match="not both"):
        hb.update_model(fitted, config=default_config, draws=10)
    assert stub_fit == []


def test_new_sampler_overrides_reach_config(fitted):
    hb.update_model(fitted, max_treedepth=12, progressbar=False,
                    sampler_kwargs={"init": "adapt_diag"})
    assert fitted.config.max_treedepth == 12
    assert fitted.config.progressbar is False
    assert fitted.config.sampler_kwargs == {"init": "adapt_diag"}


def test_replacement_config_is_copied(fitted, default_config):
    hb.update_model(fitted, config=default_config)
    assert fitted.config == default_config
    assert fitted.config is not default_config


def test_missing_design_column_copied_when_rows_match(fitted, fh_data):
    new = fh_data.drop(columns=["D"])
    with pytest.warns(UserWarning, match=r"\['D'\]"):
        hb.update_model(fitted, new_data=new)
    np.testing.assert_array_equal(fitted.data["D"].to_numpy(), fh_data["D"].to_numpy())
    assert "D" not in new.columns  # the caller's frame is untouched


def test_missing_design_column_with_other_rows_raises(fitted, fh_data, stub_fit):
    with pytest.raises(hb.DataValidationError, match=r"\['D'\]"):
        hb.update_model(fitted, new_data=fh_data.head(20).drop(columns=["D"]))
    assert stub_fit == []


def test_failed_refit_leaves_model_unchanged(fitted, monkeypatch):
    result, formula, config, priors = (
        fitted.result, fitted.formula, fitted.config, fitted._priors
    )

    def failing_fit(self: BaseModel) -> hb.ModelResult:
        raise RuntimeError("sampler failed")

    monkeypatch.setattr(BaseModel, "fit", failing_fit)
    with pytest.raises(RuntimeError, match="sampler failed"):
        hb.update_model(fitted, formula=". ~ . + x3", draws=10)
    assert fitted.result is result
    assert fitted.formula == formula
    assert fitted.config is config
    assert fitted._priors is priors
