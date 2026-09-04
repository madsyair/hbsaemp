"""Bambi-free half of the build/fit seam — no MCMC, always runs.

Covers the three things the GUI needs *before* it is willing to pay for
sampling (see docs/prd-integrasi-gui.md):

* `BaseModel.formula` — the resolved formula, for the formula preview.
* `BaseModel.response_name` — labels, without re-parsing the formula.
* `BaseModel.check_data()` — validation + preprocessing, no bambi import.

Also locks the front-door changes that support the GUI: `PriorSpecError` is
reachable from the top level, and `import hbsaemp` does not drag in `app/`.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import hbsaemp as hb
import hbsaemp.models._base  # noqa: F401  — needed for the import-site scan


# ---------------------------------------------------------------------------
# Front door
# ---------------------------------------------------------------------------

def test_prior_spec_error_exported():
    """The error a prior form triggers must be catchable from the front door."""
    assert hb.PriorSpecError is not None
    assert "PriorSpecError" in hb.__all__
    assert issubclass(hb.PriorSpecError, hb.HBSAEError)


def test_prior_spec_error_raised_by_prior():
    with pytest.raises(hb.PriorSpecError):
        hb.Prior("Normal")  # no params


def test_gui_not_imported_eagerly():
    """`import hbsaemp` must not import `hbsaemp.app`.

    Regression guard: the GUI used to be imported eagerly, so one NameError in
    a frontend tab module made the whole backend — and its test suite —
    unimportable.
    """
    code = (
        "import sys, hbsaemp; "
        "sys.exit(1 if 'hbsaemp.app' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_gui_names_still_advertised():
    """Lazy loading must not change the advertised surface."""
    for name in ("App", "AppConfig", "DEFAULT_APP_CONFIG", "launch_app"):
        assert name in hb.__all__
        assert name in dir(hb)


def test_unknown_attribute_still_raises_attribute_error():
    with pytest.raises(AttributeError):
        _ = hb.definitely_not_a_symbol


# ---------------------------------------------------------------------------
# Pre-fit inspection — formula preview, response name, data check
# ---------------------------------------------------------------------------

def test_formula_available_before_fit(data_gaussian: pd.DataFrame):
    """The preview string comes from the model, not a second implementation."""
    model = hb.hbm_gaussian("y", ["x1", "x2"], data_gaussian, area_var="group")

    assert model.is_fitted is False
    assert model.formula == "y ~ x1 + x2 + (1|group)"


def test_formula_respects_explicit_random_effect(data_gaussian: pd.DataFrame):
    """`area_var` must not duplicate an RE the formula already has."""
    model = hb.create_model(
        "y ~ x1 + (1 | group)", family="gaussian",
        data=data_gaussian, group="group",
    )

    assert model.formula.count("group") == 1


def test_response_name_before_fit(data_gaussian: pd.DataFrame):
    model = hb.hbm_gaussian("y", ["x1"], data_gaussian)

    assert model.response_name == "y"


def test_check_data_returns_clean_frame(data_gaussian: pd.DataFrame):
    model = hb.hbm_gaussian("y", ["x1", "x2"], data_gaussian, area_var="group")
    clean = model.check_data()

    assert isinstance(clean, pd.DataFrame)
    assert len(clean) == len(data_gaussian)
    assert model.is_fitted is False


def test_check_data_reports_dropped_rows(data_with_missing: pd.DataFrame):
    """The GUI needs the dropped-row count before it offers to fit."""
    model = hb.hbm_gaussian("y", ["x1"], data_with_missing, area_var="group")
    clean = model.check_data()

    assert len(clean) < len(data_with_missing)
    assert not clean[["y", "x1"]].isna().any().any()


def test_check_data_surfaces_domain_error_without_bambi(data_gaussian: pd.DataFrame):
    """A Beta model on out-of-domain y must fail in check_data(), not in MCMC."""
    df = data_gaussian.assign(y=data_gaussian["y"] + 10.0)  # far outside (0, 1)
    model = hb.hbm_beta("y", ["x1"], df, n="group", deff="group")

    with pytest.raises(hb.DataValidationError):
        model.check_data()


def test_check_data_surfaces_bad_link(data_gaussian: pd.DataFrame):
    """`_pre_fit_checks` runs first, so link errors surface here too."""
    model = hb.hbm_gaussian("y", ["x1"], data_gaussian, link="logit")

    with pytest.raises(ValueError):
        model.check_data()


# ---------------------------------------------------------------------------
# Cheap checks run before the bambi import
# ---------------------------------------------------------------------------

def _explode(self):
    raise ImportError("bambi is not installed (simulated)")


def test_bad_link_beats_missing_bambi(data_gaussian: pd.DataFrame, monkeypatch):
    """`fit()` must report the bad link, not an ImportError.

    The pre-flight checks are deliberately ordered before the bambi import so a
    user without bambi still gets the actionable error. Simulated by making
    `_import_bambi` raise, which is what a bambi-less machine does.
    """
    monkeypatch.setattr(hb.BaseModel, "_import_bambi", _explode)
    model = hb.hbm_gaussian("y", ["x1"], data_gaussian, link="logit")

    with pytest.raises(ValueError, match="logit"):
        model.fit()


def test_missing_trials_beats_missing_bambi(data_binomial: pd.DataFrame, monkeypatch):
    """Same ordering guarantee for BinomialModel's own pre-flight check."""
    monkeypatch.setattr(hb.BaseModel, "_import_bambi", _explode)
    model = hb.hbm_binomial("y", ["x1"], data_binomial, trials="n")
    model._trials_col = None  # simulate the invalid state the guard exists for

    with pytest.raises(ValueError):
        model.fit()


def test_prior_predictive_also_checks_before_import(
    data_gaussian: pd.DataFrame, monkeypatch
):
    """The prior-predictive seam shares the ordering guarantee with `fit()`."""
    monkeypatch.setattr(hb.BaseModel, "_import_bambi", _explode)
    model = hb.hbm_gaussian("y", ["x1"], data_gaussian, link="logit")

    with pytest.raises(ValueError, match="logit"):
        model.prior_predictive_idata()


def test_single_bambi_import_site():
    """`_import_bambi` is the only `import bambi` in the package.

    `_build_bambi_priors` used to carry a second one with a different message;
    the docstring claiming a single contact point is now literally true.
    """
    src = Path(hb.models._base.__file__).read_text(encoding="utf-8")
    assert len(re.findall(r"^\s*import bambi\b", src, re.M)) == 1


def test_check_data_does_not_import_bambi(data_gaussian: pd.DataFrame):
    """check_data() is the cheap gate — it must not pull in bambi."""
    code = (
        "import sys, hbsaemp as hb, pandas as pd, numpy as np;"
        "df = pd.DataFrame({'y': np.arange(10.0), 'x1': np.arange(10.0)});"
        "hb.hbm_gaussian('y', ['x1'], df).check_data();"
        "sys.exit(1 if 'bambi' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
