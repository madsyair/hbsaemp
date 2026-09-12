"""Rejection paths in `BaseModel` — the guards that protect the invariants.

Every branch here raises. They were all verified by hand during the models/
audit but none was locked by a test, which left the package's own safety rails
free to drift: the deprecated-`kind` path in particular is the only
user-facing deprecation in the package and nothing held it in place.

No MCMC anywhere — `_normalize_predict_kind` reads class attributes only, so
it can be exercised on an unfitted model.
"""
from __future__ import annotations

import sys
from typing import Any

import pytest

import hbsaemp as hb
from hbsaemp.models._base import BaseModel
from hbsaemp.models._gaussian import GaussianModel


@pytest.fixture
def gaussian_unfitted(data_gaussian, default_config) -> BaseModel:
    """An unfitted Gaussian model — enough for every guard in this file."""
    return hb.create_model(
        "y ~ x1 + (1|group)", family="gaussian", data=data_gaussian,
        config=default_config,
    )


# Family registry


def test_unknown_family_raises_model_registry_error(data_gaussian, default_config):
    """A subclass built with an unregistered family must fail informatively.

    `create_model()` screens the family first, but a subclass constructed
    directly bypasses that screen — so `BaseModel` has to carry its own guard.
    It reports through `get_family_spec()`, which raises the domain error
    carrying `family` and `registered`, not a bare ValueError.
    """
    with pytest.raises(hb.ModelRegistryError) as exc_info:
        GaussianModel(
            "y ~ x1", "not_a_family", data_gaussian, default_config,
            sampling_var_col=None,
        )
    assert exc_info.value.family == "not_a_family"
    assert "gaussian" in exc_info.value.registered


# Anti-drift: pipeline_fields must have their backing attributes


def test_missing_pipeline_field_attribute_names_the_attribute(gaussian_unfitted):
    """`FAMILY_SPECS[...].pipeline_fields` and the subclass must not drift apart.

    A field declared in the spec without the matching `self._<name>` on the
    instance has to fail immediately and say which attribute is missing —
    otherwise the mismatch surfaces as a confusing KeyError deep in the
    validator.
    """
    del gaussian_unfitted._sampling_var_col

    with pytest.raises(AttributeError) as exc_info:
        gaussian_unfitted._extra_pipeline_kwargs()

    message = str(exc_info.value)
    assert "_sampling_var_col" in message
    assert "pipeline_fields" in message


# predict(kind=...) normalisation


@pytest.mark.parametrize("alias,canonical", [
    ("pps", "response"),
    ("mean", "response_params"),
])
def test_deprecated_kind_alias_warns_and_maps(gaussian_unfitted, alias, canonical):
    """The two legacy aliases keep working, but say so and name the removal."""
    with pytest.warns(FutureWarning, match=BaseModel._ALIAS_REMOVAL_VERSION):
        assert gaussian_unfitted._normalize_predict_kind(alias) == canonical


@pytest.mark.parametrize("kind", ["response", "response_params"])
def test_canonical_kind_passes_through_silently(gaussian_unfitted, kind, recwarn):
    """No warning for the canonical values — only the aliases are deprecated."""
    assert gaussian_unfitted._normalize_predict_kind(kind) == kind
    assert not [w for w in recwarn if issubclass(w.category, FutureWarning)]


def test_removed_kind_linear_rejected(gaussian_unfitted):
    """`kind="linear"` was dropped by Bambi 0.18+; reject before touching idata."""
    with pytest.raises(ValueError, match="removed in Bambi"):
        gaussian_unfitted._normalize_predict_kind("linear")


@pytest.mark.parametrize("kind", ["", "ngawur", "Response", "RESPONSE_PARAMS"])
def test_unsupported_kind_rejected(gaussian_unfitted, kind):
    """Unknown values — including wrong case — never reach Bambi."""
    with pytest.raises(ValueError, match="Unsupported kind"):
        gaussian_unfitted._normalize_predict_kind(kind)


@pytest.mark.parametrize("kind", ["linear", "ngawur"])
def test_kind_errors_are_printable_on_any_console(gaussian_unfitted, kind):
    """Both rejection messages must survive a non-UTF-8 console.

    They once spelled the precision parameter with a Greek kappa, which cp1252
    cannot encode — so `print(e)` raised UnicodeEncodeError instead of showing
    the user which values are valid.
    """
    with pytest.raises(ValueError) as exc_info:
        gaussian_unfitted._normalize_predict_kind(kind)
    str(exc_info.value).encode("cp1252")  # raises UnicodeEncodeError on regression


# Lazy bambi import


def test_import_bambi_error_points_at_the_extra(gaussian_unfitted, monkeypatch):
    """Without bambi installed, say how to install it — not `No module named`.

    `sys.modules[name] = None` is the documented way to make `import name`
    raise ImportError without uninstalling anything.
    """
    monkeypatch.setitem(sys.modules, "bambi", None)

    with pytest.raises(ImportError) as exc_info:
        gaussian_unfitted._import_bambi()

    message = str(exc_info.value)
    assert "hbsaemp[bambi]" in message
    assert "GaussianModel" in message


# link default now lives on BaseModel


@pytest.mark.parametrize("family,expected", [
    ("gaussian", "identity"),
    ("beta", "logit"),
    ("binomial", "logit"),
])
def test_link_defaults_to_the_family_spec(family, expected, request, default_config):
    """`link=None` takes the family default from FAMILY_SPECS, not a subclass constant."""
    data = request.getfixturevalue(f"data_{family}")
    extra: dict[str, Any] = {
        "gaussian": {},
        "beta": {"n": "n", "deff": "deff"},
        "binomial": {"trials": "n"},
    }[family]
    model = hb.create_model(
        "y ~ x1", family=family, data=data, config=default_config, **extra
    )
    assert model._link == expected
    assert model._link == model._default_link
