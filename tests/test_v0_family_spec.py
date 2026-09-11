"""FAMILY_SPECS single-source contract — no Bambi, no MCMC.

Locks the P1 invariant: family metadata (mean param, links, pipeline
fields, backend/preproc family) lives only in `FAMILY_SPECS` and is read by
`BaseModel`. These tests fail loudly if a subclass re-introduces a local
metadata constant or drifts from the spec.
"""
from __future__ import annotations

import pandas as pd
import pytest

import hbsaemp as hb
from hbsaemp._exceptions import DataValidationError
from hbsaemp.models._factory import MODEL_REGISTRY
from hbsaemp.models._family_spec import FAMILY_SPECS

FAMILIES = ["gaussian", "beta", "binomial"]

# Minimum required family kwargs so create_model() succeeds and a later
# _pre_fit_checks() reaches the link check instead of failing earlier on a
# missing required argument.
_REQUIRED_KW: dict[str, dict[str, str]] = {
    "gaussian": {},
    "beta": {"n": "n", "deff": "deff"},
    "binomial": {"trials": "n"},
}


@pytest.fixture
def data_for(data_gaussian, data_beta, data_binomial):
    """Map family -> a fixture DataFrame valid for that family."""
    return {
        "gaussian": data_gaussian,
        "beta": data_beta,
        "binomial": data_binomial,
    }


def _build(family, data_for, default_config, *, link=None, **extra):
    kw = dict(_REQUIRED_KW[family], **extra)
    return hb.create_model(
        "y ~ x1", family, data_for[family],
        link=link, config=default_config, **kw,
    )


# Registry <-> spec consistency

def test_model_registry_matches_family_specs():
    # The sets must be equal: a subset would silently allow an unregistered
    # family to slip through.
    assert set(MODEL_REGISTRY) == set(FAMILY_SPECS)


def test_mean_param_keys_in_spec():
    assert FAMILY_SPECS["gaussian"].mean_param_key == "mu"
    assert FAMILY_SPECS["beta"].mean_param_key == "mu"
    assert FAMILY_SPECS["binomial"].mean_param_key == "p"


# Anti-drift: subclasses must NOT re-declare metadata constants

@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize(
    "attr", ["_MEAN_PARAM_KEY", "_SUPPORTED_LINKS", "_EXTRA_FIELD_NAMES"]
)
def test_subclass_has_no_local_metadata_constant(family, attr):
    """Re-introducing a local constant defeats the single source — forbid it."""
    cls = MODEL_REGISTRY[family]
    assert not hasattr(cls, attr), (
        f"{cls.__name__}.{attr} re-introduced — read FAMILY_SPECS instead."
    )


# BaseModel reads metadata from the spec

@pytest.mark.parametrize("family", FAMILIES)
def test_instance_mean_param_key_reads_spec(family, data_for, default_config):
    m = _build(family, data_for, default_config)
    assert m._mean_param_key == FAMILY_SPECS[family].mean_param_key


@pytest.mark.parametrize("family", FAMILIES)
def test_instance_default_link_reads_spec(family, data_for, default_config):
    # No link= given -> the family's spec default is used.
    m = _build(family, data_for, default_config)
    assert m._link == FAMILY_SPECS[family].default_link


@pytest.mark.parametrize("family", FAMILIES)
def test_backend_and_preproc_family_read_spec(family, data_for, default_config):
    m = _build(family, data_for, default_config)
    assert m._bambi_family() == FAMILY_SPECS[family].bambi_family
    assert m._preproc_family == FAMILY_SPECS[family].preproc_family


# pipeline_fields are backed by real instance attributes

@pytest.mark.parametrize("family", FAMILIES)
def test_pipeline_fields_have_backing_attributes(family, data_for, default_config):
    m = _build(family, data_for, default_config)
    for name in FAMILY_SPECS[family].pipeline_fields:
        assert hasattr(m, f"_{name}"), f"{family}: missing _{name}"
    # And the derived kwargs dict keys match the spec's pipeline_fields.
    assert set(m._extra_pipeline_kwargs()) == set(FAMILY_SPECS[family].pipeline_fields)


# Link validation reads supported_links from the spec

@pytest.mark.parametrize("family", FAMILIES)
def test_unsupported_link_rejected_from_spec(family, data_for, default_config):
    m = _build(family, data_for, default_config, link="not_a_real_link")
    with pytest.raises(ValueError, match="not supported"):
        m._pre_fit_checks()


def test_cross_family_link_rejected(data_for, default_config):
    # "logit" is valid for beta/binomial but NOT gaussian — the spec, not a
    # global list, decides. Confirms per-family link sets are honoured.
    m = _build("gaussian", data_for, default_config, link="logit")
    with pytest.raises(ValueError, match="not supported"):
        m._pre_fit_checks()


@pytest.mark.parametrize("family", FAMILIES)
def test_default_link_passes_pre_fit_checks(family, data_for, default_config):
    # The spec default link must itself be in supported_links.
    m = _build(family, data_for, default_config)
    m._pre_fit_checks()  # must not raise


# Beta squeeze guard (added with P1 — stricter than before)

def test_beta_squeeze_without_n_deff_rejected(data_beta, default_config):
    m = hb.create_model("y ~ x1", "beta", data_beta,
                        squeeze=True, config=default_config)
    with pytest.raises(ValueError, match="squeeze=True"):
        m._pre_fit_checks()


def test_beta_squeeze_with_n_deff_ok(data_beta, default_config):
    m = hb.create_model("y ~ x1", "beta", data_beta,
                        n="n", deff="deff", squeeze=True, config=default_config)
    m._pre_fit_checks()  # squeeze with the precision offset active is allowed


# Behavior callables live in the spec too (P0-1) — no Bambi, direct on the spec

def test_addition_template_only_for_binomial():
    # Only families that wrap the response carry a template; others are None.
    assert FAMILY_SPECS["binomial"].addition_template == "p({response}, {trials_col})"
    assert FAMILY_SPECS["gaussian"].addition_template is None
    assert FAMILY_SPECS["beta"].addition_template is None


def test_addition_template_formats_lhs():
    lhs = FAMILY_SPECS["binomial"].addition_template.format(response="y", trials_col="n")
    assert lhs == "p(y, n)"


def test_response_check_rejects_out_of_domain_beta():
    bad = pd.DataFrame({"y": [0.2, 1.5]})  # 1.5 is outside (0, 1)
    with pytest.raises(DataValidationError):
        FAMILY_SPECS["beta"].response_check(bad, "y", {"squeeze": False})


def test_response_check_rejects_non_integer_binomial():
    bad = pd.DataFrame({"y": [1.5, 2.0]})  # non-integer successes
    with pytest.raises(DataValidationError):
        FAMILY_SPECS["binomial"].response_check(bad, "y", {"trials_col": None})


def test_preprocess_beta_adds_log_phi():
    df = pd.DataFrame({"y": [0.2], "n": [100.0], "deff": [2.0]})
    out = FAMILY_SPECS["beta"].preprocess(
        df.copy(), "y", {"n_col": "n", "deff_col": "deff", "squeeze": False}
    )
    assert "log_phi" in out.columns


def test_preprocess_gaussian_adds_log_sqrt_d():
    df = pd.DataFrame({"y": [5.0], "D": [0.25]})
    out = FAMILY_SPECS["gaussian"].preprocess(df.copy(), "y", {"sampling_var_col": "D"})
    assert "log_sqrt_D" in out.columns


def test_binomial_has_no_preprocess():
    # Binomial needs no offset transform — the spec stores None and the
    # preprocessor skips it (mirrors the response_check=None branch).
    assert FAMILY_SPECS["binomial"].preprocess is None


# offset_col names the column preprocess adds (read by _prepare_mean_data)

def test_offset_col_in_spec():
    assert FAMILY_SPECS["gaussian"].offset_col == "log_sqrt_D"
    assert FAMILY_SPECS["beta"].offset_col == "log_phi"
    assert FAMILY_SPECS["binomial"].offset_col is None


@pytest.mark.parametrize(
    ("family", "df", "ctx"),
    [
        ("gaussian", pd.DataFrame({"y": [5.0], "D": [0.25]}), {"sampling_var_col": "D"}),
        ("beta", pd.DataFrame({"y": [0.2], "n": [100.0], "deff": [2.0]}),
         {"n_col": "n", "deff_col": "deff", "squeeze": False}),
    ],
)
def test_preprocess_adds_exactly_offset_col(family, df, ctx):
    """The spec's offset_col and the column preprocess writes must not drift."""
    out = FAMILY_SPECS[family].preprocess(df.copy(), "y", ctx)
    assert set(out.columns) - set(df.columns) == {FAMILY_SPECS[family].offset_col}
